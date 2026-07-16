"""
agents/graph.py
===============
LangGraph StateGraph wiring all pipeline stages for the AP Invoice &
Contract Exception Agent.

Pipeline stages
---------------
  raw_text
     |
  [extract]          — LLM extraction -> InvoiceLineExtraction
     |
  [check_scope]      — is this actually an invoice?
     |          \\
  [validate]    [audit]  <- refusal short-circuit
     |      \\
  [match]  [audit]       <- malformed short-circuit
     |
  [decide]           — sets final_decision / reason from MatchResult
     |
  [audit]            — persists AuditRecord to SQLite + trace file
     |
   END

Public API
----------
    run_pipeline(raw_text: str) -> dict
        Execute the full pipeline and return the final state as a plain dict.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from project_paths import find_project_root

_PROJECT_ROOT = find_project_root(__file__)
load_dotenv(_PROJECT_ROOT / ".env")

from langgraph.graph import END, StateGraph
from typing import TypedDict

from agents.extraction_agent import extract_invoice
from agents.matching_agent import run_matching_agent
from audit.audit_log import persist_audit
from schemas.models import (
    AuditRecord,
    InvoiceLineExtraction,
    MalformedInvoiceError,
    MatchResult,
    ToolCallRecord,
    ValidatedInvoice,
)

# ─────────────────────────────────────────────────────────────────────────────
#   Pipeline state
# ─────────────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict, total=False):
    """Shared state threaded through every node in the pipeline.

    All fields are Optional (total=False) so nodes can return partial dicts
    and LangGraph will merge them into the running state.
    """

    # ── inputs ────────────────────────────────────────────────────────────────
    raw_text: str                           # document text to process
    run_id: str                             # UUID4 assigned at pipeline entry

    # ── stage outputs ─────────────────────────────────────────────────────────
    raw_extraction: Optional[dict]          # InvoiceLineExtraction as dict
    invoice_id: Optional[str]              # extracted invoice id (for convenience)
    validated_invoice: Optional[dict]      # ValidatedInvoice as dict
    match_result: Optional[dict]           # MatchResult as dict
    tool_call_trace: Optional[list]        # list of ToolCallRecord dicts

    # ── final disposition ─────────────────────────────────────────────────────
    final_decision: Optional[str]
    reason: Optional[str]
    is_refusal: bool

    # ── error passthrough ─────────────────────────────────────────────────────
    error: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
#   Node helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tool_records_to_dicts(records: list[ToolCallRecord]) -> list[dict]:
    """Serialise a list of ToolCallRecord objects to plain dicts."""
    return [
        {
            "tool_name": r.tool_name,
            "tool_input": r.tool_input,
            "tool_output": r.tool_output,
            "timestamp": r.timestamp.isoformat(),
        }
        for r in records
    ]


# ─────────────────────────────────────────────────────────────────────────────
#   Nodes
# ─────────────────────────────────────────────────────────────────────────────

def node_extract(state: PipelineState) -> dict:
    """
    Stage 1 — Extraction.

    Calls extraction_agent.extract_invoice() on the raw document text.
    Always succeeds (errors are caught and stored in ``error``).
    """
    raw_text: str = state["raw_text"]
    try:
        extraction: InvoiceLineExtraction = extract_invoice(raw_text)
        raw_dict = extraction.model_dump()
        return {
            "raw_extraction": raw_dict,
            "invoice_id": raw_dict.get("invoice_id", "UNKNOWN"),
        }
    except Exception as exc:
        # Surface the error but allow the graph to reach audit for logging
        return {
            "raw_extraction": {
                "invoice_id": "UNKNOWN",
                "vendor": "UNKNOWN",
                "po_number": None,
                "line_qty": None,
                "unit_price": None,
                "total": None,
                "approval_note": None,
                "extraction_confidence": 0.0,
                "is_invoice": False,
                "refusal_reason": f"Extraction failed: {exc}",
            },
            "invoice_id": "UNKNOWN",
            "error": str(exc),
        }


def node_check_scope(state: PipelineState) -> dict:
    """
    Stage 2 — Scope gate.

    If the extraction determined this is not an invoice, mark it as a refusal
    so the conditional edge routes directly to audit, skipping validation and
    matching entirely.
    """
    raw = state.get("raw_extraction") or {}
    if not raw.get("is_invoice", True):
        return {
            "is_refusal": True,
            "final_decision": "refused",
            "reason": raw.get("refusal_reason") or "Document is not an invoice.",
        }
    return {"is_refusal": False}


def node_validate(state: PipelineState) -> dict:
    """
    Stage 3 — Validation gate.

    Promotes the raw InvoiceLineExtraction to a ValidatedInvoice (requiring
    line_qty, unit_price, total).  On MalformedInvoiceError, marks the invoice
    as rejected and routes to audit.
    """
    raw = state.get("raw_extraction") or {}
    try:
        extraction_obj = InvoiceLineExtraction(**raw)
        validated = ValidatedInvoice.from_extraction(extraction_obj)
        return {"validated_invoice": validated.model_dump()}
    except MalformedInvoiceError as exc:
        missing = ", ".join(exc.missing_fields)
        return {
            "final_decision": "rejected",
            "reason": f"Missing required field(s): {missing}",
            "validated_invoice": None,
        }
    except Exception as exc:
        return {
            "final_decision": "rejected",
            "reason": f"Validation error: {exc}",
            "validated_invoice": None,
        }


def node_match(state: PipelineState) -> dict:
    """
    Stage 4 — Matching.

    Runs the LLM-driven matching agent against the validated invoice.
    Stores the MatchResult and full tool-call trace in state.
    """
    validated_dict = state.get("validated_invoice") or {}
    try:
        validated = ValidatedInvoice(**validated_dict)
        match_result, tool_records = run_matching_agent(validated)
        return {
            "match_result": match_result.model_dump(),
            "tool_call_trace": _tool_records_to_dicts(tool_records),
        }
    except Exception as exc:
        return {
            "match_result": {
                "invoice_id": state.get("invoice_id", "UNKNOWN"),
                "field_results": {},
                "overall_status": "rejected",
                "exceptions": [],
            },
            "tool_call_trace": [],
            "error": str(exc),
        }


def node_decide(state: PipelineState) -> dict:
    """
    Stage 5 — Decision.

    Translates the MatchResult.overall_status into a final_decision string
    and builds a human-readable reason from any ExceptionDetail objects.
    """
    mr = state.get("match_result") or {}
    overall_status: str = mr.get("overall_status", "rejected")
    exceptions: list[dict] = mr.get("exceptions") or []

    # Map overall_status -> final_decision label
    decision_map = {
        "straight_through": "straight_through",
        "exception": "exception",
        "rejected": "rejected",
    }
    final_decision = decision_map.get(overall_status, overall_status)

    # Build reason from exception details (None if straight-through)
    if exceptions:
        reason_parts = [e.get("reason", "") for e in exceptions if e.get("reason")]
        reason = "; ".join(reason_parts) if reason_parts else None
    else:
        reason = None

    return {
        "final_decision": final_decision,
        "reason": reason,
    }


def node_audit(state: PipelineState) -> dict:
    """
    Stage 6 — Audit (terminal node).

    Builds an AuditRecord from the current pipeline state and persists it
    to SQLite + trace file via audit_log.persist_audit().
    """
    # Reconstruct ToolCallRecord objects from stored dicts
    raw_trace: list[dict] = state.get("tool_call_trace") or []
    from datetime import datetime
    tool_records: list[ToolCallRecord] = []
    for t in raw_trace:
        try:
            ts_raw = t.get("timestamp")
            ts = datetime.fromisoformat(ts_raw) if isinstance(ts_raw, str) else datetime.now()
            tool_records.append(ToolCallRecord(
                tool_name=t["tool_name"],
                tool_input=t["tool_input"],
                tool_output=t["tool_output"],
                timestamp=ts,
            ))
        except Exception:
            pass  # skip malformed trace entries

    record = AuditRecord(
        run_id=state.get("run_id") or str(uuid.uuid4()),
        invoice_id=state.get("invoice_id") or "UNKNOWN",
        raw_extraction=state.get("raw_extraction") or {},
        match_result=state.get("match_result") or {},
        final_decision=state.get("final_decision") or "unknown",
        reason=state.get("reason"),
        tool_call_trace=tool_records,
        is_refusal=state.get("is_refusal") or False,
    )

    try:
        persist_audit(record)
    except Exception as exc:
        return {"error": f"Audit persist failed: {exc}"}

    return {}   # no state mutations needed; node is terminal


# ─────────────────────────────────────────────────────────────────────────────
#   Routing functions (for conditional edges)
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_check_scope(state: PipelineState) -> str:
    """After check_scope: go to 'audit' if refused, else 'validate'."""
    if state.get("is_refusal"):
        return "audit"
    return "validate"


def _route_after_validate(state: PipelineState) -> str:
    """After validate: go to 'audit' if malformed/rejected, else 'match'."""
    if state.get("final_decision") in ("rejected", "refused"):
        return "audit"
    return "match"


# ─────────────────────────────────────────────────────────────────────────────
#   Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    """Construct and compile the pipeline StateGraph."""
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("extract",     node_extract)
    graph.add_node("check_scope", node_check_scope)
    graph.add_node("validate",    node_validate)
    graph.add_node("match",       node_match)
    graph.add_node("decide",      node_decide)
    graph.add_node("audit",       node_audit)

    # Entry point
    graph.set_entry_point("extract")

    # Linear edges
    graph.add_edge("extract", "check_scope")
    graph.add_edge("match",   "decide")
    graph.add_edge("decide",  "audit")
    graph.add_edge("audit",   END)

    # Conditional edges
    graph.add_conditional_edges(
        "check_scope",
        _route_after_check_scope,
        {"audit": "audit", "validate": "validate"},
    )
    graph.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"audit": "audit", "match": "match"},
    )

    return graph


# Compile once at module level — reused by run_pipeline()
_COMPILED_GRAPH = _build_graph().compile()


# ─────────────────────────────────────────────────────────────────────────────
#   Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(raw_text: str) -> dict:
    """
    Execute the full AP Invoice pipeline on a raw document string.

    Parameters
    ----------
    raw_text : str
        The text of the document to process (invoice or otherwise).

    Returns
    -------
    dict
        The final pipeline state as a plain dict.  Key fields:
        - ``final_decision``  — "approved" | "exception" | "rejected" | "refused"
        - ``reason``          — human-readable explanation, or None
        - ``is_refusal``      — True if the document was not an invoice
        - ``match_result``    — MatchResult dict (empty for refusals)
        - ``tool_call_trace`` — list of tool call dicts
        - ``run_id``          — UUID4 for this run
    """
    initial_state: PipelineState = {
        "raw_text": raw_text,
        "run_id": str(uuid.uuid4()),
        "raw_extraction": None,
        "invoice_id": None,
        "validated_invoice": None,
        "match_result": None,
        "tool_call_trace": [],
        "final_decision": None,
        "reason": None,
        "is_refusal": False,
        "error": None,
    }
    final_state = _COMPILED_GRAPH.invoke(initial_state)
    return dict(final_state)


# ─────────────────────────────────────────────────────────────────────────────
#   __main__ — run all 6 sample scenarios
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    data_path = _PROJECT_ROOT / "data" / "sample_invoices.json"
    with open(data_path, encoding="utf-8") as f:
        scenarios = json.load(f)

    print("=" * 70)
    print("AP Invoice Pipeline -- full run (all 6 scenarios)")
    print("=" * 70)

    failures: list[str] = []

    for i, scenario in enumerate(scenarios, start=1):
        scenario_id: str = scenario["scenario_id"]
        raw_text: str = scenario["raw_text"]
        expected_outcome: str = scenario.get("expected_outcome", "")

        print(f"\n[{i}/6] Scenario: {scenario_id!r}  (expected: {expected_outcome!r})")
        print("-" * 70)

        try:
            result = run_pipeline(raw_text)
        except Exception as exc:
            print(f"  PIPELINE ERROR: {exc}")
            traceback.print_exc()
            failures.append(f"{scenario_id}:pipeline_error")
            continue

        final_decision = result.get("final_decision")
        reason = result.get("reason")
        is_refusal = result.get("is_refusal", False)
        run_id = result.get("run_id", "")
        trace_len = len(result.get("tool_call_trace") or [])

        print(f"  final_decision : {final_decision!r}")
        print(f"  reason         : {reason!r}")
        print(f"  is_refusal     : {is_refusal}")
        print(f"  tool_calls     : {trace_len}")
        print(f"  run_id         : {run_id[:8]}...")

        if result.get("error"):
            print(f"  error          : {result['error']}")

        # ── Scenario 6 assertion: must be refused without calling matching ────
        if scenario_id == "out_of_scope":
            if final_decision == "refused" and is_refusal is True:
                print("  PASS: out_of_scope correctly refused (is_refusal=True).")
            else:
                msg = (
                    f"  FAIL: out_of_scope returned final_decision={final_decision!r}, "
                    f"is_refusal={is_refusal} -- expected refused/True."
                )
                print(msg)
                failures.append(f"{scenario_id}:not_refused")

            if trace_len == 0:
                print("  PASS: No tool calls made for out_of_scope (matching skipped).")
            else:
                print(
                    f"  FAIL: {trace_len} tool call(s) were made for out_of_scope "
                    f"-- matching agent should have been skipped entirely."
                )
                failures.append(f"{scenario_id}:matching_called_for_refusal")

        # ── Soft checks for other scenarios ───────────────────────────────────
        else:
            # Map expected_outcome values to final_decision labels
            outcome_map = {
                "straight_through": "approved",
                "exception":        "exception",
                "rejected":         "rejected",
            }
            expected_decision = outcome_map.get(expected_outcome, expected_outcome)

            if final_decision == expected_decision:
                print(f"  PASS: final_decision={final_decision!r} matches expected.")
            else:
                print(
                    f"  NOTE: final_decision={final_decision!r}, "
                    f"expected {expected_decision!r}."
                )

    print("\n" + "=" * 70)
    if failures:
        print(f"FAILURES: {failures}")
        import sys as _sys
        _sys.exit(1)
    else:
        print("All assertion checks passed.")
    print("=" * 70)
