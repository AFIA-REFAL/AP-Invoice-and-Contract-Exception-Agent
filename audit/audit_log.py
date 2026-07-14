"""
audit/audit_log.py
==================
Persistence layer for the AP Invoice & Contract Exception Agent audit trail.

Every completed pipeline run — including refusals — should call
:func:`persist_audit` so that the decision, its evidence, and the complete
tool-call trace are stored in two complementary formats:

1. **SQLite** (``audit/audit.db``) — row-per-run, queryable by any SQL tool.
2. **JSON trace file** (``audit/traces/{run_id}.json``) — human-readable,
   pretty-printed, includes the full tool-call-level breakdown.

Public API
----------
    persist_audit(record: AuditRecord) -> None
        Insert or replace a run record.

    get_all_audit_records() -> list[dict]
        Return all records ordered by timestamp descending.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ── path setup ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from schemas.models import AuditRecord  # noqa: E402

# ── directory / file paths ────────────────────────────────────────────────────
_AUDIT_DIR = Path(__file__).resolve().parent          # …/audit/
_TRACES_DIR = _AUDIT_DIR / "traces"
_DB_PATH = _AUDIT_DIR / "audit.db"

# ─────────────────────────────────────────────────────────────────────────────
#   SQLite helpers
# ─────────────────────────────────────────────────────────────────────────────

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    run_id          TEXT PRIMARY KEY,
    invoice_id      TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    raw_extraction  TEXT NOT NULL,
    match_result    TEXT NOT NULL,
    tool_call_trace TEXT NOT NULL,
    final_decision  TEXT NOT NULL,
    reason          TEXT,
    is_refusal      INTEGER NOT NULL DEFAULT 0
);
"""


def _get_connection() -> sqlite3.Connection:
    """Open (or create) the audit SQLite database and ensure the schema exists."""
    _AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")   # safe for concurrent readers
    conn.execute(_CREATE_TABLE_SQL)
    conn.commit()
    return conn


def _serialise(obj: Any) -> str:
    """JSON-serialise *obj*, handling datetime objects."""
    return json.dumps(obj, default=_json_default)


def _json_default(obj: Any) -> Any:
    """Fallback serialiser for types that ``json.dumps`` cannot handle."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


# ─────────────────────────────────────────────────────────────────────────────
#   Trace file writer
# ─────────────────────────────────────────────────────────────────────────────

def _write_trace_file(record: AuditRecord) -> Path:
    """
    Write a pretty-printed JSON trace to ``audit/traces/{run_id}.json``.

    The file includes:
    - All scalar fields from the record.
    - ``raw_extraction`` dict.
    - ``match_result`` dict.
    - ``tool_call_trace`` as a list of dicts (with ISO timestamp strings).

    Returns the Path of the written file.
    """
    _TRACES_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = _TRACES_DIR / f"{record.run_id}.json"

    # Build a clean, fully-serialisable dict
    trace_doc: dict[str, Any] = {
        "run_id": record.run_id,
        "invoice_id": record.invoice_id,
        "timestamp": record.timestamp.isoformat(),
        "final_decision": record.final_decision,
        "reason": record.reason,
        "is_refusal": record.is_refusal,
        "raw_extraction": record.raw_extraction,
        "match_result": record.match_result,
        "tool_call_trace": [
            {
                "tool_name": tc.tool_name,
                "tool_input": tc.tool_input,
                "tool_output": tc.tool_output,
                "timestamp": tc.timestamp.isoformat(),
            }
            for tc in record.tool_call_trace
        ],
    }

    with open(trace_path, "w", encoding="utf-8") as fh:
        json.dump(trace_doc, fh, indent=2, default=_json_default)

    return trace_path


# ─────────────────────────────────────────────────────────────────────────────
#   Public API
# ─────────────────────────────────────────────────────────────────────────────

def persist_audit(record: AuditRecord) -> None:
    """
    Persist an audit record to both SQLite and a JSON trace file.

    Uses ``INSERT OR REPLACE`` so re-running a pipeline step with the same
    ``run_id`` is idempotent (the row is updated rather than erroring).

    Parameters
    ----------
    record : AuditRecord
        The fully-populated audit record for a pipeline run.

    Side-effects
    ------------
    - Writes / updates a row in ``audit/audit.db :: audit_log``.
    - Creates ``audit/traces/{run_id}.json``.
    """
    # Serialise compound fields to JSON text for the DB columns
    raw_extraction_json = _serialise(record.raw_extraction)
    match_result_json = _serialise(record.match_result)
    tool_call_trace_json = _serialise(
        [
            {
                "tool_name": tc.tool_name,
                "tool_input": tc.tool_input,
                "tool_output": tc.tool_output,
                "timestamp": tc.timestamp.isoformat(),
            }
            for tc in record.tool_call_trace
        ]
    )

    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO audit_log (
                run_id, invoice_id, timestamp,
                raw_extraction, match_result, tool_call_trace,
                final_decision, reason, is_refusal
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.run_id,
                record.invoice_id,
                record.timestamp.isoformat(),
                raw_extraction_json,
                match_result_json,
                tool_call_trace_json,
                record.final_decision,
                record.reason,
                1 if record.is_refusal else 0,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    # Write companion JSON trace file
    _write_trace_file(record)


def get_all_audit_records() -> list[dict]:
    """
    Return all audit records ordered by timestamp descending (newest first).

    Compound JSON columns (``raw_extraction``, ``match_result``,
    ``tool_call_trace``) are deserialised back to Python dicts/lists.

    Returns
    -------
    list[dict]
        Each dict has the same keys as the ``audit_log`` table, with
        ``is_refusal`` returned as a Python ``bool``.
    """
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC"
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    results: list[dict] = []
    for row in rows:
        d = dict(row)
        # Deserialise JSON text columns back to native Python objects
        for col in ("raw_extraction", "match_result", "tool_call_trace"):
            try:
                d[col] = json.loads(d[col])
            except (json.JSONDecodeError, TypeError):
                pass  # leave as raw string if somehow malformed
        # Normalise SQLite integer bool
        d["is_refusal"] = bool(d.get("is_refusal", 0))
        results.append(d)

    return results


def clear_audit_log() -> None:
    """Delete all rows from the audit_log table and remove JSON trace files.

    This is intended for demo/reset purposes only.
    """
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM audit_log;")
        conn.commit()
    finally:
        conn.close()

    # Remove trace files
    if _TRACES_DIR.exists():
        for p in _TRACES_DIR.glob("*.json"):
            try:
                p.unlink()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
#   Public API exports
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "persist_audit",
    "get_all_audit_records",
    "clear_audit_log",
]


# ─────────────────────────────────────────────────────────────────────────────
#   __main__ — smoke test: persist + query round-trip (no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uuid
    from datetime import timezone

    from schemas.models import ToolCallRecord

    print("=" * 60)
    print("audit_log.py -- smoke test (no LLM)")
    print("=" * 60)

    # ── Build a synthetic AuditRecord ─────────────────────────────────────────
    run_id = str(uuid.uuid4())
    ts = datetime.now(tz=timezone.utc)

    tool_records = [
        ToolCallRecord(
            tool_name="lookup_po",
            tool_input={"po_number": "PO-5567"},
            tool_output={
                "vendor": "Acme Supplies",
                "contract_id": "MSA-ACME-2024",
                "line_qty": 120,
                "unit_price": 38.0,
                "expected_total": 11400.0,
                "approval_required_over": 10000,
                "approved_by": None,
            },
            timestamp=ts,
        ),
        ToolCallRecord(
            tool_name="compare_field",
            tool_input={"field_name": "unit_price", "invoice_value": 42.0, "po_value": 38.0},
            tool_output={
                "field": "unit_price",
                "status": "variance",
                "delta": 4.0,
                "citation": "Per contract, unit_price is 38.0",
            },
            timestamp=ts,
        ),
    ]

    record = AuditRecord(
        run_id=run_id,
        invoice_id="INV-SMOKE-001",
        timestamp=ts,
        raw_extraction={
            "invoice_id": "INV-SMOKE-001",
            "vendor": "Acme Supplies",
            "po_number": "PO-5567",
            "line_qty": 120,
            "unit_price": 42.0,
            "total": 5040.0,
            "approval_note": None,
            "extraction_confidence": 0.75,
            "is_invoice": True,
            "refusal_reason": None,
        },
        match_result={
            "invoice_id": "INV-SMOKE-001",
            "field_results": {
                "unit_price": {
                    "invoice_value": 42.0,
                    "po_value": 38.0,
                    "status": "variance",
                    "delta": 4.0,
                }
            },
            "overall_status": "exception",
            "exceptions": [
                {
                    "reason": "unit_price variance: invoice has 42.0, contract specifies 38.0 (delta +4.0)",
                    "citation": "Per contract, unit_price is 38.0",
                    "field": "unit_price",
                }
            ],
        },
        final_decision="returned_for_revision",
        reason="Unit price exceeds contracted rate by $4.00.",
        tool_call_trace=tool_records,
        is_refusal=False,
    )

    # ── persist ───────────────────────────────────────────────────────────────
    print(f"\nPersisting record run_id={run_id[:8]}...")
    persist_audit(record)
    trace_file = _TRACES_DIR / f"{run_id}.json"
    assert trace_file.exists(), f"Trace file not created: {trace_file}"
    print(f"Trace file written: {trace_file}")

    # ── query ─────────────────────────────────────────────────────────────────
    print("\nQuerying all records...")
    all_records = get_all_audit_records()
    assert len(all_records) >= 1, "Expected at least 1 record after insert"

    # Find our inserted record
    match = next((r for r in all_records if r["run_id"] == run_id), None)
    assert match is not None, f"Inserted record {run_id} not found in query results"

    # Verify round-trip fidelity
    assert match["invoice_id"] == "INV-SMOKE-001"
    assert match["final_decision"] == "returned_for_revision"
    assert match["is_refusal"] is False
    assert isinstance(match["match_result"], dict), "match_result should be deserialised to dict"
    assert isinstance(match["tool_call_trace"], list), "tool_call_trace should be deserialised to list"
    assert len(match["tool_call_trace"]) == 2, "Expected 2 tool call records"
    assert match["tool_call_trace"][0]["tool_name"] == "lookup_po"
    assert match["tool_call_trace"][1]["tool_name"] == "compare_field"
    print(f"Round-trip assertions passed ({len(all_records)} total records in DB).")

    # ── inspect trace file contents ───────────────────────────────────────────
    with open(trace_file, encoding="utf-8") as fh:
        trace_contents = json.load(fh)
    assert trace_contents["run_id"] == run_id
    assert len(trace_contents["tool_call_trace"]) == 2
    assert trace_contents["tool_call_trace"][1]["tool_output"]["citation"] == "Per contract, unit_price is 38.0"
    print("Trace file contents verified (citation preserved verbatim).")

    # ── also test a refusal record ────────────────────────────────────────────
    refusal_run_id = str(uuid.uuid4())
    refusal_record = AuditRecord(
        run_id=refusal_run_id,
        invoice_id="UNKNOWN",
        timestamp=datetime.now(tz=timezone.utc),
        raw_extraction={
            "invoice_id": "UNKNOWN",
            "vendor": "UNKNOWN",
            "po_number": None,
            "line_qty": None,
            "unit_price": None,
            "total": None,
            "approval_note": None,
            "extraction_confidence": 0.0,
            "is_invoice": False,
            "refusal_reason": "Document is an HR memorandum, not an invoice.",
        },
        match_result={},
        final_decision="refused",
        reason="Document is not an invoice.",
        tool_call_trace=[],
        is_refusal=True,
    )
    persist_audit(refusal_record)
    refusal_rows = [r for r in get_all_audit_records() if r["run_id"] == refusal_run_id]
    assert len(refusal_rows) == 1
    assert refusal_rows[0]["is_refusal"] is True
    assert refusal_rows[0]["tool_call_trace"] == []
    refusal_trace = _TRACES_DIR / f"{refusal_run_id}.json"
    assert refusal_trace.exists()
    print("Refusal record persisted and verified.")

    print("\n" + "=" * 60)
    print("PASS: All smoke-test assertions passed.")
    print(f"DB location  : {_DB_PATH}")
    print(f"Traces dir   : {_TRACES_DIR}")
    print("=" * 60)
