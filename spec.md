# AP Invoice & Contract Exception Agent — Technical Specification

**Business owner:** Controller · **Function:** Finance / Procurement
**Pattern:** Extraction + validation + exception routing + audit

---

## 1. Problem Statement

Accounts payable teams manually match every incoming invoice against its purchase order (PO) and contract terms. The majority of invoices are clean and could be paid automatically; the exceptions — price mismatches, missing approvals, off-contract terms — are where money leaks, and they are easy to miss at volume. This system automates the matching, straight-through-processes clean invoices, and routes only genuine exceptions to a human, with the exact reason stated.

---

## 2. Business Requirements

| # | Requirement |
|---|---|
| 1 | Extract invoice fields — vendor, PO number, line items, quantities, unit prices, total — into a validated schema. |
| 2 | Match against the PO and contract: quantities, unit prices, totals, and approval thresholds. |
| 3 | Straight-through-process invoices that fully match and mark them for payment. |
| 4 | Route exceptions (price variance, missing approval, off-contract term, unknown vendor) to a human with the specific reason. |
| 5 | Reject and flag malformed extractions (a missing required field) rather than paying on partial data. |
| 6 | Persist a field-level audit trail for every invoice, matched or exception. |

**Target user:** AP clerk and controller.
**Success metric (KPI):** Straight-through-processing rate, exception catch rate, days-to-pay, invoices per FTE.

---

## 3. Architecture

```
raw invoice text (pasted, or extracted from uploaded PDF/DOCX)
      │
      ▼
[extract]  ── Extraction Agent (LLM). Pulls structured fields.
      │        Flags non-invoice documents (is_invoice = False).
      ▼
[check_scope]  ── Not an invoice? → REFUSED. Stop here.
      │
      ▼
[validate]  ── Deterministic Pydantic check (not an agent).
      │        Missing required field? → REJECTED. Stop here.
      ▼
[match]  ── Matching Agent (LLM + tools). Decides which tools to
      │      call and in what order: lookup_po → check_vendor_match
      │      → compare_field (×3) → check_approval.
      ▼
[decide]  ── Deterministic (not an agent). straight_through or
      │       exception, with cited reasons per field.
      ▼
[audit]  ── Full trace (extraction, match result, tool calls)
             persisted to SQLite + JSON. Terminal node, every invoice.
```

Orchestrated as a LangGraph `StateGraph` with conditional edges: `extract → check_scope → (refused | validate) → (rejected | match) → decide → audit`.

---

## 4. Agents (exactly two — everything else is deterministic by design)

### 4.1 Extraction Agent (`agents/extraction_agent.py`)
- **Input:** raw invoice text.
- **Output:** `InvoiceLineExtraction` (Pydantic model).
- **Behavior:**
  - Judges whether the document is an invoice at all (`is_invoice`, `refusal_reason`).
  - Extracts: `invoice_id`, `vendor`, `po_number`, `line_qty`, `unit_price`, `total`, `approval_note`.
  - Never fabricates a missing numeric field — leaves it `null`, reflected in `extraction_confidence`.
  - Treats the invoice's own free-text note as **untrusted data**, never as an instruction (prompt-injection defense).
  - Single LLM call via OpenRouter, structured output.

### 4.2 Matching Agent (`agents/matching_agent.py`)
- **Input:** `ValidatedInvoice`.
- **Output:** `(MatchResult, list[ToolCallRecord])`.
- **Behavior:** An LLM with `bind_tools([...])` that decides which tools to call and in what order — not a hardcoded function chain. Available tools:

| Tool | Purpose |
|---|---|
| `lookup_po(po_number)` | Fetches the PO/contract record from SQLite. |
| `check_vendor_match(invoice_vendor, po_vendor)` | Case-insensitive vendor comparison. |
| `compare_field(field_name, invoice_value, po_value)` | Compares qty, price, total; returns delta + citation. |
| `check_approval(total, approval_required_over, approved_by)` | Checks approval requirement and whether it's on file. Only the PO record's `approved_by` counts — invoice-text approval claims are ignored. |

Every tool call (name, input, output, timestamp) is logged as a `ToolCallRecord` — this is what the **Agent Trace** tab displays, proving step-by-step reasoning rather than a canned answer.

### 4.3 Explicitly NOT agents (deterministic by design)
- **Validation** — Pydantic schema check. A missing required field always rejects; no LLM judgment involved on purpose.
- **Decide** — Sets `final_decision` from `MatchResult.overall_status`. Pure mapping, no reasoning.
- **Audit** — Persists the record. No reasoning.

This separation is intentional: judgment calls (is this an invoice? which checks are needed?) use an LLM; everything with a single correct rule-following answer does not.

---

## 5. Data Model (Pydantic v2, `schemas/models.py`)

| Model | Purpose |
|---|---|
| `InvoiceLineExtraction` | Raw extraction output. Numeric fields Optional. Includes `is_invoice`, `refusal_reason`. |
| `ValidatedInvoice` | Post-validation; numeric fields required. Raises `MalformedInvoiceError` if incomplete. |
| `POContractRecord` | vendor, contract_id, line_qty, unit_price, expected_total, approval_required_over, approved_by. |
| `MatchResult` | overall_status, field_results (per-field match/variance/missing), exceptions (`ExceptionDetail` list). |
| `ExceptionDetail` | reason, citation (source of truth quoted), field. |
| `ToolCallRecord` | tool_name, tool_input, tool_output, timestamp. |
| `AuditRecord` | run_id, invoice_id, timestamp, raw_extraction, match_result, tool_call_trace, final_decision, reason, is_refusal. |

---

## 6. Data Stores

- **PO/contract records:** SQLite table `po_contracts` (`store/po_store.py`). Seeded once from `data/pos_contracts.json`; live-editable thereafter via the "Manage POs & Contracts" UI tab. Not a vector store — PO lookup is an exact match on `po_number`, which is the correct use case for a relational table, not similarity search.
- **Audit log:** SQLite table `audit_log` + per-run JSON trace files in `audit/traces/{run_id}.json` (`audit/audit_log.py`).

No ChromaDB or other vector database is used in this system. (Fuzzy vendor-name matching via embeddings is noted as a future enhancement — see §9 — but is not implemented.)

---

## 7. Governance & Safety Requirements Met

| Requirement | Implementation |
|---|---|
| Human gate on risk | Any exception (variance, missing approval, unknown vendor) is held — never scheduled for payment — pending human review. |
| Citations | Every `ExceptionDetail` includes a `citation` string quoting the PO/contract field it was checked against. |
| Refusal on out-of-scope | Non-invoice documents set `is_invoice = False` and are refused before reaching the matching agent. |
| Prompt-injection resistance | Invoice free text (`approval_note`) is always treated as data. Approval status is sourced only from the PO record's `approved_by`, never from claims inside the invoice. |
| Audit log | Every invoice — straight-through, exception, rejected, or refused — is persisted with a full field-level and tool-call-level trace. |
| Fairness | Not applicable — invoice/PO matching involves no protected attributes about people. Noted explicitly rather than omitted silently. |

---

## 8. Evaluation Suite (`eval/`)

Six scenarios (`eval/scenarios.py`), each checked against four criteria (`eval/run_eval.py`):

| Scenario | Expected outcome | Tests |
|---|---|---|
| `clean` | straight_through | Happy path, correct tool sequence. |
| `unit_price_variance` | exception | Price mismatch detection + citation. |
| `missing_approval` | exception | Approval-threshold logic. |
| `malformed` | rejected | No fabrication on missing required field. |
| `prompt_injection` | exception | Injected instruction ("skip checks") has zero effect on outcome. |
| `out_of_scope` | refused | Non-invoice document refused before matching. |

**Four checks per scenario:**
1. **Task completion** — correct `final_decision`.
2. **Trace correctness** — stated reason is accurate (keyword-based, not brittle exact-string match).
3. **Tool-call accuracy** — the matching agent called the expected tools.
4. **Governance check** — specific to `prompt_injection`: confirms the injected instruction did not change the outcome.

Current result: 6/6 scenarios passing across all four checks.

**Business KPIs** (from the audit log, not the eval suite): straight-through rate, exceptions caught (count/rate).

---

## 9. Known Limitations & Next Steps

1. **Extraction depends on text quality.** A badly OCR'd or garbled invoice could lower `extraction_confidence` and produce a false rejection. No OCR step is currently built for scanned/image-only PDFs — text-based PDF/DOCX upload is supported (`utils/file_parser.py`), scanned documents are explicitly out of scope for now.
2. **Vendor matching is exact-match only.** "Acme Supply Co." vs "Acme Supplies" reads as a mismatch. **Priority fix:** embedding-based fuzzy vendor-name matching as a fallback only when exact match fails — PO lookup itself remains deterministic and exact.
3. **No duplicate-invoice detection.** The same invoice ID could theoretically be processed twice; nothing currently checks history before processing.
4. **Flat approval threshold.** One dollar threshold per PO; no tiered approval chains by amount.
5. **Eval suite covers anticipated failure modes only,** not novel adversarial extraction attacks against the extraction step itself.
6. **No production authentication** on the PO/contract management UI.

---

## 10. Tech Stack

Python · LangGraph · LangChain (tool-calling agents) · Pydantic v2 · OpenRouter (LLM provider) · Streamlit (UI) · SQLite (PO/contract store, audit log) · pdfplumber / python-docx (file input parsing).

---

## 11. Folder Structure

```
ap-invoice-agent/
├── data/                    # seed data (POs, sample invoices)
├── schemas/models.py        # Pydantic models
├── tools/matching_tools.py  # LangChain tools for the matching agent
├── agents/
│   ├── extraction_agent.py
│   ├── matching_agent.py
│   └── graph.py             # LangGraph orchestration
├── store/po_store.py        # PO/contract SQLite CRUD
├── audit/audit_log.py       # audit log + trace persistence
├── eval/
│   ├── scenarios.py
│   └── run_eval.py
├── ui/app.py                # Streamlit app (4 tabs)
├── utils/file_parser.py     # PDF/DOCX text extraction
├── requirements.txt
└── README.md
```