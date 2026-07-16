from __future__ import annotations

import html
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    if (_candidate / "project_paths.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from project_paths import find_project_root

# Project path setup
PROJECT_ROOT = find_project_root(__file__)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from agents.graph import run_pipeline
from audit.audit_log import get_all_audit_records
from eval.run_eval import run_all_evals
from store.po_store import add_po, delete_po, get_all_pos

st.set_page_config(
    page_title="AP Autopilot",
    page_icon="AP",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
    --bg: #000000; /* pure black */
    --ink: #ffffff; /* body text */
    --ink-dim: #e8e8e8; /* near-white for readability */
    --muted: #aaaaaa; /* citation / secondary text */
    --line: #2a2a2a; /* borders/dividers */
    --neon-green: #39ff14;
    --neon-amber: #ffae00;
    --neon-red: #ff3131;
    --neon-cyan: #00f0ff;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stSidebar"], [data-testid="stHeader"] {
    background: var(--bg) !important;
    color: var(--ink) !important;
    font-family: 'Inter', 'JetBrains Mono', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
}

/* Remove glows, gradients and shadows from layout containers */
.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 24px;
    background: var(--bg);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 14px;
}

.top-title { margin: 0; color: #ffffff; font-size: 1.5rem; font-weight:700; text-shadow: 0 0 12px #00e5ff, 0 0 24px #00e5ff33; }
.top-subtitle { margin: 4px 0 0 0; color: var(--ink-dim); font-size: 0.92rem; }

.kpis { display: flex; gap: 12px; }

.kpi-card {
    min-width: 165px;
    background: transparent;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 10px 12px;
}

.kpi-val { margin: 0; color: var(--neon-cyan); font-family: 'JetBrains Mono', monospace; font-size: 1.55rem; font-weight: 800; text-shadow: 0 0 8px currentColor; }
.kpi-label { margin: 0; color: var(--ink-dim); font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.06em; }

[data-testid="stTabs"] button { color: var(--ink-dim) !important; font-weight:700; outline: none !important; box-shadow: none !important; border: none !important; }
[data-testid="stTabs"] button:focus, [data-testid="stTabs"] button:focus-visible { outline: none !important; box-shadow: none !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--ink) !important; border-bottom: 2px solid var(--line) !important; }

/* Neon highlight utility classes (use sparingly) */
.neon-green { color: var(--neon-green) !important; font-weight: 700; text-shadow: none !important; }
.neon-amber { color: var(--neon-amber); text-shadow: 0 0 8px currentColor; }
.neon-red { color: var(--neon-red) !important; font-weight: 700; text-shadow: none !important; }
.neon-cyan { color: var(--neon-cyan); text-shadow: 0 0 8px currentColor; }

.badge { display:inline-block; padding:8px 14px; border-radius:999px; font-weight:700; letter-spacing:0.03em; margin-bottom:12px; background:transparent; border:1px solid var(--line); }
.badge-ok { color: var(--neon-green); background: transparent; }
.badge-exception { color: var(--neon-amber); background: transparent; }
.badge-rejected { color: var(--neon-red); background: transparent; }
.badge-refused { color: var(--neon-cyan); background: transparent; }

.banner { border-radius:8px; padding:10px 12px; margin-bottom:12px; border-left:4px solid var(--line); background:transparent; color:var(--ink-dim); }
.banner-ok { border-left-color: var(--neon-green); }
.banner-warn { border-left-color: var(--neon-amber); }
.banner-bad { border-left-color: var(--neon-red); }
.banner-muted { border-left-color: var(--line); }

.match-table { width:100%; border-collapse:collapse; border:1px solid var(--line); border-radius:8px; overflow:hidden; }
.match-table th { background: #070707; color: var(--ink-dim); font-size:0.78rem; letter-spacing:0.06em; text-transform:uppercase; text-align:left; padding:10px; }
.match-table td { padding:10px; border-top:1px solid var(--line); }
.match-yes { color: var(--neon-green); font-weight:700; text-shadow: none !important; }
.match-no { color: var(--neon-red); font-weight:700; text-shadow: none !important; }
.match-na { color: var(--ink-dim); font-style:italic; }

.trace-block { border:1px solid var(--line); background:transparent; border-radius:8px; padding:12px; margin-bottom:12px; }
.trace-title { margin:0 0 8px 0; color:var(--ink); font-weight:700; }
.trace-label { color:var(--ink-dim); font-size:0.78rem; letter-spacing:0.06em; text-transform:uppercase; }

.stButton > button, [data-testid="baseButton-secondary"], [data-testid="stButton"] button, button[kind] {
    background: rgba(255,255,255,0.04) !important;
    color: var(--ink) !important;
    border: 1px solid var(--line) !important;
    border-radius:9px !important;
    font-weight:700 !important;
}
.stButton > button:hover, [data-testid="stButton"] button:hover, button[kind]:hover {
    background: rgba(0, 229, 255, 0.10) !important;
    border-color: var(--neon-cyan) !important;
}

[data-testid="stDataFrame"], [data-testid="stDataEditor"] { border:1px solid var(--line); border-radius:8px; overflow:hidden; background:transparent; color:var(--ink); }

/* citation / prose under exceptions */
.citation { color: var(--muted); }

/* Aggressive selectors for Streamlit input/select components */
textarea, input, select { background-color: #111111 !important; color: #ffffff !important; border-color: #2a2a2a !important; border: 1px solid #2a2a2a !important; }
input[role="spinbutton"] { background-color: #111111 !important; color: var(--neon-cyan) !important; border-color: #2a2a2a !important; }
input[role="spinbutton"]::-webkit-inner-spin-button,
input[role="spinbutton"]::-webkit-outer-spin-button { background-color: #111111 !important; }
button, .stButton > button, [data-testid="stButton"] button, button[kind] { background-color: rgba(255,255,255,0.04) !important; color: var(--ink) !important; border-color: var(--line) !important; border: 1px solid var(--line) !important; }
button:hover, .stButton > button:hover, [data-testid="stButton"] button:hover, button[kind]:hover { background-color: rgba(0, 229, 255, 0.10) !important; }
div[data-baseweb="select"] > div { background-color: #111111 !important; color: #ffffff !important; border-color: #2a2a2a !important; }
div[data-baseweb="select"] * { color: #ffffff !important; }
div[data-baseweb="popover"] { background-color: #111111 !important; }
li[role="option"] { background-color: #111111 !important; color: #ffffff !important; }
.stTextArea > div > div { background-color: #111111 !important; }

/* DataFrame table styling - dark background */
[data-testid="stDataFrame"] div[role="grid"] { background: #0a0a0a !important; }
[data-testid="stDataFrame"] [role="gridcell"] { background: #0a0a0a !important; color: var(--ink) !important; }
[data-testid="stDataFrame"] th { background: #1a1a1a !important; color: var(--ink-dim) !important; border: 1px solid var(--line) !important; }
[data-testid="stDataFrame"] td { background: #0a0a0a !important; color: var(--ink) !important; border: 1px solid var(--line) !important; }

/* Slider/number input white boxes */
[data-testid="stNumberInput"] input { background-color: #111111 !important; color: var(--neon-cyan) !important; }
[data-testid="stSlider"] > div > div { background-color: #1a1a1a !important; }
[data-testid="stSlider"] [role="slider"] { background-color: var(--neon-cyan) !important; }

/* Checkbox */
[data-testid="stCheckbox"] div[role="checkbox"] { background-color: #111111 !important; border-color: var(--line) !important; }

/* Metric/KPI values */
[data-testid="metric-container"] { background: transparent !important; border: 1px solid var(--line) !important; border-radius: 8px !important; }

/* General Streamlit containers */
div.stColumn { background: transparent !important; }
div.stTabs { background: transparent !important; }
div.stExpander > button { background: transparent !important; }

/* Table/dataframe container backgrounds */
div.dataframe { background: #0a0a0a !important; color: var(--ink) !important; }
table { background: #0a0a0a !important; color: var(--ink) !important; border-color: var(--line) !important; }
thead { background: #1a1a1a !important; color: var(--ink-dim) !important; }
tbody tr { background: #0a0a0a !important; color: var(--ink) !important; }
tbody tr:hover { background: #151515 !important; }

.po-table, .score-table { width:100%; border-collapse:collapse; border:1px solid var(--line); background:#000000; color:#e8e8e8; }
.po-table th, .score-table th { background:#111111; color:#e8e8e8; text-align:left; padding:10px; border-bottom:1px solid var(--line); }
.po-table td, .score-table td { padding:10px; border-top:1px solid var(--line); color:#e8e8e8; background:#000000; }
.po-number-cell { color:#00e5ff; font-weight:700; text-shadow:0 0 8px #00e5ff33; }
.dark-pre { background:#0d0d0d; color:#e8e8e8; border:1px solid #2a2a2a; border-radius:8px; padding:12px; overflow:auto; white-space:pre-wrap; font-family:'JetBrains Mono', monospace; }

</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=30)
def load_samples() -> list[dict[str, Any]]:
    with open(PROJECT_ROOT / "data" / "sample_invoices.json", encoding="utf-8") as f:
        return json.load(f)


def _is_today(timestamp_str: str | None) -> bool:
    if not timestamp_str:
        return False
    return timestamp_str[:10] == date.today().isoformat()


def get_kpis() -> tuple[float, int]:
    records = get_all_audit_records()
    if not records:
        return 0.0, 0

    straight = 0
    exceptions_today = 0
    total = len(records)

    for row in records:
        overall = (row.get("match_result") or {}).get("overall_status")
        decision = row.get("final_decision")
        if overall == "straight_through" or decision in {"approved", "straight_through"}:
            straight += 1
        if (overall == "exception" or decision == "exception") and _is_today(row.get("timestamp")):
            exceptions_today += 1

    return round((straight / total) * 100, 1), exceptions_today


def get_business_kpis() -> tuple[float, float]:
    records = get_all_audit_records()
    if not records:
        return 0.0, 0.0

    processed = [r for r in records if not r.get("is_refusal", False)]
    if not processed:
        return 0.0, 0.0

    straight = sum(
        1
        for r in processed
        if (r.get("match_result") or {}).get("overall_status") == "straight_through"
        or r.get("final_decision") in {"approved", "straight_through"}
    )
    exceptions = sum(
        1
        for r in processed
        if (r.get("match_result") or {}).get("overall_status") == "exception"
        or r.get("final_decision") == "exception"
    )

    straight_rate = round((straight / len(processed)) * 100, 1)
    exception_rate = round((exceptions / len(processed)) * 100, 1)
    return straight_rate, exception_rate


def badge_for_decision(decision: str) -> str:
    mapping = {
        "straight_through": ("badge-ok", "STRAIGHT-THROUGH"),
        "approved": ("badge-ok", "STRAIGHT-THROUGH"),
        "exception": ("badge-exception", "EXCEPTION &#183; ROUTE TO AP CLERK"),
        "rejected": ("badge-rejected", "REJECTED"),
        "refused": ("badge-refused", "OUT OF SCOPE &#8212; REFUSED"),
    }
    cls, label = mapping.get(decision, ("badge-refused", decision.upper()))
    return f'<div class="badge {cls}">{label}</div>'


def format_money(val: Any) -> str:
    if isinstance(val, (int, float)):
        return f"${val:,.2f}"
    if val is None:
        return "-"
    return str(val)


def format_delta(delta: Any) -> str:
    if not isinstance(delta, (int, float)):
        return ""
    sign = "+" if delta >= 0 else ""
    return f"{sign}${delta:,.2f}"


def build_match_cell(status: str, delta: Any, citation: str) -> str:
    if status == "match":
        return '<span class="match-yes">&#10003; Match</span>'
    if status == "variance":
        delta_part = format_delta(delta)
        citation_part = f" - {citation}" if citation else ""
        return f'<span class="match-no">&#10007; {delta_part}{citation_part}</span>'
    return '<span class="match-na">Missing</span>'


def render_topbar() -> None:
    straight_pct, exc_today = get_kpis()
    st.markdown(
        f"""
<div class="topbar">
  <div>
    <h1 class="top-title">AP Autopilot</h1>
    <p class="top-subtitle">Invoice and contract exception intelligence with transparent agent decisions</p>
  </div>
  <div class="kpis">
    <div class="kpi-card">
      <p class="kpi-val">{straight_pct}%</p>
      <p class="kpi-label">Straight-through</p>
    </div>
    <div class="kpi-card">
      <p class="kpi-val">{exc_today}</p>
      <p class="kpi-label">Exceptions today</p>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def init_state() -> None:
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_demo_run", None)
    st.session_state.setdefault("last_raw_text", "")
    st.session_state.setdefault("demo_sample_idx", 0)
    st.session_state.setdefault("demo_last_loaded_idx", -1)
    st.session_state.setdefault("demo_text", "")
    st.session_state.setdefault("eval_results", None)
    st.session_state.setdefault("eval_last_run_time", None)
    st.session_state.setdefault("eval_run_requested", False)


def render_live_demo() -> None:
    st.subheader("Live Demo")

    samples = load_samples()
    labels = ["Custom invoice text"] + [f"{s['scenario_id']} ({s.get('expected_outcome', '?')})" for s in samples]

    selection_col, run_col = st.columns([4, 1])
    with selection_col:
        selected = st.selectbox(
            "Choose sample invoice",

            options=list(range(len(labels))),
            format_func=lambda i: labels[i],
            key="demo_sample_idx",
        )

    if selected > 0 and st.session_state.get("demo_last_loaded_idx") != selected:
        st.session_state["demo_text"] = samples[selected - 1]["raw_text"]
        st.session_state["demo_last_loaded_idx"] = selected

    st.text_area(
        "Raw invoice text",
        key="demo_text",
        height=220,
        placeholder="Paste raw invoice text here...",
    )

    with run_col:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        run_now = st.button("Run", use_container_width=True)

    if run_now:
        raw_text = st.session_state.get("demo_text", "").strip()
        if not raw_text:
            st.markdown(
                """<div style="color:#999999; padding: 12px; border: 1px solid #2a2a2a; border-radius: 4px;">Enter invoice text or select a sample first.</div>""",
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Running AP agent pipeline..."):
                result = run_pipeline(raw_text)
            st.session_state["last_result"] = result
            st.session_state["last_demo_run"] = {
                "match_result": result.get("match_result") or {},
                "tool_call_trace": result.get("tool_call_trace") or [],
                "final_decision": result.get("final_decision"),
                "reason": result.get("reason"),
                "run_id": result.get("run_id"),
            }
            st.session_state["last_raw_text"] = raw_text

    result = st.session_state.get("last_demo_run") or st.session_state.get("last_result")
    if not result:
        st.markdown(
            """<div style="color:#999999; padding: 12px; border: 1px solid #2a2a2a; border-radius: 4px;">Run an invoice to see decision details, field matches, and tool trace.</div>""",
            unsafe_allow_html=True,
        )
        return

    decision = result.get("final_decision", "unknown")
    match_result = result.get("match_result") or {}
    exceptions = match_result.get("exceptions") or []

    st.markdown(badge_for_decision(decision), unsafe_allow_html=True)

    if decision in {"approved", "straight_through"}:
        st.markdown(
            '<div class="banner banner-ok">All required fields matched against PO/contract records. Invoice can move straight-through.</div>',
            unsafe_allow_html=True,
        )
    elif decision == "exception":
        if exceptions:
            lines = "".join([f"<li><strong>{e.get('field', '?')}</strong>: {e.get('reason', '')}</li>" for e in exceptions])
            st.markdown(
                f'<div class="banner banner-warn">Exception detected and routed to AP Clerk.<ul>{lines}</ul></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="banner banner-warn">Exception detected and routed to AP Clerk.</div>',
                unsafe_allow_html=True,
            )
    elif decision == "rejected":
        st.markdown(
            f'<div class="banner banner-bad">Invoice rejected. {result.get("reason") or ""}</div>',
            unsafe_allow_html=True,
        )
    elif decision == "refused":
        st.markdown(
            f'<div class="banner banner-muted">Document out of scope and refused. {result.get("reason") or ""}</div>',
            unsafe_allow_html=True,
        )

    field_results = match_result.get("field_results") or {}
    if not field_results:
        return

    citation_by_field: dict[str, str] = {}
    for item in exceptions:
        field = item.get("field")
        if field:
            citation_by_field[field] = item.get("citation", "")

    body_rows = []
    for field, payload in field_results.items():
        invoice_val = payload.get("invoice_value")
        po_val = payload.get("po_value")
        status = str(payload.get("status", "missing"))
        delta = payload.get("delta")
        citation = citation_by_field.get(field, "")

        body_rows.append(
            "".join(
                [
                    "<tr>",
                    f"<td>{field}</td>",
                    f"<td>{format_money(invoice_val)}</td>",
                    f"<td>{format_money(po_val)}</td>",
                    f"<td>{build_match_cell(status, delta, citation)}</td>",
                    "</tr>",
                ]
            )
        )

    st.markdown(
        """
<table class="match-table">
  <thead>
    <tr>
      <th>Field</th>
      <th>Invoice</th>
      <th>PO/Contract</th>
      <th>Match</th>
    </tr>
  </thead>
  <tbody>
"""
        + "".join(body_rows)
        + """
  </tbody>
</table>
""",
        unsafe_allow_html=True,
    )


def _po_rows_from_store() -> list[dict[str, Any]]:
    items = []
    all_pos = get_all_pos()
    for po_number, rec in all_pos.items():
        items.append(
            {
                "delete": False,
                "po_number": po_number,
                "vendor": rec.get("vendor", ""),
                "contract_id": rec.get("contract_id", ""),
                "line_qty": int(rec.get("line_qty", 0) or 0),
                "unit_price": float(rec.get("unit_price", 0.0) or 0.0),
                "expected_total": float(rec.get("expected_total", 0.0) or 0.0),
                "approval_required_over": float(rec.get("approval_required_over", 0.0) or 0.0),
                "approved_by": rec.get("approved_by") or "",
            }
        )
    return items


def render_manage_pos() -> None:
    st.subheader("Manage POs & Contracts")
    st.caption("The table below is read-only and dark-themed; edit or delete records through the form below it.")

    rows = _po_rows_from_store()
    if rows:
        body_rows = []
        for row in rows:
            body_rows.append(
                "<tr>"
                f"<td class='po-number-cell'>{html.escape(str(row['po_number']))}</td>"
                f"<td>{html.escape(str(row['vendor']))}</td>"
                f"<td>{html.escape(str(row['contract_id']))}</td>"
                f"<td>{html.escape(str(int(row['line_qty'])))}</td>"
                f"<td>{html.escape(format_money(row['unit_price']))}</td>"
                f"<td>{html.escape(format_money(row['expected_total']))}</td>"
                f"<td>{html.escape(format_money(row['approval_required_over']))}</td>"
                f"<td>{html.escape(str(row['approved_by'] or ''))}</td>"
                "</tr>"
            )

        st.markdown(
            """
<table class="po-table">
  <thead>
    <tr>
      <th>PO Number</th>
      <th>Vendor</th>
      <th>Contract ID</th>
      <th>Line Qty</th>
      <th>Unit Price</th>
      <th>Expected Total</th>
      <th>Approval Threshold</th>
      <th>Approved By</th>
    </tr>
  </thead>
  <tbody>
"""
            + "".join(body_rows)
            + """
  </tbody>
</table>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div style="color:#999999; padding: 12px; border: 1px solid #2a2a2a; border-radius: 4px;">No PO records yet.</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### Edit Existing PO")

    if rows:
        po_numbers = [row["po_number"] for row in rows]
        selected_po = st.selectbox("Select PO to edit", options=po_numbers, key="manage_po_select")
        if st.button("Load selected PO", use_container_width=True):
            selected_row = next((row for row in rows if row["po_number"] == selected_po), None)
            if selected_row:
                st.session_state["edit_po_number"] = selected_row["po_number"]
                st.session_state["edit_vendor"] = selected_row["vendor"]
                st.session_state["edit_contract_id"] = selected_row["contract_id"]
                st.session_state["edit_line_qty"] = int(selected_row["line_qty"])
                st.session_state["edit_unit_price"] = float(selected_row["unit_price"])
                st.session_state["edit_expected_total"] = float(selected_row["expected_total"])
                st.session_state["edit_approval_required_over"] = float(selected_row["approval_required_over"])
                st.session_state["edit_approved_by"] = selected_row["approved_by"] or ""

        with st.form("edit_po_form", clear_on_submit=False):
            c1, c2 = st.columns(2)
            edit_po_number = c1.text_input("PO Number *", key="edit_po_number", value=st.session_state.get("edit_po_number", ""))
            edit_vendor = c2.text_input("Vendor *", key="edit_vendor", value=st.session_state.get("edit_vendor", ""))
            edit_contract_id = c1.text_input("Contract ID *", key="edit_contract_id", value=st.session_state.get("edit_contract_id", ""))
            edit_line_qty = c2.number_input("Line Qty *", min_value=1, step=1, value=int(st.session_state.get("edit_line_qty", 100)), key="edit_line_qty")
            edit_unit_price = c1.number_input("Unit Price *", min_value=0.0, step=0.01, value=float(st.session_state.get("edit_unit_price", 10.0)), format="%.2f", key="edit_unit_price")
            edit_expected_total = c2.number_input("Expected Total *", min_value=0.0, step=0.01, value=float(st.session_state.get("edit_expected_total", 1000.0)), format="%.2f", key="edit_expected_total")
            edit_approval_required_over = c1.number_input(
                "Approval Threshold *",
                min_value=0.0,
                step=100.0,
                value=float(st.session_state.get("edit_approval_required_over", 10000.0)),
                format="%.2f",
                key="edit_approval_required_over",
            )
            edit_approved_by = c2.text_input("Approved By", key="edit_approved_by", value=st.session_state.get("edit_approved_by", ""))

            edit_submit, delete_submit = st.columns(2)
            with edit_submit:
                submitted = st.form_submit_button("Save PO")
            with delete_submit:
                delete_submitted = st.form_submit_button("Delete Selected PO")

            if submitted:
                if not edit_po_number.strip() or not edit_vendor.strip() or not edit_contract_id.strip():
                    st.markdown(
                        """<div style="color:#999999; padding: 12px; border: 1px solid #2a2a2a; border-radius: 4px;">PO Number, Vendor, and Contract ID are required.</div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    if selected_po != edit_po_number.strip():
                        delete_po(selected_po)
                    add_po(
                        edit_po_number.strip(),
                        {
                            "vendor": edit_vendor.strip(),
                            "contract_id": edit_contract_id.strip(),
                            "line_qty": int(edit_line_qty),
                            "unit_price": float(edit_unit_price),
                            "expected_total": float(edit_expected_total),
                            "approval_required_over": float(edit_approval_required_over),
                            "approved_by": edit_approved_by.strip() or None,
                        },
                    )
                    st.session_state["manage_po_select"] = edit_po_number.strip()
                    st.rerun()

            if delete_submitted:
                delete_po(selected_po)
                st.session_state["manage_po_select"] = ""
                st.rerun()
    else:
        st.info("Add a PO to start editing it.")

    st.markdown("---")
    st.markdown("#### Add New PO")

    with st.form("add_po_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        po_number = c1.text_input("PO Number *", placeholder="PO-9001")
        vendor = c2.text_input("Vendor *", placeholder="Acme Supplies")
        contract_id = c1.text_input("Contract ID *", placeholder="MSA-ACME-2026")
        line_qty = c2.number_input("Line Qty *", min_value=1, step=1, value=100)
        unit_price = c1.number_input("Unit Price *", min_value=0.0, step=0.01, value=10.0, format="%.2f")
        expected_total = c2.number_input("Expected Total *", min_value=0.0, step=0.01, value=1000.0, format="%.2f")
        approval_required_over = c1.number_input(
            "Approval Threshold *", min_value=0.0, step=100.0, value=10000.0, format="%.2f"
        )
        approved_by = c2.text_input("Approved By", placeholder="J. Smith")

        submitted = st.form_submit_button("Add PO")
        if submitted:
            if not po_number.strip() or not vendor.strip() or not contract_id.strip():
                st.markdown(
                    """<div style="color:#999999; padding: 12px; border: 1px solid #2a2a2a; border-radius: 4px;">PO Number, Vendor, and Contract ID are required.</div>""",
                    unsafe_allow_html=True,
                )
            else:
                add_po(
                    po_number.strip(),
                    {
                        "vendor": vendor.strip(),
                        "contract_id": contract_id.strip(),
                        "line_qty": int(line_qty),
                        "unit_price": float(unit_price),
                        "expected_total": float(expected_total),
                        "approval_required_over": float(approval_required_over),
                        "approved_by": approved_by.strip() or None,
                    },
                )
                st.markdown(
                    """<div style="color:#999999; padding: 12px; border: 1px solid #2a2a2a; border-radius: 4px;">PO added. It can be used immediately in Live Demo.</div>""",
                    unsafe_allow_html=True,
                )
                st.rerun()


def render_agent_trace() -> None:
    st.subheader("Agent Trace")
    st.caption(
        "Most recent Live Demo run only. This reflects the same stored tool_call_trace used by Live Demo, so it persists across tabs."
    )

    result = st.session_state.get("last_demo_run") or st.session_state.get("last_result")
    if not result:
        st.markdown(
            """<div style="color:#999999; padding: 12px; border: 1px solid #2a2a2a; border-radius: 4px;">No run found yet. Run the Live Demo first.</div>""",
            unsafe_allow_html=True,
        )
        return

    trace = (result.get("tool_call_trace") if isinstance(result, dict) else None) or []
    decision = result.get("final_decision", "unknown") if isinstance(result, dict) else "unknown"
    run_id = result.get("run_id", "") if isinstance(result, dict) else ""

    st.markdown(
        f'<div class="banner banner-muted">Decision: <strong>{decision}</strong> | run_id: <strong>{run_id}</strong> | tool calls: <strong>{len(trace)}</strong></div>',
        unsafe_allow_html=True,
    )

    if not trace:
        st.markdown(
            '<div class="banner banner-muted">No tool calls were recorded for this run (for example, refused path).</div>',
            unsafe_allow_html=True,
        )
        return

    for idx, step in enumerate(trace, start=1):
        name = step.get("tool_name", "unknown")
        tool_input = step.get("tool_input", {})
        tool_output = step.get("tool_output", {})

        st.markdown(
            f'<div class="trace-block"><p class="trace-title">{idx}. {html.escape(str(name))}</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="trace-label">Input</div>', unsafe_allow_html=True)
        st.markdown(
            f"<div class='dark-pre'>{html.escape(json.dumps(tool_input, indent=2, default=str))}</div>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="trace-label">Output</div>', unsafe_allow_html=True)
        st.markdown(
            f"<div class='dark-pre'>{html.escape(json.dumps(tool_output, indent=2, default=str))}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


def render_eval_scorecard_shell() -> None:
    st.subheader("Eval Scorecard")
    st.caption("Technical eval checks run against the pipeline; business KPIs come from the audit log.")

    if st.button("▶ Run Evaluation", key="eval_run_button", use_container_width=False):
        st.session_state["eval_run_requested"] = True

    if st.session_state.get("eval_run_requested", False):
        with st.spinner("Running evaluation scenarios..."):
            try:
                eval_rows = run_all_evals()
                st.session_state["eval_results"] = eval_rows
                st.session_state["eval_last_run_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["eval_run_requested"] = False
            except Exception as exc:
                st.markdown(
                    f"""<div style="color:#999999; padding: 12px; border: 1px solid #2a2a2a; border-radius: 4px;">Evaluation run failed: {exc}</div>""",
                    unsafe_allow_html=True,
                )
                return

    eval_rows = st.session_state.get("eval_results")
    if not eval_rows:
        st.markdown(
            """<div style="color:#999999; padding: 12px; border: 1px solid #2a2a2a; border-radius: 4px;">No eval results yet. Click Run Evaluation to generate them.</div>""",
            unsafe_allow_html=True,
        )
        return

    if st.session_state.get("eval_last_run_time"):
        st.caption(f"Last run: {st.session_state['eval_last_run_time']}")

    passed = sum(1 for row in eval_rows if row.get("overall", False))
    summary_text = f"{passed}/{len(eval_rows)} passed"
    st.markdown(
        f'<div class="banner banner-ok"><strong>{summary_text}</strong> — technical checks passed for the selected evaluation scenarios.</div>',
        unsafe_allow_html=True,
    )

    headers = [
        "scenario_name",
        "task_completion",
        "trace_correctness",
        "tool_call_accuracy",
        "governance_check",
        "overall",
    ]
    body_rows = []
    for row in eval_rows:
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('scenario_name', '')))}</td>"
            f"<td>{html.escape('PASS' if row.get('task_completion') else 'FAIL')}</td>"
            f"<td>{html.escape('PASS' if row.get('trace_correctness') else 'FAIL')}</td>"
            f"<td>{html.escape('PASS' if row.get('tool_call_accuracy') else 'FAIL')}</td>"
            f"<td>{html.escape('PASS' if row.get('governance_check') else 'FAIL')}</td>"
            f"<td>{html.escape('PASS' if row.get('overall') else 'FAIL')}</td>"
            "</tr>"
        )

    st.markdown(
        """
<table class="score-table">
  <thead>
    <tr>
      <th>Scenario</th>
      <th>Task</th>
      <th>Trace</th>
      <th>Tools</th>
      <th>Governance</th>
      <th>Overall</th>
    </tr>
  </thead>
  <tbody>
"""
        + "".join(body_rows)
        + """
  </tbody>
</table>
""",
        unsafe_allow_html=True,
    )

    straight_rate, exception_rate = get_business_kpis()
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Business KPI — Straight-through rate", f"{straight_rate:.1f}%")
    with c2:
        st.metric("Business KPI — Exception catch rate", f"{exception_rate:.1f}%")

    st.caption("These business KPIs are derived from audit log outcomes, separate from the technical eval checks above.")

# Sidebar demo controls: allow clearing demo audit history (requires confirmation)
with st.sidebar:
    st.markdown("### Demo controls")
    confirm_clear = st.checkbox("I understand this clears all demo history", key="confirm_clear")
    if confirm_clear and st.button("Reset demo data"):
        from audit.audit_log import clear_audit_log
        clear_audit_log()
        st.session_state["last_result"] = None
        st.session_state["last_demo_run"] = None
        st.session_state["last_raw_text"] = ""
        st.session_state["demo_sample_idx"] = 0
        st.session_state["demo_last_loaded_idx"] = -1
        st.session_state["demo_text"] = ""
        st.session_state["eval_results"] = None
        st.session_state["eval_last_run_time"] = None
        st.session_state["eval_run_requested"] = False
        st.rerun()

init_state()
render_topbar()

tab_demo, tab_eval, tab_manage, tab_trace = st.tabs(
    ["Live Demo", "Eval Scorecard", "Manage POs & Contracts", "Agent Trace"]
)

with tab_demo:
    render_live_demo()

with tab_eval:
    render_eval_scorecard_shell()

with tab_manage:
    render_manage_pos()

with tab_trace:
    render_agent_trace()
