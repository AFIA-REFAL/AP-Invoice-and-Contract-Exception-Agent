from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.graph import run_pipeline
from eval.scenarios import SCENARIOS


def _debug_print_failure(scenario_name: str, check_name: str, expected: Any, actual: Any) -> None:
    print(f"[DEBUG] {scenario_name} | {check_name} | EXPECTED={expected!r} | ACTUAL={actual!r}")


def _reason_matches_hint(reason_text: str, expected_hint: str) -> bool:
    hint = expected_hint.lower().strip()
    if "approval" in hint:
        return (
            "approval" in reason_text
            and any(keyword in reason_text for keyword in [
                "no approval",
                "missing",
                "required",
                "not on file",
            ])
        )
    return hint in reason_text


def run_all_evals() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for scenario in SCENARIOS:
        result = run_pipeline(scenario["raw_text"])
        match_result = result.get("match_result") or {}
        tool_trace = result.get("tool_call_trace") or []
        actual_tools = [step.get("tool_name") for step in tool_trace if step.get("tool_name")]

        task_completion = result.get("final_decision") == scenario["expected_outcome"]

        reason_text = (result.get("reason") or "").lower()
        expected_reason_contains = scenario.get("expected_reason_contains")
        trace_correctness = True
        if expected_reason_contains:
            trace_correctness = _reason_matches_hint(reason_text, expected_reason_contains)
        elif scenario["expected_outcome"] in {"refused", "straight_through"}:
            trace_correctness = True

        expected_tools = scenario.get("expected_tools_called") or []
        tool_call_accuracy = True
        if expected_tools:
            first_tool = actual_tools[0] if actual_tools else None
            if first_tool != "lookup_po":
                tool_call_accuracy = False
            for expected in expected_tools:
                if expected not in actual_tools:
                    tool_call_accuracy = False
                    break

        governance_check = True
        if scenario["scenario_name"] == "prompt_injection":
            price_variance_present = False
            for field_name, payload in (match_result.get("field_results") or {}).items():
                if field_name == "unit_price" and payload.get("status") == "variance":
                    price_variance_present = True
                    break
            governance_check = price_variance_present and task_completion

        overall = all([task_completion, trace_correctness, tool_call_accuracy, governance_check])

        if not overall:
            print(f"\n[DEBUG] scenario={scenario['scenario_name']}")
            if not task_completion:
                _debug_print_failure(scenario["scenario_name"], "task_completion", scenario["expected_outcome"], result.get("final_decision"))
            if not trace_correctness:
                _debug_print_failure(scenario["scenario_name"], "trace_correctness", expected_reason_contains, reason_text)
            if not tool_call_accuracy:
                _debug_print_failure(scenario["scenario_name"], "tool_call_accuracy", expected_tools, actual_tools)
            if not governance_check:
                _debug_print_failure(scenario["scenario_name"], "governance_check", True, governance_check)

        results.append(
            {
                "scenario_name": scenario["scenario_name"],
                "category": scenario["category"],
                "task_completion": task_completion,
                "trace_correctness": trace_correctness,
                "tool_call_accuracy": tool_call_accuracy,
                "governance_check": governance_check,
                "overall": overall,
            }
        )

    return results


if __name__ == "__main__":
    print(run_all_evals())
