from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from project_paths import find_project_root

PROJECT_ROOT = find_project_root(__file__)

from agents.graph import run_pipeline


def load_sample_scenarios() -> list[dict[str, Any]]:
    data_path = PROJECT_ROOT / "data" / "sample_invoices.json"
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


def format_tool_call(step: dict[str, Any]) -> str:
    tool_name = step.get("tool_name", "unknown")
    tool_input = step.get("tool_input", {})
    return f"- {tool_name}: {json.dumps(tool_input, indent=2, default=str)}"


def run_debug_scenarios() -> None:
    scenarios = load_sample_scenarios()
    selected = [s for s in scenarios if s["scenario_id"] in {"clean", "missing_approval", "malformed"}]

    for scenario in selected:
        name = scenario["scenario_id"]
        raw_text = scenario["raw_text"]

        print(f"=== SCENARIO: {name} ===")
        print("Raw text:")
        print(raw_text)
        print()

        result = run_pipeline(raw_text)

        raw_extraction = result.get("raw_extraction")
        print("Raw extraction:")
        print(json.dumps(raw_extraction, indent=2, default=str))
        print()

        match_result = result.get("match_result")
        print("Match result:")
        if isinstance(match_result, dict):
            print(json.dumps(
                {
                    "overall_status": match_result.get("overall_status"),
                    "field_results": match_result.get("field_results"),
                    "exceptions": match_result.get("exceptions"),
                },
                indent=2,
                default=str,
            ))
        else:
            print(str(match_result))
        print()

        trace = result.get("tool_call_trace") or []
        print("Tool call trace:")
        for step in trace:
            print(format_tool_call(step))
        if not trace:
            print("- (no tool calls)")
        print()

        print(f"Final decision: {result.get('final_decision')}")
        print(f"Reason: {result.get('reason')}")
        print("\n")


if __name__ == "__main__":
    run_debug_scenarios()
