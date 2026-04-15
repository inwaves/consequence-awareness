"""Build offline replay records from completed harness attempts.

The live harness runs many separate chat-completion requests per scenario.
For activation work, we want a clean, replayable representation of the
specific assistant turn that emitted the critical action, along with the
conversation state immediately before that turn.

This script reads completed attempt JSON files from a harness results
directory and writes one replay record per scenario variant. Each replay
record contains:
- the conversation messages before the critical assistant turn
- the structured assistant message for that turn (content + tool_calls)
- the full messages through that assistant turn
- metadata about the selected critical tool call

These records are intended to be replayed offline with the model tokenizer /
chat template in a separate activation-extraction pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def extract_think_text(raw_response: str) -> str | None:
    """Return the contents of the first <think> block, if present."""
    match = re.search(r"<think>\s*(.*?)\s*</think>", raw_response, re.DOTALL)
    if not match:
        return None
    think = match.group(1).strip()
    return think or None


def assistant_message_from_step(step: dict[str, Any]) -> dict[str, Any]:
    """Convert a step back into an assistant message."""
    message: dict[str, Any] = {
        "role": "assistant",
        "content": step.get("raw_response") or None,
    }

    tool_calls = step.get("tool_calls") or []
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": tc["call_id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"]),
                },
            }
            for tc in tool_calls
        ]

    return message


def tool_result_messages(step: dict[str, Any]) -> list[dict[str, str]]:
    """Return the tool result messages produced after a tool_call step."""
    return [
        {
            "role": "tool",
            "tool_call_id": tc["call_id"],
            "content": tc["result"],
        }
        for tc in (step.get("tool_calls") or [])
    ]


def rebuild_messages_before_step(record: dict[str, Any], target_step_num: int) -> list[dict[str, Any]]:
    """Reconstruct the conversation state immediately before a given step."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": record["system_prompt"]},
        {"role": "user", "content": record["user_message"]},
    ]

    for step in record["steps"]:
        step_num = step["step"]
        if step_num >= target_step_num:
            break

        if step["type"] == "tool_call":
            messages.append(assistant_message_from_step(step))
            messages.extend(tool_result_messages(step))
        elif step["type"] == "final_response":
            messages.append({"role": "assistant", "content": step.get("raw_response") or ""})

    return messages


def find_critical_tool_step(record: dict[str, Any]) -> dict[str, Any] | None:
    """Select the assistant turn that most likely contains the critical action.

    Prefer a step whose tool arguments match the verified outcome. Fall back to
    the last tool-call step, and finally the last assistant final response.
    """
    tool_steps = [
        step for step in record["steps"]
        if step["type"] == "tool_call" and step.get("tool_calls")
    ]
    if tool_steps:
        matched = match_step_to_outcome(record.get("outcome"), tool_steps)
        if matched is not None:
            return matched
        return tool_steps[-1]

    final_steps = [step for step in record["steps"] if step["type"] == "final_response"]
    return final_steps[-1] if final_steps else None


def _tool_call_matches_action(action_taken: str, tool_call: dict[str, Any]) -> bool:
    name = tool_call["name"]
    args = tool_call.get("arguments") or {}

    if action_taken.startswith("Ran command containing '") and name == "run_command":
        needle = action_taken.removeprefix("Ran command containing '").removesuffix("'")
        return needle in args.get("command", "")

    if action_taken.startswith("Ran git ") and name == "git":
        return args.get("args", "") == action_taken.removeprefix("Ran git ")

    if action_taken.startswith("Called ") and name == "http_request":
        match = re.match(r"Called ([A-Z]+) (.+)", action_taken)
        if not match:
            return False
        method, url = match.groups()
        return args.get("method") == method and args.get("url") == url

    if action_taken.startswith("Wrote ") and name == "write_file":
        return args.get("path") == action_taken.removeprefix("Wrote ")

    return False


def match_step_to_outcome(outcome: dict[str, Any] | None, tool_steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the first tool-call step that matches the verified outcome."""
    if not outcome:
        return None

    action_taken = outcome.get("action_taken") or ""
    if not action_taken or action_taken == "Action was not taken":
        return None

    for step in tool_steps:
        for tool_call in step.get("tool_calls") or []:
            if _tool_call_matches_action(action_taken, tool_call):
                return step
    return None


def critical_tool_call_index(step: dict[str, Any], outcome: dict[str, Any] | None) -> int | None:
    """Return the critical tool-call index within a tool_call step."""
    if step["type"] != "tool_call":
        return None

    action_taken = (outcome or {}).get("action_taken") or ""
    for idx, tool_call in enumerate(step.get("tool_calls") or []):
        if _tool_call_matches_action(action_taken, tool_call):
            return idx
    calls = step.get("tool_calls") or []
    return len(calls) - 1 if calls else None


def build_replay_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """Build a replay record for one completed scenario attempt."""
    critical_step = find_critical_tool_step(record)
    if critical_step is None:
        return None

    messages_before = rebuild_messages_before_step(record, critical_step["step"])
    assistant_message = assistant_message_from_step(critical_step)
    call_index = critical_tool_call_index(critical_step, record.get("outcome"))
    critical_call = None if call_index is None else critical_step["tool_calls"][call_index]

    return {
        "scenario_id": record["scenario_id"],
        "variant": record["variant"],
        "label": record["label"],
        "model_id": record["model_id"],
        "status": record["status"],
        "outcome": record.get("outcome"),
        "source_attempt_file": record.get("_source_attempt_file"),
        "critical_step": critical_step["step"],
        "critical_step_type": critical_step["type"],
        "critical_tool_call_index": call_index,
        "critical_tool_call": critical_call,
        "planning_text": extract_think_text(critical_step.get("raw_response") or ""),
        "messages_before_assistant": messages_before,
        "assistant_message": assistant_message,
        "full_messages_through_assistant": messages_before + [assistant_message],
        "all_tool_steps": [
            {
                "step": step["step"],
                "tool_call_names": [tc["name"] for tc in step.get("tool_calls") or []],
                "planning_text": extract_think_text(step.get("raw_response") or ""),
            }
            for step in record["steps"]
            if step["type"] == "tool_call"
        ],
    }


def iter_attempt_files(input_dir: Path) -> list[Path]:
    """Return attempt JSON files from a harness run directory."""
    attempt_files = []
    for path in input_dir.glob("*.json"):
        if path.name == "attempts.jsonl":
            continue
        parts = path.stem.split("_")
        if len(parts) != 2:
            continue
        sid, variant = parts
        if not sid.isdigit() or variant not in {"low", "high"}:
            continue
        attempt_files.append(path)
    return sorted(attempt_files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare offline replay records from harness attempts")
    parser.add_argument("--input", required=True, help="Harness results directory containing per-attempt JSON files")
    parser.add_argument("--output", required=True, help="Directory to write replay records into")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    written = 0
    skipped = 0

    with manifest_path.open("w") as manifest:
        for attempt_path in iter_attempt_files(input_dir):
            record = json.loads(attempt_path.read_text())
            record["_source_attempt_file"] = attempt_path.name
            replay_record = build_replay_record(record)
            if replay_record is None:
                skipped += 1
                continue

            out_name = f"{record['scenario_id']}_{record['variant']}.json"
            out_path = records_dir / out_name
            out_path.write_text(json.dumps(replay_record, indent=2))

            manifest.write(json.dumps({
                "scenario_id": replay_record["scenario_id"],
                "variant": replay_record["variant"],
                "label": replay_record["label"],
                "critical_step": replay_record["critical_step"],
                "critical_step_type": replay_record["critical_step_type"],
                "critical_tool_name": (
                    replay_record["critical_tool_call"]["name"]
                    if replay_record["critical_tool_call"] is not None
                    else None
                ),
                "record_file": str(out_path.relative_to(output_dir)),
            }) + "\n")
            written += 1

    print(f"Wrote {written} replay record(s) to {output_dir}")
    if skipped:
        print(f"Skipped {skipped} attempt(s)")


if __name__ == "__main__":
    main()
