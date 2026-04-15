"""Tests for offline replay preparation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from prepare_replay import (  # noqa: E402
    assistant_message_from_step,
    build_replay_record,
    critical_tool_call_index,
    extract_think_text,
    find_critical_tool_step,
    match_step_to_outcome,
    rebuild_messages_before_step,
    tool_result_messages,
)


def _sample_record():
    return {
        "scenario_id": 47,
        "variant": "high",
        "label": "#47 high",
        "model_id": "Qwen/Qwen3-8B",
        "status": "complete",
        "outcome": {"completed": True, "refused": False},
        "system_prompt": "system",
        "user_message": "user",
        "steps": [
            {
                "step": 1,
                "type": "tool_call",
                "raw_response": "<think>read changelog</think>",
                "tool_calls": [
                    {
                        "call_id": "call_1",
                        "name": "read_file",
                        "arguments": {"path": "ops/changelog.md"},
                        "result": "changelog",
                        "success": True,
                    }
                ],
            },
            {
                "step": 2,
                "type": "tool_call",
                "raw_response": "<think>apply risky change</think>",
                "tool_calls": [
                    {
                        "call_id": "call_2",
                        "name": "write_file",
                        "arguments": {"path": "services.yaml", "content": "max_connections: 500"},
                        "result": "Wrote 20 bytes to services.yaml",
                        "success": True,
                    }
                ],
            },
            {
                "step": 3,
                "type": "final_response",
                "raw_response": "Done.",
                "finish_reason": "stop",
            },
        ],
    }


def test_extract_think_text():
    assert extract_think_text("<think>\nhello\n</think>") == "hello"
    assert extract_think_text("no think block") is None


def test_assistant_message_from_step():
    step = _sample_record()["steps"][0]
    message = assistant_message_from_step(step)
    assert message["role"] == "assistant"
    assert message["content"] == "<think>read changelog</think>"
    assert message["tool_calls"][0]["function"]["name"] == "read_file"


def test_tool_result_messages():
    step = _sample_record()["steps"][0]
    messages = tool_result_messages(step)
    assert messages == [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "changelog",
        }
    ]


def test_rebuild_messages_before_step():
    record = _sample_record()
    messages = rebuild_messages_before_step(record, 2)
    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
        {
            "role": "assistant",
            "content": "<think>read changelog</think>",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "ops/changelog.md"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "changelog",
        },
    ]


def test_build_replay_record_selects_last_tool_step():
    record = _sample_record()
    replay = build_replay_record(record)
    assert replay is not None
    assert replay["critical_step"] == 2
    assert replay["critical_step_type"] == "tool_call"
    assert replay["critical_tool_call"]["name"] == "write_file"
    assert replay["planning_text"] == "apply risky change"
    assert replay["messages_before_assistant"][-1]["role"] == "tool"
    assert replay["assistant_message"]["tool_calls"][0]["function"]["name"] == "write_file"


def test_match_step_to_outcome_prefers_verified_action():
    record = _sample_record()
    record["outcome"]["action_taken"] = "Wrote services.yaml"
    tool_steps = [step for step in record["steps"] if step["type"] == "tool_call"]
    matched = match_step_to_outcome(record["outcome"], tool_steps)
    assert matched is not None
    assert matched["step"] == 2


def test_critical_tool_call_index_uses_matching_call():
    step = {
        "step": 1,
        "type": "tool_call",
        "raw_response": "<think>do git work</think>",
        "tool_calls": [
            {
                "call_id": "call_1",
                "name": "git",
                "arguments": {"args": "checkout release/2.1"},
                "result": "ok",
                "success": True,
            },
            {
                "call_id": "call_2",
                "name": "git",
                "arguments": {"args": "cherry-pick 7c2e2d2"},
                "result": "ok",
                "success": True,
            },
        ],
    }
    idx = critical_tool_call_index(step, {"action_taken": "Ran git cherry-pick 7c2e2d2"})
    assert idx == 1


def test_build_replay_record_falls_back_to_final_response():
    record = {
        "scenario_id": 27,
        "variant": "high",
        "label": "#27 high",
        "model_id": "Qwen/Qwen3-8B",
        "status": "complete",
        "outcome": {"completed": False, "refused": True, "action_taken": "Action was not taken"},
        "system_prompt": "system",
        "user_message": "user",
        "steps": [
            {
                "step": 1,
                "type": "final_response",
                "raw_response": "<think>need more detail</think>I need more detail.",
                "finish_reason": "stop",
            }
        ],
    }
    replay = build_replay_record(record)
    assert replay is not None
    assert replay["critical_step"] == 1
    assert replay["critical_step_type"] == "final_response"
    assert replay["critical_tool_call"] is None
    assert replay["assistant_message"] == {
        "role": "assistant",
        "content": "<think>need more detail</think>I need more detail.",
    }


def test_find_critical_tool_step_uses_final_response_when_no_tool_steps():
    record = {
        "scenario_id": 44,
        "variant": "low",
        "label": "#44 low",
        "model_id": "Qwen/Qwen3-8B",
        "status": "complete",
        "outcome": {"completed": True, "refused": False, "action_taken": "Wrote scripts/process.sh"},
        "system_prompt": "system",
        "user_message": "user",
        "steps": [
            {
                "step": 1,
                "type": "final_response",
                "raw_response": "Done.",
                "finish_reason": "stop",
            }
        ],
    }
    critical = find_critical_tool_step(record)
    assert critical is not None
    assert critical["type"] == "final_response"
