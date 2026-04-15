"""Tests for activation-extraction helpers."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from extract_activations import (  # noqa: E402
    activation_suffix,
    build_region_queries,
    find_region_char_spans,
    manifest_row_from_metadata,
    metadata_matches_request,
    record_output_paths,
    render_assistant_message_fallback,
    render_prefix_fallback,
    token_region_memberships,
)


def _sample_replay_record():
    return {
        "scenario_id": 47,
        "variant": "high",
        "label": "#47 high",
        "critical_step": 2,
        "critical_step_type": "tool_call",
        "critical_tool_call_index": 0,
        "critical_tool_call": {
            "name": "write_file",
            "arguments": {"path": "services.yaml", "content": "max_connections: 500"},
        },
        "planning_text": "apply risky change",
        "messages_before_assistant": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        "assistant_message": {
            "role": "assistant",
            "content": "<think>apply risky change</think>\n",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path": "services.yaml", "content": "max_connections: 500"}',
                    },
                }
            ],
        },
        "full_messages_through_assistant": [],
    }


def test_render_assistant_message_fallback_includes_tool_call_block():
    replay = _sample_replay_record()
    rendered = render_assistant_message_fallback(replay["assistant_message"])
    assert "<tool_call>" in rendered
    assert '"name": "write_file"' in rendered
    assert '"path": "services.yaml"' in rendered


def test_render_prefix_fallback_adds_assistant_prompt():
    replay = _sample_replay_record()
    rendered = render_prefix_fallback(replay["messages_before_assistant"])
    assert rendered.endswith("<|assistant|>\n")


def test_build_region_queries_captures_planning_and_args():
    replay = _sample_replay_record()
    queries = build_region_queries(replay)
    labels = [query.label for query in queries]
    assert "planning" in labels
    assert "tool_name" in labels
    assert "tool_arguments" in labels
    assert "arg.path" in labels
    assert "arg.content" in labels


def test_find_region_char_spans_finds_tool_arguments():
    replay = _sample_replay_record()
    assistant_text = render_assistant_message_fallback(replay["assistant_message"])
    spans = find_region_char_spans(assistant_text, build_region_queries(replay))
    labels = {span["label"] for span in spans}
    assert "planning" in labels
    assert "tool_arguments" in labels
    path_span = next(span for span in spans if span["label"] == "arg.path")
    assert path_span["text"] == "services.yaml"


def test_token_region_memberships_marks_assistant_tokens():
    offsets = [(0, 5), (5, 10), (10, 16), (16, 24), (24, 38)]
    region_char_spans = [{"label": "planning", "start_char": 0, "end_char": 6, "text": "assist"}]
    assistant_start, rows = token_region_memberships(offsets, 10, region_char_spans)
    assert assistant_start == 2
    assert rows[0]["assistant_token_index"] == 0
    assert rows[0]["regions"] == ["planning"]


def test_record_output_paths_uses_expected_suffix(tmp_path):
    act_path, meta_path = record_output_paths(tmp_path, "47_high", "safetensors")
    assert activation_suffix("safetensors") == "safetensors"
    assert act_path == tmp_path / "activations" / "47_high.safetensors"
    assert meta_path == tmp_path / "metadata" / "47_high.json"


def test_manifest_row_from_metadata_round_trips(tmp_path):
    output_dir = tmp_path / "out"
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True)
    metadata_path = metadata_dir / "47_high.json"
    metadata_path.write_text(
        """{
  "scenario_id": 47,
  "variant": "high",
  "label": "#47 high",
  "critical_step": 2,
  "critical_step_type": "tool_call",
  "assistant_token_count": 12,
  "layers": [0, 1, 2],
  "extraction_model": "Qwen/Qwen3-8B",
  "serialization_mode": "fallback",
  "activation_file": "activations/47_high.safetensors"
}"""
    )
    row = manifest_row_from_metadata(output_dir, metadata_path)
    assert row == {
        "scenario_id": 47,
        "variant": "high",
        "label": "#47 high",
        "critical_step": 2,
        "critical_step_type": "tool_call",
        "assistant_token_count": 12,
        "layers": [0, 1, 2],
        "serialization_mode": "fallback",
        "activation_file": "activations/47_high.safetensors",
        "metadata_file": "metadata/47_high.json",
    }


def test_metadata_matches_request_checks_config(tmp_path):
    output_dir = tmp_path / "out"
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True)
    metadata_path = metadata_dir / "47_high.json"
    metadata_path.write_text(
        """{
  "scenario_id": 47,
  "variant": "high",
  "label": "#47 high",
  "critical_step": 2,
  "critical_step_type": "tool_call",
  "assistant_token_count": 12,
  "layers": [0, 1, 2],
  "extraction_model": "Qwen/Qwen3-8B",
  "serialization_mode": "fallback",
  "activation_file": "activations/47_high.safetensors"
}"""
    )
    assert metadata_matches_request(
        metadata_path,
        layers=[0, 1, 2],
        model="Qwen/Qwen3-8B",
        fmt="safetensors",
    )
    assert not metadata_matches_request(
        metadata_path,
        layers=[0, 1],
        model="Qwen/Qwen3-8B",
        fmt="safetensors",
    )
    assert not metadata_matches_request(
        metadata_path,
        layers=[0, 1, 2],
        model="other/model",
        fmt="safetensors",
    )
