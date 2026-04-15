"""Extract assistant-turn activations from replay records.

This script consumes replay records produced by `prepare_replay.py` and writes:
- one activation file per replay record (all assistant-turn tokens, selected layers)
- one metadata JSON per replay record with token/region annotations
- a manifest JSONL summarising all extracted records

It prefers the tokenizer chat template when reconstructing the conversation. If
that fails, it falls back to a deterministic local serializer so the pipeline
remains usable during development.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SerializedReplay:
    prefix_text: str
    assistant_text: str
    full_text: str
    mode: str


@dataclass
class RegionQuery:
    label: str
    needle: str
    start_hint: int = 0


def iter_replay_records(input_dir: Path) -> list[Path]:
    records_dir = input_dir / "records"
    return sorted(records_dir.glob("*.json"))


def activation_suffix(fmt: str) -> str:
    return "pt" if fmt == "pt" else "safetensors"


def record_output_paths(output_dir: Path, stem: str, fmt: str) -> tuple[Path, Path]:
    activations_dir = output_dir / "activations"
    metadata_dir = output_dir / "metadata"
    return (
        activations_dir / f"{stem}.{activation_suffix(fmt)}",
        metadata_dir / f"{stem}.json",
    )


def manifest_row_from_metadata(output_dir: Path, metadata_path: Path) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text())
    return {
        "scenario_id": metadata["scenario_id"],
        "variant": metadata["variant"],
        "label": metadata["label"],
        "critical_step": metadata["critical_step"],
        "critical_step_type": metadata["critical_step_type"],
        "assistant_token_count": metadata["assistant_token_count"],
        "layers": metadata["layers"],
        "serialization_mode": metadata["serialization_mode"],
        "activation_file": metadata["activation_file"],
        "metadata_file": str(metadata_path.relative_to(output_dir)),
    }


def metadata_matches_request(
    metadata_path: Path,
    *,
    layers: list[int],
    model: str,
    fmt: str,
) -> bool:
    metadata = json.loads(metadata_path.read_text())
    expected_suffix = f".{activation_suffix(fmt)}"
    return (
        metadata.get("layers") == layers
        and metadata.get("extraction_model") == model
        and str(metadata.get("activation_file", "")).endswith(expected_suffix)
    )


def render_tool_call_fallback(tool_call: dict[str, Any]) -> str:
    """Render a tool call with stable formatting while preserving raw args."""
    name = json.dumps(tool_call["function"]["name"])
    arguments = tool_call["function"]["arguments"]
    return (
        "<tool_call>\n"
        f'{{"name": {name}, "arguments": {arguments}}}\n'
        "</tool_call>"
    )


def render_assistant_message_fallback(message: dict[str, Any]) -> str:
    parts: list[str] = []
    if message.get("content"):
        parts.append(message["content"])
    for tool_call in message.get("tool_calls") or []:
        parts.append(render_tool_call_fallback(tool_call))
    return "\n\n".join(parts)


def render_message_fallback(message: dict[str, Any]) -> str:
    role = message["role"]
    if role == "assistant":
        body = render_assistant_message_fallback(message)
    elif role == "tool":
        body = json.dumps(
            {
                "tool_call_id": message["tool_call_id"],
                "content": message["content"],
            },
            ensure_ascii=False,
        )
    else:
        body = message.get("content") or ""
    return f"<|{role}|>\n{body}"


def render_prefix_fallback(messages: list[dict[str, Any]]) -> str:
    parts = [render_message_fallback(message) for message in messages]
    parts.append("<|assistant|>\n")
    return "\n\n".join(parts)


def serialize_replay_record(tokenizer: Any, record: dict[str, Any]) -> SerializedReplay:
    """Build the serialized conversation before and through the assistant turn."""
    if tokenizer is not None:
        try:
            prefix_text = tokenizer.apply_chat_template(
                record["messages_before_assistant"],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                record["full_messages_through_assistant"],
                tokenize=False,
                add_generation_prompt=False,
            )
            if full_text.startswith(prefix_text):
                return SerializedReplay(
                    prefix_text=prefix_text,
                    assistant_text=full_text[len(prefix_text):],
                    full_text=full_text,
                    mode="chat_template",
                )
        except Exception:
            pass

    prefix_text = render_prefix_fallback(record["messages_before_assistant"])
    assistant_text = render_assistant_message_fallback(record["assistant_message"])
    return SerializedReplay(
        prefix_text=prefix_text,
        assistant_text=assistant_text,
        full_text=prefix_text + assistant_text,
        mode="fallback",
    )


def _iter_scalar_args(value: Any, prefix: str = "arg") -> list[tuple[str, str]]:
    """Flatten scalar argument values for span annotation."""
    pairs: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            pairs.extend(_iter_scalar_args(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            pairs.extend(_iter_scalar_args(child, f"{prefix}[{idx}]"))
    elif value is not None:
        needle = value if isinstance(value, str) else json.dumps(value)
        pairs.append((prefix, needle))
    return pairs


def build_region_queries(record: dict[str, Any]) -> list[RegionQuery]:
    """Describe semantically meaningful spans inside the assistant turn."""
    queries: list[RegionQuery] = []
    planning_text = record.get("planning_text")
    if planning_text:
        queries.append(RegionQuery(label="planning", needle=planning_text))

    tool_call = record.get("critical_tool_call")
    call_index = record.get("critical_tool_call_index")
    assistant_tool_calls = record.get("assistant_message", {}).get("tool_calls") or []
    if tool_call is None or call_index is None or call_index >= len(assistant_tool_calls):
        return queries

    assistant_tool_call = assistant_tool_calls[call_index]
    tool_name = assistant_tool_call["function"]["name"]
    arguments = assistant_tool_call["function"]["arguments"]
    queries.append(RegionQuery(label="tool_name", needle=tool_name))
    queries.append(RegionQuery(label="tool_arguments", needle=arguments))

    for label, needle in _iter_scalar_args(tool_call.get("arguments") or {}):
        queries.append(RegionQuery(label=label, needle=needle))

    return queries


def find_region_char_spans(assistant_text: str, queries: list[RegionQuery]) -> list[dict[str, Any]]:
    """Locate region queries inside the assistant text."""
    spans: list[dict[str, Any]] = []
    anchors: dict[str, int] = {}

    for query in queries:
        start_hint = query.start_hint
        if query.label.startswith("arg.") and "tool_arguments" in anchors:
            start_hint = anchors["tool_arguments"]
        elif query.label == "tool_name" and "planning" in anchors:
            start_hint = anchors["planning"]

        start = assistant_text.find(query.needle, start_hint)
        if start == -1 and start_hint:
            start = assistant_text.find(query.needle)
        if start == -1:
            continue

        spans.append(
            {
                "label": query.label,
                "start_char": start,
                "end_char": start + len(query.needle),
                "text": query.needle,
            }
        )
        anchors[query.label] = start

    return spans


def token_region_memberships(
    offsets: list[tuple[int, int]],
    assistant_char_start: int,
    region_char_spans: list[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    """Map token offsets to assistant-turn-relative region labels."""
    assistant_token_start = 0
    for idx, (_, end) in enumerate(offsets):
        if end > assistant_char_start:
            assistant_token_start = idx
            break

    token_rows: list[dict[str, Any]] = []
    for token_idx, (start, end) in enumerate(offsets):
        if end <= assistant_char_start:
            continue

        labels = [
            span["label"]
            for span in region_char_spans
            if end > assistant_char_start + span["start_char"]
            and start < assistant_char_start + span["end_char"]
        ]
        token_rows.append(
            {
                "token_index": token_idx,
                "assistant_token_index": len(token_rows),
                "start_char": start,
                "end_char": end,
                "regions": sorted(set(labels)),
            }
        )

    return assistant_token_start, token_rows


def parse_layers(spec: str, num_layers: int) -> list[int]:
    if spec == "all":
        return list(range(num_layers))
    layers = sorted({int(part.strip()) for part in spec.split(",") if part.strip()})
    if not layers:
        raise ValueError("No layers selected")
    if layers[0] < 0 or layers[-1] >= num_layers:
        raise ValueError(f"Layer out of range for model with {num_layers} layers")
    return layers


def choose_device(spec: str) -> str:
    if spec != "auto":
        return spec

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(spec: str):
    import torch

    mapping = {
        "auto": None,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[spec]


def save_activation_file(
    tensors: dict[str, Any],
    output_path: Path,
    fmt: str,
) -> None:
    if fmt == "pt":
        import torch

        torch.save(tensors, output_path)
        return

    from safetensors.torch import save_file

    save_file(tensors, str(output_path))


def extract_record(
    record_path: Path,
    output_dir: Path,
    tokenizer: Any,
    model: Any,
    layers: list[int],
    save_dtype: Any,
    fmt: str,
    device: str,
    model_name: str,
) -> dict[str, Any]:
    import torch

    record = json.loads(record_path.read_text())
    serialised = serialize_replay_record(tokenizer, record)
    encoding = tokenizer(
        serialised.full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    if "offset_mapping" not in encoding:
        raise ValueError("Tokenizer must support return_offsets_mapping for activation extraction")

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offsets = [tuple(pair) for pair in encoding["offset_mapping"][0].tolist()]
    assistant_char_start = len(serialised.prefix_text)
    region_char_spans = find_region_char_spans(serialised.assistant_text, build_region_queries(record))
    assistant_token_start, token_rows = token_region_memberships(offsets, assistant_char_start, region_char_spans)
    assistant_token_indices = [row["token_index"] for row in token_rows]

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states[1:]
    tensors: dict[str, Any] = {}
    for layer in layers:
        tensor = hidden_states[layer][0, assistant_token_indices, :].detach().to(save_dtype).cpu()
        tensors[f"layer_{layer}"] = tensor

    stem = f"{record['scenario_id']}_{record['variant']}"
    act_path, meta_path = record_output_paths(output_dir, stem, fmt)
    act_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    save_activation_file(tensors, act_path, fmt)

    token_texts = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    for row in token_rows:
        row["text"] = token_texts[row["token_index"]]
        row["token_id"] = int(input_ids[0, row["token_index"]].item())

    metadata = {
        "scenario_id": record["scenario_id"],
        "variant": record["variant"],
        "label": record["label"],
        "model_id": record["model_id"],
        "source_record_file": str(record_path),
        "serialization_mode": serialised.mode,
        "critical_step": record["critical_step"],
        "critical_step_type": record["critical_step_type"],
        "critical_tool_call_index": record["critical_tool_call_index"],
        "critical_tool_call": record["critical_tool_call"],
        "assistant_char_start": assistant_char_start,
        "assistant_token_start": assistant_token_start,
        "assistant_token_count": len(token_rows),
        "layers": layers,
        "extraction_model": model_name,
        "prefix_text": serialised.prefix_text,
        "assistant_text": serialised.assistant_text,
        "full_text_sha256": hashlib.sha256(serialised.full_text.encode("utf-8")).hexdigest(),
        "region_char_spans": region_char_spans,
        "tokens": token_rows,
        "activation_file": str(act_path.relative_to(output_dir)),
    }

    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    return {
        "scenario_id": record["scenario_id"],
        "variant": record["variant"],
        "label": record["label"],
        "critical_step": record["critical_step"],
        "critical_step_type": record["critical_step_type"],
        "assistant_token_count": len(token_rows),
        "layers": layers,
        "serialization_mode": serialised.mode,
        "activation_file": str(act_path.relative_to(output_dir)),
        "metadata_file": str(meta_path.relative_to(output_dir)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract activations from replay records")
    parser.add_argument("--input", required=True, help="Replay directory containing records/")
    parser.add_argument("--output", required=True, help="Directory to write activation artifacts into")
    parser.add_argument("--model", required=True, help="Model id or local path for AutoModelForCausalLM")
    parser.add_argument("--layers", default="all", help="Comma-separated layer list, or 'all'")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps")
    parser.add_argument("--load-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--save-dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--format", default="safetensors", choices=["safetensors", "pt"])
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of replay records")
    parser.add_argument("--records", default=None, help="Optional comma-separated list like 22_high,47_low")
    parser.add_argument("--skip-existing", action="store_true", help="Skip records whose activation and metadata files already exist")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing activation-extraction dependencies. Install with "
            "`uv pip install torch transformers safetensors`."
        ) from exc

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    record_paths = iter_replay_records(input_dir)

    if args.records:
        allowed = {item.strip() for item in args.records.split(",") if item.strip()}
        record_paths = [path for path in record_paths if path.stem in allowed]
    if args.limit is not None:
        record_paths = record_paths[:args.limit]

    if not record_paths:
        raise SystemExit("No replay records matched the requested selection")

    device = choose_device(args.device)
    load_dtype = resolve_dtype(args.load_dtype)
    save_dtype = resolve_dtype(args.save_dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_kwargs = {"trust_remote_code": True}
    if load_dtype is not None:
        model_kwargs["torch_dtype"] = load_dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.to(device)
    model.eval()

    layers = parse_layers(args.layers, model.config.num_hidden_layers)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    with manifest_path.open("w") as manifest:
        for record_path in record_paths:
            stem = record_path.stem
            act_path, meta_path = record_output_paths(output_dir, stem, args.format)
            skipped = False
            if args.skip_existing and act_path.exists() and meta_path.exists():
                if not metadata_matches_request(
                    meta_path,
                    layers=layers,
                    model=args.model,
                    fmt=args.format,
                ):
                    raise SystemExit(
                        f"Existing artifacts for {stem} do not match the requested extraction config"
                    )
                row = manifest_row_from_metadata(output_dir, meta_path)
                skipped = True
                print(f"Skipping {stem}; activation artifacts already exist")
            else:
                row = extract_record(
                    record_path=record_path,
                    output_dir=output_dir,
                    tokenizer=tokenizer,
                    model=model,
                    layers=layers,
                    save_dtype=save_dtype,
                    fmt=args.format,
                    device=device,
                    model_name=args.model,
                )
            manifest.write(json.dumps(row) + "\n")
            if not skipped:
                print(f"Extracted {stem} ({row['assistant_token_count']} assistant tokens)")


if __name__ == "__main__":
    main()
