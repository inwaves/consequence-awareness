"""Rank layers by low/high activation separation on matched scenario pairs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_metadata_map(metadata_dir: Path) -> dict[tuple[int, str], dict[str, Any]]:
    return {
        (record["scenario_id"], record["variant"]): record
        for record in (
            load_json(path)
            for path in sorted(metadata_dir.glob("*.json"))
        )
    }


def load_pair_buckets(summary_path: Path) -> dict[int, str]:
    return {row["scenario_id"]: row["bucket"] for row in load_json(summary_path)}


def token_region_mask(metadata: dict[str, Any], region: str) -> list[int]:
    indices: list[int] = []
    for token in metadata["tokens"]:
        labels = token.get("regions") or []
        if region == "all":
            indices.append(token["assistant_token_index"])
        elif region == "args":
            if any(label.startswith("arg.") for label in labels):
                indices.append(token["assistant_token_index"])
        elif region in labels:
            indices.append(token["assistant_token_index"])
    return indices


def cosine_distance(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return 1.0 - float(np.dot(a, b) / denom)


def l2_distance(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def pooled_region_vectors(metadata: dict[str, Any], tensor_map: dict[str, Any], regions: list[str]) -> dict[str, dict[int, Any]]:
    import numpy as np

    region_vectors: dict[str, dict[int, Any]] = {}
    for region in regions:
        indices = token_region_mask(metadata, region)
        if not indices:
            continue
        region_vectors[region] = {}
        for layer_key, tensor in tensor_map.items():
            layer = int(layer_key.removeprefix("layer_"))
            region_vectors[region][layer] = np.asarray(tensor, dtype=np.float32)[indices].mean(axis=0, dtype=np.float32)
    return region_vectors


def load_tensor_map(path: Path) -> dict[str, Any]:
    from safetensors.numpy import load_file

    return load_file(path)


def analyze_pairs(
    activations_dir: Path,
    metadata_dir: Path,
    summary_path: Path,
    regions: list[str],
    scenario_ids: list[int] | None,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    metadata_map = load_metadata_map(metadata_dir)
    buckets = load_pair_buckets(summary_path)

    if scenario_ids is None:
        scenario_ids = sorted({sid for sid, _ in metadata_map if buckets.get(sid) == "keep"})

    per_pair: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for sid in scenario_ids:
        low_meta = metadata_map.get((sid, "low"))
        high_meta = metadata_map.get((sid, "high"))
        if not low_meta or not high_meta:
            continue

        low_tensor_path = activations_dir / f"{sid}_low.safetensors"
        high_tensor_path = activations_dir / f"{sid}_high.safetensors"
        if not low_tensor_path.exists() or not high_tensor_path.exists():
            continue

        low_vectors = pooled_region_vectors(low_meta, load_tensor_map(low_tensor_path), regions)
        high_vectors = pooled_region_vectors(high_meta, load_tensor_map(high_tensor_path), regions)
        pair_result = {
            "scenario_id": sid,
            "bucket": buckets.get(sid),
            "regions": {},
        }

        for region in regions:
            if region not in low_vectors or region not in high_vectors:
                continue
            pair_result["regions"][region] = {}
            shared_layers = sorted(set(low_vectors[region]) & set(high_vectors[region]))
            for layer in shared_layers:
                low_vec = low_vectors[region][layer]
                high_vec = high_vectors[region][layer]
                row = {
                    "scenario_id": sid,
                    "bucket": buckets.get(sid),
                    "region": region,
                    "layer": layer,
                    "cosine_distance": cosine_distance(low_vec, high_vec),
                    "l2_distance": l2_distance(low_vec, high_vec),
                }
                rows.append(row)
                pair_result["regions"][region][layer] = row

        per_pair[sid] = pair_result

    return rows, per_pair


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import math

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not math.isfinite(row["cosine_distance"]) or not math.isfinite(row["l2_distance"]):
            continue
        grouped[(row["region"], row["layer"])].append(row)

    summary = []
    for (region, layer), items in grouped.items():
        summary.append(
            {
                "region": region,
                "layer": layer,
                "pairs": len(items),
                "mean_cosine_distance": sum(item["cosine_distance"] for item in items) / len(items),
                "mean_l2_distance": sum(item["l2_distance"] for item in items) / len(items),
            }
        )
    summary.sort(key=lambda row: (row["region"], -row["mean_cosine_distance"], row["layer"]))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank layers by activation separation across low/high pairs")
    parser.add_argument("--activations", required=True, help="Directory containing .safetensors activation files")
    parser.add_argument("--metadata", required=True, help="Directory containing metadata JSON files")
    parser.add_argument("--summary", required=True, help="Path to reverified-summary.json")
    parser.add_argument("--output", required=True, help="Directory for analysis outputs")
    parser.add_argument("--regions", default="all,planning,tool_name,tool_arguments,args", help="Comma-separated regions to analyze")
    parser.add_argument("--scenario-ids", default=None, help="Optional comma-separated scenario ids")
    args = parser.parse_args()

    regions = [part.strip() for part in args.regions.split(",") if part.strip()]
    scenario_ids = None
    if args.scenario_ids:
        scenario_ids = [int(part.strip()) for part in args.scenario_ids.split(",") if part.strip()]

    rows, per_pair = analyze_pairs(
        activations_dir=Path(args.activations),
        metadata_dir=Path(args.metadata),
        summary_path=Path(args.summary),
        regions=regions,
        scenario_ids=scenario_ids,
    )
    summary = summarize_rows(rows)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "layer_ranking.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "per_pair.json").write_text(json.dumps(per_pair, indent=2))

    for region in regions:
        top = [row for row in summary if row["region"] == region][:5]
        if not top:
            continue
        print(f"\n[{region}]")
        for row in top:
            print(
                f"layer {row['layer']}: "
                f"mean_cos={row['mean_cosine_distance']:.4f} "
                f"mean_l2={row['mean_l2_distance']:.4f} "
                f"pairs={row['pairs']}"
            )


if __name__ == "__main__":
    main()
