"""Reconcile harness result directories.

The harness writes one per-attempt JSON file per scenario variant plus an
append-only `attempts.jsonl` summary. If a scenario variant is rerun, the JSON
file is overwritten but `attempts.jsonl` keeps both rows.

This script:
- detects duplicate scenario/variant rows in `attempts.jsonl`
- keeps the last row per scenario/variant as canonical
- verifies that canonical rows line up with the per-variant JSON files
- writes a deduped `attempts.reconciled.jsonl`
- writes a `reconciliation.json` report
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_attempt_rows(results_dir: Path) -> list[dict[str, Any]]:
    attempts_path = results_dir / "attempts.jsonl"
    return [
        json.loads(line)
        for line in attempts_path.read_text().splitlines()
        if line.strip()
    ]


def pair_key(row: dict[str, Any]) -> tuple[int, str]:
    return row["scenario_id"], row["variant"]


def canonical_rows(rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    canon: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        canon[pair_key(row)] = row
    return canon


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile harness results")
    parser.add_argument("--input", required=True, help="Results directory containing attempts.jsonl")
    args = parser.parse_args()

    results_dir = Path(args.input)
    rows = load_attempt_rows(results_dir)
    counts = Counter(pair_key(row) for row in rows)
    canon = canonical_rows(rows)

    duplicate_pairs = {
        f"{sid}_{variant}": count
        for (sid, variant), count in counts.items()
        if count > 1
    }

    file_mismatches: list[str] = []
    for (sid, variant), row in sorted(canon.items()):
        path = results_dir / f"{sid}_{variant}.json"
        if not path.exists():
            file_mismatches.append(f"missing:{sid}_{variant}")
            continue
        obj = json.loads(path.read_text())
        fields = ("duration_s", "total_steps", "total_prompt_tokens", "total_gen_tokens", "status")
        for field in fields:
            if obj.get(field) != row.get(field):
                file_mismatches.append(f"{sid}_{variant}:{field}")

    reconciled_path = results_dir / "attempts.reconciled.jsonl"
    with reconciled_path.open("w") as f:
        for key in sorted(canon):
            f.write(json.dumps(canon[key]) + "\n")

    report = {
        "attempt_rows": len(rows),
        "canonical_rows": len(canon),
        "duplicate_pairs": duplicate_pairs,
        "json_files": len(list(results_dir.glob("*_*.json"))),
        "file_mismatches": file_mismatches,
        "canonical_pair_ids": sorted({sid for sid, _ in canon}),
    }
    (results_dir / "reconciliation.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
