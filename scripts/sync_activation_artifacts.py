"""Sync replay and activation artifacts between local storage and a remote box."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def remote_spec(host: str, path: Path | str) -> str:
    return f"{host}:{Path(path).as_posix()}"


def rsync_command(source: str, dest: str, *, delete: bool = False) -> list[str]:
    cmd = ["rsync", "-az", "--partial"]
    if delete:
        cmd.append("--delete")
    cmd.extend([source, dest])
    return cmd


def run_sync(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def sync_replay_inputs(args: argparse.Namespace) -> None:
    local_replay = Path(args.local_replay)
    remote_root = Path(args.remote_root)
    remote_replay = remote_root / args.remote_replay_rel

    run_sync(
        rsync_command(
            str(local_replay) + "/",
            remote_spec(args.remote, remote_replay),
            delete=False,
        ),
        dry_run=args.dry_run,
    )

    for rel_path in ("scripts/extract_activations.py", "pyproject.toml", "uv.lock"):
        local_path = Path(rel_path)
        remote_path = remote_root / rel_path
        run_sync(
            rsync_command(
                str(local_path),
                remote_spec(args.remote, remote_path),
                delete=False,
            ),
            dry_run=args.dry_run,
        )


def pull_activation_outputs(args: argparse.Namespace) -> None:
    local_output = Path(args.local_output)
    local_output.mkdir(parents=True, exist_ok=True)
    run_sync(
        rsync_command(
            remote_spec(args.remote, Path(args.remote_output)),
            str(local_output),
            delete=False,
        ),
        dry_run=args.dry_run,
    )


def push_activation_outputs(args: argparse.Namespace) -> None:
    local_output = Path(args.local_output)
    run_sync(
        rsync_command(
            str(local_output) + "/",
            remote_spec(args.remote, Path(args.remote_output)),
            delete=False,
        ),
        dry_run=args.dry_run,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync replay and activation artifacts to/from a remote box")
    subparsers = parser.add_subparsers(dest="command", required=True)

    push_replay = subparsers.add_parser("push-replay", help="Push local replay inputs and extractor code to the remote box")
    push_replay.add_argument("--remote", required=True, help="Remote host, e.g. ubuntu@216.81.245.36")
    push_replay.add_argument("--remote-root", required=True, help="Remote repo root, e.g. /home/ubuntu/consequence-awareness")
    push_replay.add_argument("--local-replay", required=True, help="Local replay directory to sync")
    push_replay.add_argument(
        "--remote-replay-rel",
        required=True,
        help="Replay directory path relative to the remote repo root",
    )
    push_replay.add_argument("--dry-run", action="store_true")
    push_replay.set_defaults(func=sync_replay_inputs)

    pull_outputs = subparsers.add_parser("pull-activations", help="Pull activation outputs from the remote box to local storage")
    pull_outputs.add_argument("--remote", required=True)
    pull_outputs.add_argument("--remote-output", required=True, help="Remote activation output directory")
    pull_outputs.add_argument("--local-output", required=True, help="Local canonical activation directory")
    pull_outputs.add_argument("--dry-run", action="store_true")
    pull_outputs.set_defaults(func=pull_activation_outputs)

    push_outputs = subparsers.add_parser("push-activations", help="Push locally stored activation outputs back to a remote box")
    push_outputs.add_argument("--remote", required=True)
    push_outputs.add_argument("--remote-output", required=True, help="Remote activation output directory")
    push_outputs.add_argument("--local-output", required=True, help="Local canonical activation directory")
    push_outputs.add_argument("--dry-run", action="store_true")
    push_outputs.set_defaults(func=push_activation_outputs)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
