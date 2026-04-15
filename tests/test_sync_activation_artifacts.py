"""Tests for activation artifact sync helpers."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from sync_activation_artifacts import (  # noqa: E402
    remote_spec,
    rsync_command,
)


def test_remote_spec_formats_host_and_path():
    assert remote_spec("ubuntu@216.81.245.36", "/home/ubuntu/out") == "ubuntu@216.81.245.36:/home/ubuntu/out"


def test_rsync_command_defaults():
    assert rsync_command("src/", "dst") == ["rsync", "-az", "--partial", "src/", "dst"]


def test_rsync_command_with_delete():
    assert rsync_command("src/", "dst", delete=True) == ["rsync", "-az", "--partial", "--delete", "src/", "dst"]
