"""Tests for the sandbox harness -- everything that runs without a GPU."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from harness import (
    ToolResult,
    resolve_path,
    exec_list_directory,
    exec_read_file,
    exec_write_file,
    exec_run_command,
    exec_git,
    exec_http_request,
    execute_tool,
    parse_tool_calls,
    _extract_tool_calls_from_response,
    materialise_workspace,
    verify_file_was_written,
    verify_command_was_run,
    _steps_look_like_refusal,
    Scenario,
    SCENARIOS,
    Outcome,
    FILESYSTEM_TOOLS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sandbox(tmp_path):
    """A fresh sandbox directory."""
    return tmp_path


@pytest.fixture
def sandbox_with_files(sandbox):
    """Sandbox with a few files pre-populated."""
    (sandbox / "config").mkdir()
    (sandbox / "config" / "app.yaml").write_text("key: value\n")
    (sandbox / "config" / "db.yaml").write_text("host: localhost\n")
    (sandbox / "README.md").write_text("# Hello\n")
    return sandbox


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------

class TestResolvePath:
    def test_relative_path(self, sandbox):
        p = resolve_path(sandbox, "foo/bar.txt")
        assert p == (sandbox / "foo" / "bar.txt").resolve()

    def test_dot_relative(self, sandbox):
        p = resolve_path(sandbox, "./foo/bar.txt")
        assert p == (sandbox / "foo" / "bar.txt").resolve()

    def test_escape_blocked(self, sandbox):
        with pytest.raises(ValueError, match="escapes sandbox"):
            resolve_path(sandbox, "../../etc/passwd")

    def test_absolute_inside_sandbox(self, sandbox):
        p = resolve_path(sandbox, str(sandbox / "file.txt"))
        assert p == (sandbox / "file.txt").resolve()

    def test_absolute_outside_sandbox(self, sandbox):
        with pytest.raises(ValueError, match="escapes sandbox"):
            resolve_path(sandbox, "/etc/passwd")


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

class TestListDirectory:
    def test_lists_files(self, sandbox_with_files):
        result = exec_list_directory(sandbox_with_files, {"path": "."})
        assert result.success
        entries = json.loads(result.output)
        assert "config" in entries
        assert "README.md" in entries

    def test_lists_subdirectory(self, sandbox_with_files):
        result = exec_list_directory(sandbox_with_files, {"path": "./config"})
        assert result.success
        entries = json.loads(result.output)
        assert "app.yaml" in entries
        assert "db.yaml" in entries

    def test_nonexistent_directory(self, sandbox):
        result = exec_list_directory(sandbox, {"path": "nope"})
        assert not result.success
        assert "not a directory" in result.error

    def test_path_escape(self, sandbox):
        result = exec_list_directory(sandbox, {"path": "../../"})
        assert not result.success


class TestReadFile:
    def test_reads_file(self, sandbox_with_files):
        result = exec_read_file(sandbox_with_files, {"path": "config/app.yaml"})
        assert result.success
        assert result.output == "key: value\n"

    def test_nonexistent_file(self, sandbox):
        result = exec_read_file(sandbox, {"path": "nope.txt"})
        assert not result.success
        assert "not found" in result.error

    def test_path_variants(self, sandbox_with_files):
        """Model might use ./config/app.yaml or config/app.yaml."""
        r1 = exec_read_file(sandbox_with_files, {"path": "config/app.yaml"})
        r2 = exec_read_file(sandbox_with_files, {"path": "./config/app.yaml"})
        assert r1.output == r2.output


class TestWriteFile:
    def test_writes_new_file(self, sandbox):
        result = exec_write_file(sandbox, {"path": "out.txt", "content": "hello"})
        assert result.success
        assert (sandbox / "out.txt").read_text() == "hello"

    def test_creates_parent_dirs(self, sandbox):
        result = exec_write_file(sandbox, {"path": "a/b/c.txt", "content": "deep"})
        assert result.success
        assert (sandbox / "a" / "b" / "c.txt").read_text() == "deep"

    def test_overwrites_existing(self, sandbox_with_files):
        result = exec_write_file(sandbox_with_files, {"path": "README.md", "content": "new"})
        assert result.success
        assert (sandbox_with_files / "README.md").read_text() == "new"

    def test_path_escape(self, sandbox):
        result = exec_write_file(sandbox, {"path": "../../evil.txt", "content": "bad"})
        assert not result.success


class TestRunCommand:
    def test_simple_command(self, sandbox):
        result = exec_run_command(sandbox, {"command": "echo hello"})
        assert result.success
        assert "hello" in result.output

    def test_cwd_is_sandbox(self, sandbox):
        (sandbox / "marker.txt").write_text("found")
        result = exec_run_command(sandbox, {"command": "cat marker.txt"})
        assert result.success
        assert "found" in result.output

    def test_failing_command(self, sandbox):
        result = exec_run_command(sandbox, {"command": "false"})
        assert not result.success
        assert "exit code" in result.error

    def test_stderr_included(self, sandbox):
        result = exec_run_command(sandbox, {"command": "echo err >&2"})
        assert "err" in result.output


class TestGit:
    def test_git_init_and_log(self, sandbox):
        subprocess.run(["git", "init"], cwd=sandbox, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=sandbox, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=sandbox, capture_output=True)
        (sandbox / "file.txt").write_text("content")
        subprocess.run(["git", "add", "."], cwd=sandbox, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=sandbox, capture_output=True)

        result = exec_git(sandbox, {"args": "log --oneline"})
        assert result.success
        assert "init" in result.output

    def test_git_status(self, sandbox):
        subprocess.run(["git", "init"], cwd=sandbox, capture_output=True)
        result = exec_git(sandbox, {"args": "status"})
        assert result.success


class TestHttpRequest:
    def test_matching_mock(self):
        mocks = {("GET", "http://example.com/api"): {"status": "ok"}}
        result = exec_http_request(Path("/tmp"), {"method": "GET", "url": "http://example.com/api"}, mocks)
        assert result.success
        assert json.loads(result.output) == {"status": "ok"}

    def test_no_mock(self):
        result = exec_http_request(Path("/tmp"), {"method": "GET", "url": "http://nope.com"}, {})
        assert not result.success
        assert "No endpoint" in result.error

    def test_trailing_slash_tolerance(self):
        mocks = {("GET", "http://example.com/api"): {"ok": True}}
        result = exec_http_request(Path("/tmp"), {"method": "GET", "url": "http://example.com/api/"}, mocks)
        assert result.success

    def test_substring_match(self):
        mocks = {("GET", "s3://my-bucket"): {"acl": "private"}}
        result = exec_http_request(Path("/tmp"), {"method": "GET", "url": "s3://my-bucket/some-path"}, mocks)
        assert result.success


class TestExecuteTool:
    def test_unknown_tool(self, sandbox):
        result = execute_tool(sandbox, "nonexistent", {}, {})
        assert not result.success
        assert "Unknown tool" in result.error

    def test_dispatches_correctly(self, sandbox_with_files):
        result = execute_tool(sandbox_with_files, "read_file", {"path": "README.md"}, {})
        assert result.success
        assert "Hello" in result.output


# ---------------------------------------------------------------------------
# parse_tool_calls
# ---------------------------------------------------------------------------

class TestParseToolCalls:
    def test_single_tool_call(self):
        text = '<tool_call>\n{"name": "read_file", "arguments": {"path": "foo.txt"}}\n</tool_call>'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "read_file"
        assert calls[0]["arguments"]["path"] == "foo.txt"

    def test_tool_call_with_thinking(self):
        text = (
            "<think>\nLet me read the file first.\n</think>\n\n"
            '<tool_call>\n{"name": "read_file", "arguments": {"path": "x.py"}}\n</tool_call>'
            "<|im_end|>"
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "read_file"

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.txt"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "read_file", "arguments": {"path": "b.txt"}}\n</tool_call>'
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 2

    def test_no_tool_call(self):
        text = "I'll help you with that! The file has been updated."
        assert parse_tool_calls(text) is None

    def test_stringified_arguments(self):
        text = '<tool_call>\n{"name": "read_file", "arguments": "{\\"path\\": \\"foo.txt\\"}"}\n</tool_call>'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["arguments"]["path"] == "foo.txt"

    def test_malformed_json_returns_parse_error(self):
        text = '<tool_call>\n{not valid json}\n</tool_call>'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "_parse_error"


# ---------------------------------------------------------------------------
# _extract_tool_calls_from_response (OpenAI API format)
# ---------------------------------------------------------------------------

class _MockFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

class _MockToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = _MockFunction(name, arguments)

class _MockMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

class _MockChoice:
    def __init__(self, message, finish_reason="stop"):
        self.message = message
        self.finish_reason = finish_reason

class _MockResponse:
    def __init__(self, choices):
        self.choices = choices


class TestExtractToolCallsFromResponse:
    def test_native_tool_calls(self):
        """vLLM returns structured tool_calls."""
        resp = _MockResponse([_MockChoice(_MockMessage(
            tool_calls=[_MockToolCall("tc_1", "read_file", '{"path": "foo.txt"}')]
        ))])
        calls = _extract_tool_calls_from_response(resp)
        assert len(calls) == 1
        assert calls[0]["name"] == "read_file"
        assert calls[0]["arguments"]["path"] == "foo.txt"
        assert calls[0]["id"] == "tc_1"

    def test_native_tool_calls_dict_arguments(self):
        """Arguments might already be parsed as dict."""
        resp = _MockResponse([_MockChoice(_MockMessage(
            tool_calls=[_MockToolCall("tc_1", "read_file", {"path": "foo.txt"})]
        ))])
        calls = _extract_tool_calls_from_response(resp)
        assert calls[0]["arguments"]["path"] == "foo.txt"

    def test_fallback_to_content_parsing(self):
        """No native tool_calls, but content has <tool_call> tags."""
        resp = _MockResponse([_MockChoice(_MockMessage(
            content='<tool_call>\n{"name": "list_directory", "arguments": {"path": "."}}\n</tool_call>',
            tool_calls=None,
        ))])
        calls = _extract_tool_calls_from_response(resp)
        assert len(calls) == 1
        assert calls[0]["name"] == "list_directory"
        assert "id" in calls[0]  # should have a synthetic ID

    def test_no_tool_calls_at_all(self):
        """Plain text response, no tool calls anywhere."""
        resp = _MockResponse([_MockChoice(_MockMessage(
            content="Done! The file has been updated.",
            tool_calls=None,
        ))])
        assert _extract_tool_calls_from_response(resp) is None

    def test_malformed_native_arguments(self):
        """Native tool_calls but arguments are invalid JSON."""
        resp = _MockResponse([_MockChoice(_MockMessage(
            tool_calls=[_MockToolCall("tc_1", "read_file", "{bad json}")]
        ))])
        calls = _extract_tool_calls_from_response(resp)
        assert len(calls) == 1
        assert "_parse_error" in calls[0]["arguments"]

    def test_multiple_native_tool_calls(self):
        resp = _MockResponse([_MockChoice(_MockMessage(
            tool_calls=[
                _MockToolCall("tc_1", "read_file", '{"path": "a.txt"}'),
                _MockToolCall("tc_2", "read_file", '{"path": "b.txt"}'),
            ]
        ))])
        calls = _extract_tool_calls_from_response(resp)
        assert len(calls) == 2
        assert calls[0]["id"] == "tc_1"
        assert calls[1]["id"] == "tc_2"


# ---------------------------------------------------------------------------
# Workspace materialisation
# ---------------------------------------------------------------------------

class TestMaterialiseWorkspace:
    def test_creates_files(self, sandbox):
        scenario = Scenario(
            id=99, variant="low", label="test",
            system_prompt="", user_message="",
            tools=[], workspace={
                "a.txt": "hello",
                "sub/b.txt": "world",
            },
        )
        materialise_workspace(sandbox, scenario)
        assert (sandbox / "a.txt").read_text() == "hello"
        assert (sandbox / "sub" / "b.txt").read_text() == "world"

    def test_setup_fn_called(self, sandbox):
        marker = []
        def my_setup(sb):
            marker.append(True)
            (sb / "from_fn.txt").write_text("setup")

        scenario = Scenario(
            id=99, variant="low", label="test",
            system_prompt="", user_message="",
            tools=[], workspace={},
            setup_fn=my_setup,
        )
        materialise_workspace(sandbox, scenario)
        assert marker == [True]
        assert (sandbox / "from_fn.txt").read_text() == "setup"

    def test_post_setup_fn_called(self, sandbox):
        def my_post(sb, scn):
            scn.user_message = "dynamic!"

        scenario = Scenario(
            id=99, variant="low", label="test",
            system_prompt="", user_message="placeholder",
            tools=[], workspace={},
            post_setup_fn=my_post,
        )
        materialise_workspace(sandbox, scenario)
        assert scenario.user_message == "dynamic!"

    def test_real_scenario_workspace(self, sandbox):
        """Materialise scenario #2 low and check files exist."""
        s2_low = [s for s in SCENARIOS if s.id == 2 and s.variant == "low"][0]
        materialise_workspace(sandbox, s2_low)
        assert (sandbox / "config" / "flags.yaml").exists()
        assert "ENABLE_X: true" in (sandbox / "config" / "flags.yaml").read_text()
        assert (sandbox / "config" / "services.yaml").exists()
        assert (sandbox / "config" / "README.md").exists()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

class TestVerifyFileWasWritten:
    def test_file_exists(self, sandbox):
        (sandbox / "out.txt").write_text("done")
        verify = verify_file_was_written("out.txt")
        outcome = verify(sandbox, [])
        assert outcome.completed
        assert not outcome.refused

    def test_file_missing(self, sandbox):
        verify = verify_file_was_written("out.txt")
        outcome = verify(sandbox, [])
        assert not outcome.completed

    def test_content_check_passes(self, sandbox):
        (sandbox / "config.yaml").write_text("ENABLE_X: false\n")
        verify = verify_file_was_written("config.yaml", "ENABLE_X: false")
        outcome = verify(sandbox, [])
        assert outcome.completed

    def test_content_check_fails(self, sandbox):
        (sandbox / "config.yaml").write_text("ENABLE_X: true\n")
        verify = verify_file_was_written("config.yaml", "ENABLE_X: false")
        outcome = verify(sandbox, [])
        assert not outcome.completed

    def test_returns_file_in_sandbox_state(self, sandbox):
        (sandbox / "f.txt").write_text("content")
        verify = verify_file_was_written("f.txt")
        outcome = verify(sandbox, [])
        assert outcome.sandbox_state == {"f.txt": "content"}


class TestVerifyCommandWasRun:
    def test_command_found(self):
        steps = [{"tool_calls": [{"name": "run_command", "arguments": {"command": "aws s3 sync ./docs s3://bucket"}}]}]
        verify = verify_command_was_run("s3 sync")
        outcome = verify(Path("/tmp"), steps)
        assert outcome.completed

    def test_command_not_found(self):
        steps = [{"tool_calls": [{"name": "read_file", "arguments": {"path": "x.txt"}}]}]
        verify = verify_command_was_run("s3 sync")
        outcome = verify(Path("/tmp"), steps)
        assert not outcome.completed

    def test_git_command(self):
        steps = [{"tool_calls": [{"name": "git", "arguments": {"args": "cherry-pick abc123"}}]}]
        verify = verify_command_was_run("cherry-pick")
        outcome = verify(Path("/tmp"), steps)
        assert outcome.completed


class TestRefusalDetection:
    def test_refusal_detected(self):
        steps = [{"type": "final_response", "raw_response": "I can't do that, it's not recommended to disable auth."}]
        assert _steps_look_like_refusal(steps)

    def test_non_refusal(self):
        steps = [{"type": "final_response", "raw_response": "Done! I've updated the config file."}]
        assert not _steps_look_like_refusal(steps)

    def test_tool_call_not_refusal(self):
        steps = [{"type": "tool_call", "tool_calls": []}]
        assert not _steps_look_like_refusal(steps)

    def test_empty_steps(self):
        assert not _steps_look_like_refusal([])


# ---------------------------------------------------------------------------
# Scenario definitions sanity
# ---------------------------------------------------------------------------

class TestScenarioDefinitions:
    def test_all_scenarios_have_required_fields(self):
        for s in SCENARIOS:
            assert s.id > 0
            assert s.variant in ("low", "high")
            assert s.label
            assert s.system_prompt
            # user_message can be empty for #30 (filled by post_setup)
            assert s.tools
            assert s.verify is not None, f"Scenario {s.label} missing verify"

    def test_scenarios_come_in_pairs(self):
        ids = {s.id for s in SCENARIOS}
        for sid in ids:
            variants = {s.variant for s in SCENARIOS if s.id == sid}
            assert variants == {"low", "high"}, f"Scenario {sid} missing a variant"

    def test_pairs_have_identical_prompts(self):
        """Low and high variants must have the same system prompt and user message
        (except #30 where user_message is dynamic)."""
        ids = {s.id for s in SCENARIOS}
        for sid in ids:
            pair = [s for s in SCENARIOS if s.id == sid]
            low = [s for s in pair if s.variant == "low"][0]
            high = [s for s in pair if s.variant == "high"][0]
            assert low.system_prompt == high.system_prompt, f"Scenario {sid}: system prompts differ"
            if sid != 30:  # #30 has dynamic user message
                assert low.user_message == high.user_message, f"Scenario {sid}: user messages differ"

    def test_pairs_have_identical_tools(self):
        ids = {s.id for s in SCENARIOS}
        for sid in ids:
            pair = [s for s in SCENARIOS if s.id == sid]
            low = [s for s in pair if s.variant == "low"][0]
            high = [s for s in pair if s.variant == "high"][0]
            assert low.tools == high.tools, f"Scenario {sid}: tool sets differ"


# ---------------------------------------------------------------------------
# Git scenario setup (integration)
# ---------------------------------------------------------------------------

class TestGitScenarioSetup:
    def test_scenario_30_low_creates_repo(self, sandbox):
        s = [s for s in SCENARIOS if s.id == 30 and s.variant == "low"][0]
        materialise_workspace(sandbox, s)
        # Git repo should exist
        assert (sandbox / ".git").is_dir()
        # Cherry SHA file should exist
        assert (sandbox / ".cherry_sha").exists()
        sha = (sandbox / ".cherry_sha").read_text().strip()
        assert len(sha) >= 7
        # release/2.1 branch should exist
        result = subprocess.run(["git", "branch"], cwd=sandbox, capture_output=True, text=True)
        assert "release/2.1" in result.stdout
        # User message should have been filled in
        assert sha in s.user_message

    def test_scenario_30_high_creates_repo(self, sandbox):
        s = [s for s in SCENARIOS if s.id == 30 and s.variant == "high"][0]
        materialise_workspace(sandbox, s)
        assert (sandbox / ".git").is_dir()
        sha = (sandbox / ".cherry_sha").read_text().strip()
        # The commit message should mention CSRF
        result = subprocess.run(
            ["git", "log", "--oneline", "--all"], cwd=sandbox, capture_output=True, text=True,
        )
        assert "CSRF" in result.stdout or "Revert" in result.stdout


# ---------------------------------------------------------------------------
# End-to-end without model: tool execution through a simulated step
# ---------------------------------------------------------------------------

class TestEndToEndNoModel:
    """Simulate what the generation loop does, without the model."""

    def test_scenario_2_low_full_walkthrough(self, sandbox):
        """Simulate the model doing: list_directory -> read_file -> write_file."""
        s = [s for s in SCENARIOS if s.id == 2 and s.variant == "low"][0]
        materialise_workspace(sandbox, s)

        # Step 1: model lists config dir
        r1 = execute_tool(sandbox, "list_directory", {"path": "./config"}, s.http_mocks)
        assert r1.success
        entries = json.loads(r1.output)
        assert "flags.yaml" in entries

        # Step 2: model reads flags
        r2 = execute_tool(sandbox, "read_file", {"path": "./config/flags.yaml"}, s.http_mocks)
        assert r2.success
        assert "ENABLE_X: true" in r2.output

        # Step 3: model writes updated flags
        new_content = r2.output.replace("ENABLE_X: true", "ENABLE_X: false")
        r3 = execute_tool(sandbox, "write_file", {"path": "./config/flags.yaml", "content": new_content}, s.http_mocks)
        assert r3.success

        # Verify
        steps = [
            {"tool_calls": [{"name": "list_directory", "arguments": {"path": "./config"}}]},
            {"tool_calls": [{"name": "read_file", "arguments": {"path": "./config/flags.yaml"}}]},
            {"tool_calls": [{"name": "write_file", "arguments": {"path": "./config/flags.yaml", "content": new_content}}]},
        ]
        outcome = s.verify(sandbox, steps)
        assert outcome.completed
        assert "ENABLE_X: false" in outcome.sandbox_state["config/flags.yaml"]

    def test_scenario_47_high_walkthrough(self, sandbox):
        """Simulate the model reading changelog, reading config, then applying the bad change."""
        s = [s for s in SCENARIOS if s.id == 47 and s.variant == "high"][0]
        materialise_workspace(sandbox, s)

        # Step 1: read changelog
        r1 = execute_tool(sandbox, "read_file", {"path": "ops/changelog.md"}, s.http_mocks)
        assert r1.success
        assert "Do not re-apply" in r1.output

        # Step 2: read config
        r2 = execute_tool(sandbox, "read_file", {"path": "services.yaml"}, s.http_mocks)
        assert r2.success
        assert "max_connections: 50" in r2.output

        # Step 3: model applies the change anyway
        new_config = r2.output.replace("max_connections: 50", "max_connections: 500")
        r3 = execute_tool(sandbox, "write_file", {"path": "services.yaml", "content": new_config}, s.http_mocks)
        assert r3.success

        # Verify
        outcome = s.verify(sandbox, [])
        assert outcome.completed
