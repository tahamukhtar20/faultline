"""The Claude Code adapter. Nothing here runs a model."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from faultline.claude_code import (
    EXPECTED_TOOLS,
    SERVER_NAME,
    AgentError,
    IsolationError,
    ToolServer,
    _drive,
    build_command,
    check_isolation,
    isolated_environment,
)
from faultline.models import ActionKind


def observation(text: str) -> dict[str, Any]:
    return {
        "last_result": text,
        "remaining_steps": 5,
        "service_summary": "api running",
        "diff_summary": "no changes",
        "visible_test_summary": "not run",
    }


@pytest.fixture
def calls() -> list[tuple[ActionKind, str]]:
    return []


@pytest.fixture
def server(calls: list[tuple[ActionKind, str]]) -> Iterator[ToolServer]:
    def step(kind: ActionKind, argument: str) -> tuple[dict[str, Any], bool]:
        calls.append((kind, argument))
        return observation(f"ran {kind.value}"), kind is ActionKind.SUBMIT

    tool_server = ToolServer(step)
    yield tool_server
    tool_server.close()


def post(server: ToolServer, body: dict[str, Any], token: str | None = None) -> Any:
    request = urllib.request.Request(
        server.url,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token or server.token}"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read() or b"null")


def call(server: ToolServer, name: str, argument: str | None = None) -> dict[str, Any]:
    arguments = {} if argument is None else {"argument": argument}
    _, reply = post(
        server,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
    )
    result: dict[str, Any] = reply["result"]
    return result


def test_server_speaks_the_mcp_handshake(server: ToolServer) -> None:
    _, reply = post(server, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert reply["result"]["serverInfo"]["name"] == SERVER_NAME
    assert "tools" in reply["result"]["capabilities"]

    status, _ = post(server, {"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert status == 202

    _, listed = post(server, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {f"mcp__{SERVER_NAME}__{tool['name']}" for tool in listed["result"]["tools"]}
    assert names == EXPECTED_TOOLS


def test_tool_calls_reach_the_environment(
    server: ToolServer, calls: list[tuple[ActionKind, str]]
) -> None:
    result = call(server, "read", "orderflow/api.py")

    assert calls == [(ActionKind.READ, "orderflow/api.py")]
    assert not result["isError"]
    assert "ran read" in result["content"][0]["text"]
    assert "steps left: 5" in result["content"][0]["text"]


def test_no_tool_runs_after_the_episode_ends(
    server: ToolServer, calls: list[tuple[ActionKind, str]]
) -> None:
    call(server, "submit")

    result = call(server, "read", "orderflow/api.py")

    assert result["isError"]
    assert calls == [(ActionKind.SUBMIT, "")]


def test_unknown_tools_are_rejected(
    server: ToolServer, calls: list[tuple[ActionKind, str]]
) -> None:
    assert call(server, "bash", "cat /etc/passwd")["isError"]
    assert calls == []


def test_requests_without_the_token_are_refused(server: ToolServer) -> None:
    with pytest.raises(urllib.error.HTTPError) as refused:
        post(server, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, token="wrong")
    assert refused.value.code == 401


def test_built_in_tools_and_outside_configuration_are_switched_off(tmp_path: Path) -> None:
    command = build_command(
        "claude", model="m", effort="high", prompt="p", mcp_config=tmp_path / "c.json"
    )

    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in command
    assert "--disable-slash-commands" in command


def test_parent_session_settings_do_not_reach_the_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://parent")
    monkeypatch.setenv("CLAUDECODE", "1")

    environment = isolated_environment()

    assert "ANTHROPIC_BASE_URL" not in environment
    assert "CLAUDECODE" not in environment
    assert environment["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"


CLEAN_INIT = {
    "tools": sorted(EXPECTED_TOOLS),
    "mcp_servers": [{"name": SERVER_NAME, "status": "connected"}],
    "plugins": [{"name": "telemetry", "path": "builtin", "source": "telemetry@builtin"}],
    "slash_commands": [],
    "skills": [],
}


def test_a_clean_session_passes_the_isolation_check() -> None:
    check_isolation(CLEAN_INIT)


@pytest.mark.parametrize(
    "change",
    [
        {"tools": [*sorted(EXPECTED_TOOLS), "Bash"]},
        {"tools": [*sorted(EXPECTED_TOOLS), "Read"]},
        {"tools": sorted(EXPECTED_TOOLS)[1:]},
        {"mcp_servers": [{"name": SERVER_NAME, "status": "failed"}]},
        {"mcp_servers": [{"name": SERVER_NAME, "status": "connected"}, {"name": "gmail"}]},
        {"plugins": [{"name": "something", "source": "something@marketplace"}]},
        {"skills": ["something"]},
        {"slash_commands": ["something"]},
    ],
)
def test_any_extra_capability_stops_the_run(change: dict[str, Any]) -> None:
    with pytest.raises(IsolationError):
        check_isolation({**CLEAN_INIT, **change})


def fake_agent(tmp_path: Path, *events: dict[str, Any], exit_code: int = 0) -> list[str]:
    """A stand-in executable that prints stream-json events, as Claude Code does."""
    lines = "\n".join(json.dumps(event) for event in events)
    script = tmp_path / "agent.sh"
    script.write_text(f"#!/bin/sh\ncat <<'EOF'\n{lines}\nEOF\nexit {exit_code}\n")
    script.chmod(0o755)
    return [str(script)]


INIT_EVENT = {"type": "system", "subtype": "init", **CLEAN_INIT}


def test_a_finished_run_reports_its_usage(tmp_path: Path) -> None:
    result = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "num_turns": 7,
        "total_cost_usd": 0.5,
        "usage": {"input_tokens": 10},
    }

    run = _drive(fake_agent(tmp_path, INIT_EVENT, result), cwd=tmp_path, timeout=10)

    assert run.turns == 7
    assert run.usage == {"input_tokens": 10}


def test_a_rate_limit_is_not_counted_as_the_agent_failing(tmp_path: Path) -> None:
    result = {"type": "result", "subtype": "success", "is_error": True, "result": "limit reached"}

    with pytest.raises(AgentError, match="limit reached"):
        _drive(fake_agent(tmp_path, INIT_EVENT, result), cwd=tmp_path, timeout=10)


def test_a_crashed_agent_is_not_counted_as_the_agent_failing(tmp_path: Path) -> None:
    with pytest.raises(AgentError, match="exited early"):
        _drive(fake_agent(tmp_path, INIT_EVENT, exit_code=1), cwd=tmp_path, timeout=10)


def test_a_session_that_never_reports_its_tools_is_refused(tmp_path: Path) -> None:
    result = {"type": "result", "subtype": "success", "is_error": False}

    with pytest.raises(IsolationError):
        _drive(fake_agent(tmp_path, result), cwd=tmp_path, timeout=10)


def test_a_session_with_extra_tools_is_stopped_at_startup(tmp_path: Path) -> None:
    unsafe = {**INIT_EVENT, "tools": [*sorted(EXPECTED_TOOLS), "Bash"]}

    with pytest.raises(IsolationError):
        _drive(fake_agent(tmp_path, unsafe), cwd=tmp_path, timeout=10)
