"""A real coding agent as a baseline: Claude Code, driving the environment through MCP.

The agent's only tools are Faultline's, served from this process over a local MCP endpoint. Claude
Code's built-in file and shell tools are removed, and it runs from an empty folder, so it cannot
read this repository's gold patches or private scenarios. The session's startup report is checked
against that expectation before the first action, and a run that fails the check is abandoned.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from faultline.env import FaultlineEnv
from faultline.models import ActionKind

DEFAULT_MODEL = "claude-opus-5-5"
SERVER_NAME = "faultline"
TOOL_PREFIX = f"mcp__{SERVER_NAME}__"
_PROTOCOL_VERSION = "2025-06-18"
_TOOL_TIMEOUT_MILLISECONDS = 600_000

SYSTEM_PROMPT = """You are repairing a production order service. The service is a FastAPI app \
with a transactional outbox and delivery workers, backed by SQLite, that calls an external payment \
and shipping provider.

A hidden verifier will score your final code against scenarios you cannot see. It checks real \
behavior through the provider's own records, including crashes, retries, slow responses and old \
data. The visible tests may already pass while the service is still broken, so reason about \
correctness and do not stop at green tests. Reporting status without doing the work, disabling \
features, or changing tests will fail.

Investigate with the tools, fix the root causes, and call submit when you are confident. You have \
a limited number of tool calls, and every call uses one. Submit before you run out."""


def _tool(kind: ActionKind, description: str, argument: str | None = None) -> dict[str, Any]:
    properties = {"argument": {"type": "string", "description": argument}} if argument else {}
    return {
        "name": kind.value,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


TOOLS = (
    _tool(
        ActionKind.READ,
        "Read a workspace file with line numbers.",
        "path, path:line or path:first-last",
    ),
    _tool(
        ActionKind.SEARCH,
        "Search all workspace files for exact text. Empty text lists every file.",
        "text to find",
    ),
    _tool(
        ActionKind.RUN,
        "Run the visible tests.",
        "pytest -q, optionally followed by a path under tests/",
    ),
    _tool(
        ActionKind.LOGS,
        "Read recent logs and the provider's records.",
        "all, api, worker or provider",
    ),
    _tool(
        ActionKind.QUERY,
        "Run a read-only SQL query against the live database.",
        "one SELECT statement",
    ),
    _tool(
        ActionKind.PATCH,
        "Replace text that occurs exactly once in a file.",
        'JSON object with string fields "path", "old" and "new"',
    ),
    _tool(
        ActionKind.RESTART,
        "Restart the live service on the current code and resend the incident traffic.",
    ),
    _tool(ActionKind.CHECKPOINT, "Save the current files and data. Returns a checkpoint ID."),
    _tool(
        ActionKind.RESTORE,
        "Return files and data to a checkpoint. The tool budget keeps counting.",
        "checkpoint ID",
    ),
    _tool(
        ActionKind.SUBMIT,
        "Submit the current code for final verification. This ends the episode.",
    ),
)
EXPECTED_TOOLS = frozenset(f"{TOOL_PREFIX}{tool['name']}" for tool in TOOLS)


class IsolationError(RuntimeError):
    """The agent had access to something beyond the environment's tools."""


class AgentError(RuntimeError):
    """The agent could not run, for example on a rate limit. This episode says nothing about it."""


def status_line(observation: dict[str, Any]) -> str:
    return (
        f"[steps left: {int(observation['remaining_steps'])}; "
        f"services: {observation['service_summary']}; "
        f"changes: {observation['diff_summary']}; "
        f"last tests: {observation['visible_test_summary']}]"
    )


Step = Callable[[ActionKind, str], tuple[dict[str, Any], bool]]


class ToolServer:
    """The MCP endpoint: JSON-RPC over HTTP on loopback, with a bearer token for each episode."""

    def __init__(self, step: Step):
        self._step = step
        self._lock = threading.Lock()
        self.done = False
        self.token = secrets.token_urlsafe(32)
        self.calls = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                if self.headers.get("Authorization") != f"Bearer {owner.token}":
                    self._reply(401, None)
                    return
                length = int(self.headers.get("Content-Length", 0))
                try:
                    request = json.loads(self.rfile.read(length))
                except ValueError:
                    self._reply(400, None)
                    return
                response = owner.handle(request)
                if response is None:
                    self._reply(202, None)
                else:
                    self._reply(200, response)

            def do_GET(self) -> None:
                self._reply(405, None)

            def do_DELETE(self) -> None:
                self._reply(405, None)

            def _reply(self, status: int, body: dict[str, Any] | None) -> None:
                payload = b"" if body is None else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/mcp"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        if "id" not in request:
            return None
        if method == "initialize":
            result: dict[str, Any] = {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": "1.0.0"},
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": list(TOOLS)}
        elif method == "tools/call":
            result = self._call(request.get("params", {}))
        else:
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "error": {"code": -32601, "message": f"unknown method: {method}"},
            }
        return {"jsonrpc": "2.0", "id": request["id"], "result": result}

    def _call(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self.done:
                return _tool_text("not run: the episode has ended", error=True)
            try:
                kind = ActionKind(str(params.get("name")))
            except ValueError:
                return _tool_text(f"unknown tool: {params.get('name')}", error=True)
            argument = params.get("arguments", {}).get("argument", "")
            self.calls += 1
            observation, self.done = self._step(kind, str(argument))
            return _tool_text(f"{observation['last_result']}\n\n{status_line(observation)}")

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _tool_text(text: str, *, error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": error}


def find_executable() -> str:
    """Claude Code's path, from CLAUDE_CODE_EXECPATH or the PATH."""
    executable = os.environ.get("CLAUDE_CODE_EXECPATH") or shutil.which("claude")
    if executable is None:
        raise FileNotFoundError("Claude Code is not installed: `claude` is not on the PATH")
    return executable


def isolated_environment() -> dict[str, str]:
    """The agent's process environment, without anything inherited from a parent Claude session."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("CLAUDE", "ANTHROPIC"))
    }
    environment.update(
        {
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
            "MCP_TOOL_TIMEOUT": str(_TOOL_TIMEOUT_MILLISECONDS),
        }
    )
    return environment


def build_command(
    executable: str, *, model: str, effort: str, prompt: str, mcp_config: Path
) -> list[str]:
    return [
        executable,
        "--print",
        prompt,
        "--model",
        model,
        "--effort",
        effort,
        "--output-format",
        "stream-json",
        "--verbose",
        "--system-prompt",
        SYSTEM_PROMPT,
        "--tools",
        "",
        "--allowedTools",
        f"mcp__{SERVER_NAME}",
        "--mcp-config",
        str(mcp_config),
        "--strict-mcp-config",
        "--setting-sources",
        "",
        "--disable-slash-commands",
        "--no-session-persistence",
    ]


def check_isolation(init: dict[str, Any]) -> None:
    """Refuse to continue unless the agent's only tools and sources are the expected ones."""
    tools = set(init.get("tools", []))
    if tools != EXPECTED_TOOLS:
        raise IsolationError(
            f"unexpected tools: {sorted(tools - EXPECTED_TOOLS)}, "
            f"missing: {sorted(EXPECTED_TOOLS - tools)}"
        )
    servers = init.get("mcp_servers", [])
    if [server.get("name") for server in servers] != [SERVER_NAME]:
        raise IsolationError(f"unexpected MCP servers: {servers}")
    if servers[0].get("status") != "connected":
        raise IsolationError(f"the environment's MCP server did not connect: {servers}")
    for name in ("slash_commands", "skills"):
        if init.get(name):
            raise IsolationError(f"unexpected {name}: {init[name]}")
    # Claude Code's own built-in plugins are always listed. Installed ones must not be.
    installed = [p for p in init.get("plugins", []) if not p.get("source", "").endswith("@builtin")]
    if installed:
        raise IsolationError(f"unexpected plugins: {installed}")


@dataclass(slots=True)
class AgentRun:
    usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    turns: int = 0
    stop: str = ""


def make_claude_code_policy(
    model: str = DEFAULT_MODEL,
    effort: str = "high",
    usage_log: Path | None = None,
) -> Callable[[FaultlineEnv, dict[str, Any], Step], None]:
    def policy(env: FaultlineEnv, observation: dict[str, Any], step: Step) -> None:
        server = ToolServer(step)
        prompt = f"Issue report:\n{observation['issue']}\n\n{status_line(observation)}"
        with tempfile.TemporaryDirectory(prefix="faultline-agent-") as directory:
            config = Path(directory) / ".mcp-config.json"
            config.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            SERVER_NAME: {
                                "type": "http",
                                "url": server.url,
                                "headers": {"Authorization": f"Bearer {server.token}"},
                            }
                        }
                    }
                )
            )
            config.chmod(0o600)
            workspace = Path(directory) / "empty"
            workspace.mkdir()
            run = _drive(
                build_command(
                    find_executable(), model=model, effort=effort, prompt=prompt, mcp_config=config
                ),
                cwd=workspace,
                timeout=float(env.instance.manifest.time_budget_seconds),
            )
        server.close()
        if usage_log is not None:
            usage_log.parent.mkdir(parents=True, exist_ok=True)
            record = {"task_id": env.instance.manifest.task_id, "model": model, **asdict(run)}
            with usage_log.open("a") as output:
                output.write(json.dumps(record) + "\n")

    return policy


def _drive(command: list[str], *, cwd: Path, timeout: float) -> AgentRun:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=isolated_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    timed_out = threading.Event()

    def stop_at_deadline() -> None:
        timed_out.set()
        process.kill()

    timer = threading.Timer(timeout, stop_at_deadline)
    timer.start()
    run = AgentRun()
    checked = False
    finished = False
    try:
        assert process.stdout is not None
        for line in process.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "system" and event.get("subtype") == "init":
                check_isolation(event)
                checked = True
            elif event.get("type") == "result":
                finished = True
                if event.get("is_error"):
                    raise AgentError(f"{event.get('subtype')}: {event.get('result')}")
                run.usage = event.get("usage", {})
                run.cost_usd = float(event.get("total_cost_usd", 0.0))
                run.turns = int(event.get("num_turns", 0))
                run.stop = str(event.get("subtype", ""))
    finally:
        timer.cancel()
        process.kill()
        process.wait()
    if not checked:
        raise IsolationError("Claude Code exited before reporting its configuration")
    if not finished and not timed_out.is_set():
        raise AgentError(f"Claude Code exited early with code {process.returncode}")
    run.stop = run.stop or "time budget"
    return run
