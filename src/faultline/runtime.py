"""One episode's sandbox: an idle solver container plus a running copy of the service.

The solver container runs the agent's tests and has no network. Tool actions execute inside it
through a toolbox the workspace cannot shadow. The live service gives the agent real logs and
data to inspect, but scoring never uses it: verification happens elsewhere in fresh containers.
"""

from __future__ import annotations

import random
import re
from typing import Any

from faultline.models import ActionKind, ToolAction
from faultline.sandbox.archive import Tree, build_tar, read_tar
from faultline.sandbox.docker import Container, Sandbox, ensure_image
from faultline.scenarios import SCENARIOS_BY_NAME, run_scenario
from faultline.stack import ServiceStack
from faultline.taskgen import TaskInstance

MAX_OUTPUT = 8_000
COMMAND_TIMEOUT_SECONDS = 60
_TOOLBOX = ["python", "-I", "/opt/faultline/toolbox.py"]
_RUN_COMMAND = re.compile(r"(?:python -m )?pytest -q(?: (tests(?:/[\w.-]+)*(?:::\w+)?))?")
_LOG_TAIL_LINES = 60
_NOISE_LINES = (
    "orderflow.cache INFO refreshed config snapshot",
    "orderflow.metrics INFO flushed 128 samples",
    "uvicorn.access WARNING slow client disconnected",
    "orderflow.worker WARNING poll loop lagging by 12ms",
    "sqlite WARNING busy retry succeeded",
    "orderflow.cache WARNING stale entry evicted",
    "provider.client INFO connection pool resized",
    "orderflow.metrics ERROR exporter timeout, will retry",
)


class EpisodeRuntime:
    def __init__(self, instance: TaskInstance, *, replay_incidents: bool = True):
        self.instance = instance
        self.replay_incidents = replay_incidents
        self.noise_counter = 0
        self.last_test_summary = "not run"
        self._sandbox = Sandbox(ensure_image())
        self._workspace = self._sandbox.volume("workspace")
        self.stack = ServiceStack(self._sandbox, self._workspace, "live")
        self.solver: Container = self._sandbox.start(
            "solver",
            ["sleep", "infinity"],
            mounts={"/workspace": self._workspace, "/data": self.stack.data_volume},
            networked=False,
        )
        self.write_workspace(instance.files)
        self.start_services()
        if replay_incidents:
            self.run_incidents()

    def write_workspace(self, tree: Tree) -> None:
        self._replace_directory("/workspace", tree)

    def _replace_directory(self, directory: str, tree: Tree) -> None:
        self.solver.run(["find", directory, "-mindepth", "1", "-delete"], timeout=30)
        self.solver.run(
            ["tar", "--extract", "--no-same-owner", f"--directory={directory}"],
            stdin=build_tar(tree),
            timeout=30,
        )

    def _export_directory(self, directory: str) -> Tree:
        completed = self.solver.run(
            ["tar", "--create", f"--directory={directory}", "."], timeout=30
        )
        return read_tar(completed.stdout)

    def export_workspace(self) -> Tree:
        return self._export_directory("/workspace")

    def start_services(self) -> None:
        self.stack.start_api()
        self.stack.wait_ready()
        self.stack.start_worker()

    def run_incidents(self) -> None:
        """Send the traffic behind the reported symptoms through the live service."""
        for name in self.instance.manifest.visible_incidents:
            self.stack.stop_services()
            run_scenario(SCENARIOS_BY_NAME[name], self.stack)
        self.stack.provider.call("release")
        self.stack.provider.call("latency", seconds=0.0)
        self.stack.stop_services()
        self.start_services()

    def _toolbox(self, command: str, argument: str, timeout: float = 20) -> str:
        result = self.solver.text([*_TOOLBOX, command], stdin=argument.encode(), timeout=timeout)
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit code {result.returncode}"
            return f"tool error: {detail}"
        return result.stdout.rstrip("\n")

    def execute(self, action: ToolAction) -> str:
        kind = action.kind
        if kind is ActionKind.READ:
            return self._toolbox("read", action.argument)
        if kind is ActionKind.SEARCH:
            return self._toolbox("search", action.argument)
        if kind is ActionKind.PATCH:
            return self._toolbox("patch", action.argument)
        if kind is ActionKind.QUERY:
            return self._toolbox("query", action.argument)
        if kind is ActionKind.RUN:
            return self._run_tests(action.argument)
        if kind is ActionKind.LOGS:
            return self._logs(action.argument)
        if kind is ActionKind.RESTART:
            return self._restart()
        if kind is ActionKind.SUBMIT:
            return "solution submitted for private verification"
        raise ValueError(f"action is handled by the controller: {kind}")

    def _run_tests(self, argument: str) -> str:
        match = _RUN_COMMAND.fullmatch(argument.strip())
        if match is None:
            return (
                f"tool error: command is not allowlisted: {argument!r}. Use: pytest -q [tests/...]"
            )
        target = [match.group(1)] if match.group(1) else ["tests"]
        result = self.solver.text(
            ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", *target],
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
        if result.returncode in {124, 137}:
            self.last_test_summary = f"tests timed out after {COMMAND_TIMEOUT_SECONDS}s"
            return f"tool error: {self.last_test_summary}"
        output = result.stdout + result.stderr
        lines = output.strip().splitlines()
        self.last_test_summary = lines[-1][:500] if lines else "no output"
        return output

    def _logs(self, argument: str) -> str:
        source = argument.strip() or "all"
        if source not in {"all", "api", "worker", "provider"}:
            return "tool error: use logs with one of: all, api, worker, provider"
        sections = []
        if source in {"all", "provider"}:
            sections.append(self._provider_log())
        if source in {"all", "api"}:
            sections.append(self._service_log("api", [self.stack.api] if self.stack.api else []))
        if source in {"all", "worker"}:
            sections.append(self._service_log("worker", self.stack.workers))
        return "\n".join(sections)

    def _provider_log(self) -> str:
        ledger = self.stack.ledger()
        lines = [
            f"provider charge key={key} order={charge['order_id']} "
            f"amount={charge['amount_cents']} {charge['currency']}"
            for key, charge in ledger["charges"].items()
        ]
        lines.extend(f"provider shipment order={order}" for order in ledger["shipments"])
        return "\n".join(["--- provider ---", *lines[-_LOG_TAIL_LINES:]])

    def _service_log(self, role: str, containers: list[Container]) -> str:
        text = "\n".join([*self.stack.retired_logs[role], *[c.logs() for c in containers]])
        lines = [line for line in text.splitlines() if line.strip()][-_LOG_TAIL_LINES:]
        return "\n".join([f"--- {role} ---", *self._with_noise(lines)])

    def _with_noise(self, lines: list[str]) -> list[str]:
        """Mix in unrelated log lines according to the task's noise level."""
        noise = self.instance.manifest.difficulty.observation_noise
        if noise == 0:
            return lines
        self.noise_counter += 1
        rng = random.Random(f"{self.instance.manifest.seed}:{self.noise_counter}")
        mixed = list(lines)
        for _ in range(max(2, len(lines) // 4) * noise):
            mixed.insert(rng.randrange(len(mixed) + 1), rng.choice(_NOISE_LINES))
        return mixed

    def _restart(self) -> str:
        self.stack.stop_services()
        self.start_services()
        if self.replay_incidents:
            self.run_incidents()
        return f"services restarted with the current code\n{self.service_summary()}"

    def service_summary(self) -> str:
        api = self.stack.api
        api_state = "running" if api is not None and api.is_running() else "down"
        running = sum(worker.is_running() for worker in self.stack.workers)
        return f"api {api_state}, {running} of {len(self.stack.workers)} workers running"

    def capture_state(self) -> tuple[Tree, Tree, dict[str, Any]]:
        """Stop the service, then read its files, so that the database is at rest."""
        self.stack.stop_services()
        state = self.export_workspace(), self._export_directory("/data"), self.stack.ledger()
        self.start_services()
        return state

    def restore_state(self, workspace: Tree, data: Tree, ledger: dict[str, Any]) -> None:
        self.stack.stop_services()
        self.write_workspace(workspace)
        self._replace_directory("/data", data)
        self.stack.provider.call(
            "load",
            charges=ledger["charges"],
            shipments=ledger["shipments"],
            charge_calls=ledger["charge_calls"],
        )
        self.stack.retired_logs = {"api": [], "worker": []}
        self.start_services()

    def close(self) -> None:
        self.stack.close()
        self._sandbox.close()
