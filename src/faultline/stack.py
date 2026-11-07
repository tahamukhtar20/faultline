"""A running copy of the order service in containers: API, workers and the provider sidecar."""

from __future__ import annotations

import secrets

from faultline.sandbox.docker import Container, Provider, Sandbox
from faultline.service_stack import BaseStack

_API_PORT = 8000
_API_COMMAND = [
    "python",
    "-m",
    "uvicorn",
    "orderflow.api:app",
    "--host",
    "0.0.0.0",
    "--port",
    str(_API_PORT),
    "--log-level",
    "warning",
]
_CRASH_TEST_API_COMMAND = ["python", "/opt/faultline/crash_api.py"]
_WORKER_COMMAND = ["python", "-m", "orderflow.worker"]


class ServiceStack(BaseStack):
    def __init__(self, sandbox: Sandbox, workspace_volume: str):
        self._sandbox = sandbox
        # Container names are visible to agent code, so they must not reveal what is being tested.
        prefix = secrets.token_hex(4)
        self._prefix = prefix
        self._workspace = workspace_volume
        self._serial = 0
        self.data_volume = sandbox.volume(f"{prefix}-data")
        self.provider: Provider = Provider(sandbox, f"{prefix}-provider")
        self.api: Container | None = None
        self.workers: list[Container] = []
        self.retired_logs: dict[str, list[str]] = {"api": [], "worker": []}

    def _start(self, role: str, command: list[str]) -> Container:
        self._serial += 1
        return self._sandbox.start(
            f"{self._prefix}-{role}-{self._serial}",
            command,
            mounts={"/workspace": f"{self._workspace}:ro", "/data": self.data_volume},
            env={"ORDERFLOW_PROVIDER_URL": f"http://{self.provider.host}:8100"},
        )

    def start_api(self, *, crash_test: bool = False) -> Container:
        self.stop_api()
        self.api = self._start("api", _CRASH_TEST_API_COMMAND if crash_test else _API_COMMAND)
        return self.api

    def start_worker(self) -> Container:
        worker = self._start("worker", _WORKER_COMMAND)
        self.workers.append(worker)
        return worker

    def _retire(self, role: str, container: Container) -> None:
        container.kill()
        self.retired_logs[role].append(container.logs())
        container.remove()

    def stop_api(self) -> None:
        if self.api is not None:
            self._retire("api", self.api)
            self.api = None

    def kill_worker(self, worker: Container) -> None:
        self.workers.remove(worker)
        self._retire("worker", worker)

    def stop_services(self) -> None:
        self.stop_api()
        for worker in list(self.workers):
            self.kill_worker(worker)

    def _api_url(self, path: str) -> str:
        assert self.api is not None
        return f"http://{self.api.name}:{_API_PORT}{path}"

    def close(self) -> None:
        self.provider.close()
