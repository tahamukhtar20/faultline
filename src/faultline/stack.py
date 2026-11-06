"""A running copy of the order service: API, workers and the provider sidecar."""

from __future__ import annotations

import json
import time
from typing import Any

from faultline.sandbox.docker import Container, Provider, Sandbox

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
_WORKER_COMMAND = ["python", "-m", "orderflow.worker"]


class ServiceStack:
    def __init__(self, sandbox: Sandbox, workspace_volume: str, prefix: str):
        self._sandbox = sandbox
        self._prefix = prefix
        self._workspace = workspace_volume
        self._serial = 0
        self.data_volume = sandbox.volume(f"{prefix}-data")
        self.provider = Provider(sandbox, f"{prefix}-provider")
        self.api: Container | None = None
        self.workers: list[Container] = []
        self.retired_logs: dict[str, list[str]] = {"api": [], "worker": []}

    def _start(self, role: str, command: list[str], env: dict[str, str] | None) -> Container:
        self._serial += 1
        return self._sandbox.start(
            f"{self._prefix}-{role}-{self._serial}",
            command,
            mounts={"/workspace": f"{self._workspace}:ro", "/data": self.data_volume},
            env={"ORDERFLOW_PROVIDER_URL": f"http://{self.provider.host}:8100", **(env or {})},
        )

    def start_api(self, env: dict[str, str] | None = None) -> Container:
        self.stop_api()
        self.api = self._start("api", _API_COMMAND, env)
        return self.api

    def start_worker(self) -> Container:
        worker = self._start("worker", _WORKER_COMMAND, None)
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

    def request(
        self, method: str, path: str, body: Any = None, timeout: float = 5
    ) -> dict[str, Any]:
        assert self.api is not None
        return self.provider.call(
            "request",
            url=f"http://{self.api.name}:{_API_PORT}{path}",
            method=method,
            json=body,
            request_timeout=timeout,
            timeout=timeout + 5,
        )

    def wait_ready(self, timeout: float = 15) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.api is None or not self.api.is_running():
                return False
            if self.request("GET", "/health", timeout=1).get("status") == 200:
                return True
            time.sleep(0.05)
        return False

    def order_status(self, order_id: str) -> str | None:
        reply = self.request("GET", f"/orders/{order_id}")
        if reply.get("status") != 200:
            return None
        try:
            status = json.loads(reply["text"]).get("status")
        except (ValueError, AttributeError):
            return None
        return status if isinstance(status, str) else None

    def wait_for_status(self, order_id: str, wanted: str, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.order_status(order_id) == wanted:
                return True
            time.sleep(0.05)
        return False

    def ledger(self) -> dict[str, Any]:
        return self.provider.call("ledger")

    def captures(self, order_id: str) -> list[dict[str, Any]]:
        return [
            charge for charge in self.ledger()["charges"].values() if charge["order_id"] == order_id
        ]

    def shipments(self, order_id: str) -> int:
        return sum(1 for shipped in self.ledger()["shipments"] if shipped == order_id)

    def close(self) -> None:
        self.provider.close()
