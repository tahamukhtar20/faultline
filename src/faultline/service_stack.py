"""What a scenario needs from a running copy of the service, however that copy is hosted."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Protocol


class CommandResult(Protocol):
    @property
    def returncode(self) -> int: ...

    @property
    def stderr(self) -> str: ...


class Service(Protocol):
    def is_running(self) -> bool: ...

    def wait_exit(self, timeout: float) -> bool: ...

    def text(self, command: list[str], *, timeout: float) -> CommandResult: ...


class ProviderChannel(Protocol):
    def call(self, operation: str, *, timeout: float = 15, **arguments: Any) -> dict[str, Any]: ...


class BaseStack(ABC):
    provider: ProviderChannel
    api: Service | None

    @abstractmethod
    def start_api(self, *, crash_test: bool = False) -> Service: ...

    @abstractmethod
    def start_worker(self) -> Service: ...

    @abstractmethod
    def stop_api(self) -> None: ...

    @abstractmethod
    def stop_services(self) -> None: ...

    @abstractmethod
    def _api_url(self, path: str) -> str: ...

    def request(
        self, method: str, path: str, body: Any = None, timeout: float = 5
    ) -> dict[str, Any]:
        return self.provider.call(
            "request",
            url=self._api_url(path),
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
