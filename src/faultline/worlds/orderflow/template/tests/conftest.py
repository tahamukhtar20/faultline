from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from orderflow import config, db, providers
from orderflow.api import app


class FakeProvider:
    def __init__(self) -> None:
        self.charges: dict[str, tuple[str, int, str]] = {}
        self.shipments: list[str] = []

    def charge(self, idempotency_key: str, order_id: str, amount_cents: int, currency: str) -> None:
        if currency == "XXX":
            raise providers.PermanentFailure("unsupported currency")
        self.charges.setdefault(idempotency_key, (order_id, amount_cents, currency))

    def ship(self, order_id: str) -> None:
        self.shipments.append(order_id)


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> FakeProvider:
    fake = FakeProvider()
    monkeypatch.setattr(providers, "charge", fake.charge)
    monkeypatch.setattr(providers, "ship", fake.ship)
    return fake


@pytest.fixture
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(config, "DATABASE_PATH", tmp_path / "orders.db")
    db.initialize()
    with TestClient(app) as test_client:
        yield test_client
