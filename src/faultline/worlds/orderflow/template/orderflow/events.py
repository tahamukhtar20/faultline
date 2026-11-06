"""Order event payloads. Old producers keep writing v1 during a rolling upgrade."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OrderCreated:
    amount_cents: int
    currency: str


def parse_event(payload: dict[str, Any]) -> OrderCreated:
    if payload.get("schema_version", 1) == 1:
        return OrderCreated(amount_cents=payload["amount_cents"], currency="USD")
    amount = payload["amount"]
    return OrderCreated(amount_cents=amount["value"], currency=amount["currency"])


def encode_event(amount_cents: int, currency: str) -> dict[str, Any]:
    return {"schema_version": 2, "amount": {"value": amount_cents, "currency": currency}}
