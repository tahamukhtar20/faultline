"""Clients for the external payment and fulfilment provider."""

import httpx

from orderflow import config


class PermanentFailure(Exception):
    """The provider rejected the request and retrying cannot help."""


def _post(path: str, body: dict[str, object]) -> None:
    response = httpx.post(
        f"{config.PROVIDER_URL}{path}", json=body, timeout=config.PROVIDER_TIMEOUT_SECONDS
    )
    if response.status_code == 422:
        raise PermanentFailure(response.text)
    response.raise_for_status()


def charge(idempotency_key: str, order_id: str, amount_cents: int, currency: str) -> None:
    _post(
        "/charges",
        {
            "key": idempotency_key,
            "order_id": order_id,
            "amount_cents": amount_cents,
            "currency": currency,
        },
    )


def ship(order_id: str) -> None:
    """Shipments have no idempotency key: every successful call sends a parcel."""
    _post("/shipments", {"order_id": order_id})
