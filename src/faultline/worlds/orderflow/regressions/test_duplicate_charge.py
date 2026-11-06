import pytest
from orderflow import config, providers, worker


def test_redelivery_after_a_crash_reuses_the_payment_key(client, provider, monkeypatch) -> None:
    monkeypatch.setattr(config, "LEASE_SECONDS", 0.0)
    client.post("/orders", json={"order_id": "order-9", "amount_cents": 700})

    class Crash(BaseException):
        pass

    def crash_on_ship(order_id: str) -> None:
        raise Crash

    with monkeypatch.context() as patched:
        patched.setattr(providers, "ship", crash_on_ship)
        with pytest.raises(Crash):
            worker.run_once("worker-a")

    assert worker.run_once("worker-b")

    assert len(provider.charges) == 1
