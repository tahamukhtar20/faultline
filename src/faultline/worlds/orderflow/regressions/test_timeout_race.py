import time

from orderflow import outbox


def test_claimed_event_stays_claimed_while_the_provider_responds(client, provider) -> None:
    client.post("/orders", json={"order_id": "order-8", "amount_cents": 800})

    assert outbox.claim_next("worker-a") is not None
    time.sleep(1.5)

    assert outbox.claim_next("worker-b") is None
