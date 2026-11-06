import time

from orderflow import worker


def test_rejected_order_does_not_block_later_orders(client, provider) -> None:
    client.post("/orders", json={"order_id": "bad", "amount_cents": 100, "currency": "XXX"})
    client.post("/orders", json={"order_id": "good", "amount_cents": 200})

    deadline = time.monotonic() + 2
    while (
        time.monotonic() < deadline and client.get("/orders/good").json()["status"] != "fulfilled"
    ):
        worker.run_once("test-worker")

    assert client.get("/orders/good").json()["status"] == "fulfilled"
    assert client.get("/orders/bad").json()["status"] == "failed"
