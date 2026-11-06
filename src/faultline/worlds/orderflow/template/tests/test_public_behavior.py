from orderflow import worker


def drain(owner: str = "test-worker") -> None:
    while worker.run_once(owner):
        pass


def test_order_is_charged_and_shipped(client, provider) -> None:
    created = client.post("/orders", json={"order_id": "order-1", "amount_cents": 2500})
    assert created.status_code == 201

    drain()

    assert client.get("/orders/order-1").json()["status"] == "fulfilled"
    assert list(provider.charges.values()) == [("order-1", 2500, "USD")]
    assert provider.shipments == ["order-1"]


def test_duplicate_order_is_rejected(client, provider) -> None:
    body = {"order_id": "order-2", "amount_cents": 3100}

    assert client.post("/orders", json=body).status_code == 201
    assert client.post("/orders", json=body).status_code == 409


def test_unknown_order_is_not_found(client) -> None:
    assert client.get("/orders/missing").status_code == 404


def test_rejected_payment_fails_the_order(client, provider) -> None:
    client.post("/orders", json={"order_id": "order-3", "amount_cents": 900, "currency": "XXX"})

    assert worker.run_once("test-worker")

    assert client.get("/orders/order-3").json()["status"] in {"pending", "failed"}
    assert provider.shipments == []


def test_invalid_amount_is_rejected(client) -> None:
    response = client.post("/orders", json={"order_id": "order-4", "amount_cents": 0})

    assert response.status_code == 422
