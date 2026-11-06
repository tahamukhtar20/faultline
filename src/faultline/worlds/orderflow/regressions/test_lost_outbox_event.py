import pytest
from orderflow import api, db


@pytest.mark.parametrize("failing_step", ["insert_order", "insert_event"])
def test_order_and_event_commit_together(client, monkeypatch, failing_step) -> None:
    def fail(*args) -> None:
        raise RuntimeError("database write failed")

    monkeypatch.setattr(api, failing_step, fail)
    with pytest.raises(RuntimeError):
        client.post("/orders", json={"order_id": "order-5", "amount_cents": 500})

    connection = db.connect()
    orders = connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    events = connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
    connection.close()
    assert orders == events
