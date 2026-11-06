from orderflow import legacy_producer, worker


def test_events_from_the_previous_release_are_processed(client, provider) -> None:
    legacy_producer.write_order("order-6", 1250)

    assert worker.run_once("test-worker")

    assert list(provider.charges.values()) == [("order-6", 1250, "USD")]
    assert client.get("/orders/order-6").json()["status"] == "fulfilled"
