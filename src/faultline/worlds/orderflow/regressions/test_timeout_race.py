import threading
import time

from orderflow import providers, worker


def test_two_workers_do_not_process_one_order_with_a_slow_provider(
    client, provider, monkeypatch
) -> None:
    def slow_charge(*args) -> None:
        time.sleep(1.2)

    monkeypatch.setattr(providers, "charge", slow_charge)
    client.post("/orders", json={"order_id": "order-8", "amount_cents": 800})

    def work(owner: str) -> None:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            worker.run_once(owner)
            time.sleep(0.02)

    threads = [threading.Thread(target=work, args=(name,)) for name in ("worker-a", "worker-b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert provider.shipments == ["order-8"]
