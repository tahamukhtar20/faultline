from orderflow.pipeline import OrderEvent, OrderPipeline


def test_order_is_charged_and_fulfilled(tmp_path) -> None:
    pipeline = OrderPipeline(tmp_path / "orders.db")
    pipeline.create_order("order-1", 2500)

    pipeline.process(OrderEvent("order-1"))

    assert pipeline.order_status("order-1") == "fulfilled"
    assert pipeline.charge_count("order-1") == 1
    assert pipeline.total_charged("order-1") == 2500


def test_same_delivery_is_idempotent(tmp_path) -> None:
    pipeline = OrderPipeline(tmp_path / "orders.db")
    pipeline.create_order("order-2", 3100)
    event = OrderEvent("order-2", attempt=0)

    pipeline.process(event)
    pipeline.process(event)

    assert pipeline.charge_count("order-2") == 1
