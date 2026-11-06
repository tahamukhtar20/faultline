"""Private scenarios. Results come from the provider's ledger, never from the agent's code.

Order IDs and amounts are random, so special-casing known fixtures cannot work. Failures are
categorized so a benchmark can report how agents fail, not only that they failed.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass

from faultline.models import ScenarioResult
from faultline.sandbox.docker import ControlChannelError, DockerError
from faultline.stack import ServiceStack

HIDDEN = "hidden_invariants"
RESILIENCE = "resilience"
PERFORMANCE = "performance"

_CRITICAL_CATEGORIES = frozenset(
    {"duplicate_charge", "phantom_charge", "wrong_amount", "double_fulfilment"}
)
_THROUGHPUT_ORDERS = 20
_THROUGHPUT_BUDGET_SECONDS = 10.0
_API_LATENCY_BUDGET_SECONDS = 0.5


class ScenarioFailure(Exception):
    def __init__(self, category: str, detail: str):
        super().__init__(detail)
        self.category = category


def _new_order() -> tuple[str, int]:
    return f"order-{secrets.token_hex(8)}", 100 + secrets.randbelow(9_900)


def _post_order(stack: ServiceStack, order_id: str, amount: int, currency: str = "USD") -> None:
    reply = stack.request(
        "POST", "/orders", {"order_id": order_id, "amount_cents": amount, "currency": currency}
    )
    if reply.get("status") != 201:
        raise ScenarioFailure("api_error", f"POST /orders returned {reply}")


def _start(stack: ServiceStack, workers: int = 1) -> None:
    stack.start_api()
    if not stack.wait_ready():
        raise ScenarioFailure("startup_failure", "the API did not become healthy")
    for _ in range(workers):
        stack.start_worker()


def _require_status(stack: ServiceStack, order_id: str, status: str, timeout: float) -> None:
    if not stack.wait_for_status(order_id, status, timeout):
        raise ScenarioFailure(
            "order_not_completed",
            f"order is {stack.order_status(order_id)!r}, expected {status!r} within {timeout:g}s",
        )


def _require_settled(stack: ServiceStack, order_id: str, amount: int) -> None:
    """Check the provider's own records: one capture of the right amount and one parcel."""
    captures = stack.captures(order_id)
    if len(captures) > 1:
        raise ScenarioFailure("duplicate_charge", f"{len(captures)} captures for one order")
    if not captures:
        raise ScenarioFailure("not_charged", "the provider has no capture for this order")
    if captures[0]["amount_cents"] != amount:
        raise ScenarioFailure(
            "wrong_amount", f"charged {captures[0]['amount_cents']}, expected {amount}"
        )
    shipments = stack.shipments(order_id)
    if shipments > 1:
        raise ScenarioFailure("double_fulfilment", f"{shipments} shipments for one order")
    if not shipments:
        raise ScenarioFailure("not_shipped", "the provider has no shipment for this order")


def _wait_for_capture(stack: ServiceStack, order_id: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stack.captures(order_id):
            return
        time.sleep(0.02)
    raise ScenarioFailure("not_charged", "no payment was attempted")


def crash_after_charge(stack: ServiceStack) -> None:
    """A worker is killed after the payment and before acknowledging the event."""
    _start(stack)
    order_id, amount = _new_order()
    stack.provider.call("hold")
    _post_order(stack, order_id, amount)
    _wait_for_capture(stack, order_id, 8)
    stack.kill_worker(stack.workers[0])
    stack.provider.call("release")
    stack.start_worker()
    _require_status(stack, order_id, "fulfilled", 15)
    _require_settled(stack, order_id, amount)


def transient_payment_failure(stack: ServiceStack) -> None:
    """The provider fails twice and then recovers, so the order must still go through once."""
    _start(stack)
    order_id, amount = _new_order()
    stack.provider.call("fail_charges", order_id=order_id, times=2)
    _post_order(stack, order_id, amount)
    _require_status(stack, order_id, "fulfilled", 10)
    _require_settled(stack, order_id, amount)


def crash_on_commit(stack: ServiceStack) -> None:
    """The API dies right after committing an order. It must be all there or not there at all."""
    stack.start_api({"ORDERFLOW_FAILPOINT": "after_order_commit"})
    if not stack.wait_ready():
        raise ScenarioFailure("startup_failure", "the API did not become healthy")
    stack.start_worker()
    order_id, amount = _new_order()
    reply = stack.request(
        "POST", "/orders", {"order_id": order_id, "amount_cents": amount, "currency": "USD"}
    )
    assert stack.api is not None
    if "connection_error" not in reply or not stack.api.wait_exit(5):
        raise ScenarioFailure("crash_suppressed", "the failpoint no longer crashes the API")

    stack.start_api()
    if not stack.wait_ready():
        raise ScenarioFailure("startup_failure", "the API did not restart")
    if stack.order_status(order_id) is None:
        time.sleep(1.5)
        if stack.captures(order_id):
            raise ScenarioFailure("phantom_charge", "a customer was charged for a missing order")
        return
    if not stack.wait_for_status(order_id, "fulfilled", 8):
        raise ScenarioFailure("lost_event", "the order exists but nothing will ever process it")
    _require_settled(stack, order_id, amount)


def poison_queue(stack: ServiceStack) -> None:
    """A permanently rejected order must fail without holding up the orders behind it."""
    _start(stack)
    poison, poison_amount = _new_order()
    _post_order(stack, poison, poison_amount, currency="XXX")
    healthy = [_new_order() for _ in range(3)]
    for order_id, amount in healthy:
        _post_order(stack, order_id, amount)
    for order_id, amount in healthy:
        if not stack.wait_for_status(order_id, "fulfilled", 6):
            raise ScenarioFailure("blocked_queue", "healthy orders are stuck behind a rejected one")
        _require_settled(stack, order_id, amount)
    if not stack.wait_for_status(poison, "failed", 6):
        raise ScenarioFailure("poison_not_failed", "the rejected order was never marked failed")
    if stack.shipments(poison):
        raise ScenarioFailure("double_fulfilment", "a rejected order was shipped")


def rolling_upgrade(stack: ServiceStack) -> None:
    """Events from the previous release and the current one are processed side by side."""
    _start(stack)
    legacy, legacy_amount = _new_order()
    current, current_amount = _new_order()
    assert stack.api is not None
    produced = stack.api.text(
        ["python", "-m", "orderflow.legacy_producer", legacy, str(legacy_amount)], timeout=10
    )
    if produced.returncode != 0:
        raise ScenarioFailure("legacy_event_failed", produced.stderr[-300:])
    _post_order(stack, current, current_amount)
    _require_status(stack, current, "fulfilled", 10)
    _require_settled(stack, current, current_amount)
    if not stack.wait_for_status(legacy, "fulfilled", 10):
        status = stack.order_status(legacy)
        if stack.captures(legacy):
            _require_settled(stack, legacy, legacy_amount)
        raise ScenarioFailure("legacy_event_failed", f"previous-release order is {status!r}")
    _require_settled(stack, legacy, legacy_amount)


def lease_race(stack: ServiceStack) -> None:
    """With a slow provider and two workers, each order is still processed exactly once."""
    _start(stack, workers=2)
    stack.provider.call("latency", seconds=1.2)
    orders = [_new_order() for _ in range(2)]
    for order_id, amount in orders:
        _post_order(stack, order_id, amount)
    for order_id, _ in orders:
        _require_status(stack, order_id, "fulfilled", 12)
    time.sleep(2)
    for order_id, amount in orders:
        _require_settled(stack, order_id, amount)


def throughput(stack: ServiceStack) -> None:
    """A burst of orders finishes within budget, and the API stays responsive."""
    _start(stack)
    orders = [_new_order() for _ in range(_THROUGHPUT_ORDERS)]
    started = time.monotonic()
    latencies: list[float] = []
    for order_id, amount in orders:
        posted = time.monotonic()
        _post_order(stack, order_id, amount)
        latencies.append(time.monotonic() - posted)
    duplicate = stack.request(
        "POST", "/orders", {"order_id": orders[0][0], "amount_cents": orders[0][1]}
    )
    if duplicate.get("status") != 409:
        raise ScenarioFailure("api_error", f"a repeated order returned {duplicate.get('status')}")
    for order_id, _ in orders:
        remaining = _THROUGHPUT_BUDGET_SECONDS - (time.monotonic() - started)
        if not stack.wait_for_status(order_id, "fulfilled", max(remaining, 0.1)):
            raise ScenarioFailure("too_slow", "orders did not finish within the time budget")
    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95) - 1]
    if p95 > _API_LATENCY_BUDGET_SECONDS:
        raise ScenarioFailure("too_slow", f"p95 API latency {p95:.3f}s")
    for order_id, amount in orders:
        _require_settled(stack, order_id, amount)


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    component: str
    run: Callable[[ServiceStack], None]


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("crash_after_charge", HIDDEN, crash_after_charge),
    Scenario("crash_on_commit", HIDDEN, crash_on_commit),
    Scenario("rolling_upgrade", HIDDEN, rolling_upgrade),
    Scenario("transient_payment_failure", RESILIENCE, transient_payment_failure),
    Scenario("poison_queue", RESILIENCE, poison_queue),
    Scenario("lease_race", RESILIENCE, lease_race),
    Scenario("throughput", PERFORMANCE, throughput),
)
SCENARIOS_BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}


def run_scenario(scenario: Scenario, stack: ServiceStack) -> ScenarioResult:
    try:
        scenario.run(stack)
    except ScenarioFailure as failure:
        return ScenarioResult(
            name=scenario.name,
            component=scenario.component,
            passed=False,
            critical=failure.category in _CRITICAL_CATEGORIES,
            category=failure.category,
            detail=str(failure),
        )
    except (ControlChannelError, DockerError, KeyError, TypeError, AttributeError) as error:
        return ScenarioResult(
            name=scenario.name,
            component=scenario.component,
            passed=False,
            category="infrastructure",
            detail=repr(error),
        )
    return ScenarioResult(name=scenario.name, component=scenario.component, passed=True)
