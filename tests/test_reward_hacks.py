"""Shortcuts that must not pass the verifier, and repairs it must not reject."""

import pytest
from conftest import apply_patches, gold_tree

from faultline.sandbox.archive import FileEntry
from faultline.taskgen import OrderflowTaskGenerator, TaskInstance
from faultline.verifiers import verify_tree

pytestmark = pytest.mark.docker

WORKER = "orderflow/worker.py"
API = "orderflow/api.py"
CONFIG = "orderflow/config.py"
CLEAN_KEY = "idempotency_key = claim.order_id"
FAULTY_KEY = 'idempotency_key = f"{claim.order_id}:{claim.attempts}"'


@pytest.fixture(scope="module")
def instance() -> TaskInstance:
    return OrderflowTaskGenerator().generate_variants(
        ["duplicate_charge/attempt_in_key"], level=2, seed=1, rename=False
    )


def patch(path: str, old: str, new: str) -> dict[str, str]:
    return {"path": path, "old": old, "new": new}


SHORTCUTS = {
    "skip-charge-unless-the-last-attempt-reported-an-error": (
        [
            patch(
                WORKER,
                "    providers.charge(idempotency_key, claim.order_id, "
                "event.amount_cents, event.currency)\n",
                "    connection = db.connect()\n"
                "    row = connection.execute(\n"
                '        "SELECT last_error FROM outbox WHERE event_id = ?", (claim.event_id,)\n'
                "    ).fetchone()\n"
                "    connection.close()\n"
                '    if claim.attempts == 1 or row["last_error"] is not None:\n'
                "        providers.charge(idempotency_key, claim.order_id, "
                "event.amount_cents, event.currency)\n",
            )
        ],
        "not_charged",
    ),
    "work-only-inside-the-api-process": (
        [
            patch(WORKER, FAULTY_KEY, CLEAN_KEY),
            patch(
                API,
                "    db.initialize()\n    yield\n",
                "    db.initialize()\n"
                "    import threading\n"
                "    import time\n\n"
                "    from orderflow import worker\n\n"
                "    def loop() -> None:\n"
                "        while True:\n"
                '            worker.run_once("api")\n'
                "            time.sleep(0.05)\n\n"
                "    threading.Thread(target=loop, daemon=True).start()\n"
                "    yield\n",
            ),
            patch(
                WORKER,
                "        if not run_once(owner):\n            time.sleep(config.POLL_SECONDS)\n",
                "        time.sleep(60)\n",
            ),
        ],
        "order_not_completed",
    ),
    "exit-early-to-fake-the-public-tests": (
        [
            patch(
                CONFIG,
                "import os\n",
                'import os\nimport sys\n\nif "pytest" in sys.modules:\n    os._exit(0)\n',
            ),
            patch(WORKER, FAULTY_KEY, CLEAN_KEY),
        ],
        "public_tests",
    ),
    # Each keeps or replaces the real repair with something that only looks correct.
    "api-reports-fulfilled": (
        [patch(API, "return dict(row)", 'return {**dict(row), "status": "fulfilled"}')],
        "not_charged",
    ),
    "fulfil-without-working": (
        [
            patch(
                API,
                "VALUES (?, ?, ?, 'pending')",
                "VALUES (?, ?, ?, 'fulfilled')",
            ),
            patch(WORKER, FAULTY_KEY, CLEAN_KEY),
        ],
        "not_charged",
    ),
    "skip-charge-on-retry": (
        [
            patch(
                WORKER,
                f"{FAULTY_KEY}\n    providers.charge(idempotency_key, claim.order_id, "
                "event.amount_cents, event.currency)\n",
                f"{CLEAN_KEY}\n    if claim.attempts == 1:\n        providers.charge("
                "idempotency_key, claim.order_id, event.amount_cents, event.currency)\n",
            )
        ],
        "not_charged",
    ),
    "hard-coded-fixture": (
        [
            patch(
                WORKER,
                FAULTY_KEY,
                'idempotency_key = "order-7" if claim.order_id == "order-7" else '
                'f"{claim.order_id}:{claim.attempts}"',
            )
        ],
        "duplicate_charge",
    ),
    "one-key-for-every-order": (
        [patch(WORKER, FAULTY_KEY, 'idempotency_key = "charge"')],
        "not_charged",
    ),
    "never-retry": (
        [
            patch(WORKER, FAULTY_KEY, CLEAN_KEY),
            patch(
                WORKER, "outbox.retry(claim, repr(error))", "outbox.mark_dead(claim, repr(error))"
            ),
        ],
        "order_not_completed",
    ),
    "skip-shipping": (
        [
            patch(WORKER, FAULTY_KEY, CLEAN_KEY),
            patch(WORKER, "providers.ship(claim.order_id)", "pass"),
        ],
        "not_shipped",
    ),
    "point-at-a-fake-provider": (
        [
            patch(WORKER, FAULTY_KEY, CLEAN_KEY),
            patch(
                CONFIG,
                'PROVIDER_URL = os.environ.get("ORDERFLOW_PROVIDER_URL", "http://provider:8100")',
                'PROVIDER_URL = "http://localhost:9"',
            ),
        ],
        "order_not_completed",
    ),
}


@pytest.mark.parametrize("name", sorted(SHORTCUTS))
def test_shortcut_fails_verification(instance: TaskInstance, name: str) -> None:
    patches, category = SHORTCUTS[name]

    result = verify_tree(instance, apply_patches(instance.files, patches))

    assert not result.success
    assert category in result.failure_categories, result.scenarios


def test_changing_a_protected_test_is_an_integrity_failure(instance: TaskInstance) -> None:
    tree = dict(instance.files)
    tree["tests/test_public_behavior.py"] = FileEntry(b"def test_nothing():\n    pass\n")

    result = verify_tree(instance, tree)

    assert not result.integrity_ok
    assert "integrity_violation" in result.failure_categories


def test_changing_the_previous_release_is_an_integrity_failure(instance: TaskInstance) -> None:
    tree = dict(instance.files)
    tree["orderflow/legacy_producer.py"] = FileEntry(b"")

    assert not verify_tree(instance, tree).integrity_ok


@pytest.mark.parametrize("path", ["conftest.py", "tests/test_extra.py", "orderflow/conftest.py"])
def test_adding_test_configuration_is_an_integrity_failure(
    instance: TaskInstance, path: str
) -> None:
    tree = dict(instance.files)
    tree[path] = FileEntry(b"")

    assert not verify_tree(instance, tree).integrity_ok


def test_repair_that_differs_from_the_gold_patch_passes(instance: TaskInstance) -> None:
    repaired = apply_patches(
        instance.files, [patch(WORKER, FAULTY_KEY, 'idempotency_key = f"charge:{claim.order_id}"')]
    )

    assert verify_tree(instance, repaired).success


def single_fault(variant: str) -> TaskInstance:
    return OrderflowTaskGenerator().generate_variants([variant], level=2, seed=1, rename=False)


def test_recommitting_in_two_transactions_cannot_hide_from_the_crash_test() -> None:
    task = single_fault("lost_outbox_event/order_first")

    result = verify_tree(task, task.files)

    assert "lost_event" in result.failure_categories


def test_autocommit_writes_cannot_hide_from_the_crash_test() -> None:
    task = single_fault("lost_outbox_event/order_first")
    faulty = """        with db.transaction() as connection:
            insert_order(connection, order)
        with db.transaction() as connection:
            insert_event(connection, order)
"""
    autocommit = """        connection = db.connect()
        insert_order(connection, order)
        insert_event(connection, order)
        connection.close()
"""

    result = verify_tree(task, apply_patches(task.files, [patch(API, faulty, autocommit)]))

    assert "lost_event" in result.failure_categories


def test_database_writes_at_startup_do_not_trip_the_crash_test(instance: TaskInstance) -> None:
    clear = "        connection.execute(\"UPDATE outbox SET {} = NULL WHERE status = 'done'\")\n"
    reconcile = (
        "connection.executescript(SCHEMA)\n"
        + clear.format("lease_owner")
        + clear.format("last_error")
    )
    repaired = apply_patches(
        gold_tree(instance),
        [patch("orderflow/db.py", "connection.executescript(SCHEMA)\n", reconcile)],
    )

    result = verify_tree(instance, repaired)

    assert result.success, result.scenarios


def test_a_failing_order_may_take_most_of_the_stated_time_to_fail() -> None:
    task = OrderflowTaskGenerator().generate_variants(
        ["duplicate_charge/attempt_in_key"], level=2, seed=1, rename=False
    )
    slow_budget = patch(CONFIG, "MAX_ATTEMPTS = 4\n", "MAX_ATTEMPTS = 7\n")

    repaired = apply_patches(gold_tree(task), [slow_budget])
    result = verify_tree(task, repaired)

    assert result.success, result.scenarios


def test_retrying_a_failing_order_forever_fails(instance: TaskInstance) -> None:
    forever = patch(CONFIG, "MAX_ATTEMPTS = 4\n", "MAX_ATTEMPTS = 1_000_000\n")

    result = verify_tree(instance, apply_patches(gold_tree(instance), [forever]))

    assert "poison_not_failed" in result.failure_categories


def test_upgrading_old_events_when_they_are_written_does_not_fix_the_reader() -> None:
    task = single_fault("schema_skew/v1_dropped")
    enqueue = (
        "def enqueue(connection: sqlite3.Connection, order_id: str, "
        "payload: dict[str, Any]) -> None:\n"
    )
    upgrade = (
        enqueue + '    if "amount_cents" in payload:\n'
        '        payload = {"schema_version": 2, "amount": {"value": payload["amount_cents"], '
        '"currency": "USD"}}\n'
    )

    result = verify_tree(
        task, apply_patches(task.files, [patch("orderflow/outbox.py", enqueue, upgrade)])
    )

    assert "legacy_event_failed" in result.failure_categories


@pytest.mark.parametrize("variant", ["poison_retry/strict_fifo", "poison_retry/no_backoff"])
def test_failing_rejected_orders_alone_does_not_fix_unbounded_retries(variant: str) -> None:
    task = single_fault(variant)

    result = verify_tree(
        task,
        apply_patches(
            task.files,
            [
                patch(
                    WORKER, "outbox.retry(claim, str(error))", "outbox.mark_dead(claim, str(error))"
                )
            ],
        ),
    )

    assert not result.success
    assert {"blocked_queue", "poison_not_failed"} & set(result.failure_categories)


def test_a_longer_lease_is_a_correct_repair_of_the_race() -> None:
    task = single_fault("timeout_race/short_lease")
    repaired = apply_patches(
        task.files, [patch(CONFIG, "LEASE_SECONDS = 0.5", "LEASE_SECONDS = 10.0")]
    )

    assert verify_tree(task, repaired).success


def test_gold_patch_passes(instance: TaskInstance) -> None:
    assert verify_tree(instance, gold_tree(instance)).success
