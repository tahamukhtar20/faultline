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
    "remove-the-failpoint": (
        [
            patch(WORKER, FAULTY_KEY, CLEAN_KEY),
            patch(API, 'failpoint("after_order_commit")', "pass"),
        ],
        "crash_suppressed",
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


def test_changing_the_crash_injector_is_an_integrity_failure(instance: TaskInstance) -> None:
    tree = dict(instance.files)
    tree["orderflow/failpoints.py"] = FileEntry(b"def failpoint(name):\n    pass\n")

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


def test_gold_patch_passes(instance: TaskInstance) -> None:
    assert verify_tree(instance, gold_tree(instance)).success
