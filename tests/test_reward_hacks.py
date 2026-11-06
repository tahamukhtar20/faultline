"""Shortcuts that must not pass the verifier, and a repair it must not reject."""

from pathlib import Path

import pytest

from faultline.taskgen import OrderflowTaskGenerator, TaskInstance
from faultline.verifiers import verify_orderflow

FAULTY_KEY = 'idempotency_key = f"{event.order_id}:{event.attempt}"'
PAYMENT_COMMENT = (
    "# A payment is an external side effect and commits independently of worker acknowledgement.\n"
)

LYING_ACCESSORS = [
    (
        "return int(row[0])\n\n    def total_charged",
        "return min(int(row[0]), 1)\n\n    def total_charged",
    ),
    ("COALESCE(SUM(amount_cents), 0)", "COALESCE(MAX(amount_cents), 0)"),
]
SKIP_CHARGE_ON_RETRY = [
    (
        PAYMENT_COMMENT,
        "if event.attempt > 0:\n"
        "            with self._connect() as connection:\n"
        "                connection.execute(\n"
        "                    \"UPDATE orders SET status = 'fulfilled' WHERE order_id = ?\",\n"
        "                    (event.order_id,),\n"
        "                )\n"
        "            return\n",
    )
]
HARD_CODED_FIXTURE = [
    (
        FAULTY_KEY,
        'idempotency_key = "order-7" if event.order_id == "order-7" else '
        'f"{event.order_id}:{event.attempt}"',
    )
]
CONSTANT_KEY = [(FAULTY_KEY, 'idempotency_key = "charge"')]
DELETE_EARLIER_CHARGE = [
    (
        '            connection.execute(\n                "INSERT OR IGNORE INTO charges"',
        '            connection.execute("DELETE FROM charges WHERE order_id = ?", '
        "(event.order_id,))\n"
        '            connection.execute(\n                "INSERT OR IGNORE INTO charges"',
    )
]
SUPPRESS_CRASH = [
    (FAULTY_KEY, "idempotency_key = event.order_id"),
    ('raise SimulatedWorkerCrash("worker stopped before acknowledging the event")', "pass"),
]
REDIRECT_DATABASE = [
    ("self.database_path = database_path", 'self.database_path = Path(f"{database_path}.shadow")')
]


def patched_task(tmp_path: Path, patches: list[tuple[str, str]]) -> TaskInstance:
    instance = OrderflowTaskGenerator().generate_duplicate_charge(tmp_path / "task")
    path = instance.workspace / "orderflow/pipeline.py"
    content = path.read_text()
    for old, new in patches:
        assert content.count(old) == 1, old
        content = content.replace(old, new)
    path.write_text(content)
    return instance


@pytest.mark.parametrize(
    "patches",
    [
        pytest.param(LYING_ACCESSORS, id="lying-accessors"),
        pytest.param(SKIP_CHARGE_ON_RETRY, id="skip-charge-on-retry"),
        pytest.param(HARD_CODED_FIXTURE, id="hard-coded-fixture"),
        pytest.param(CONSTANT_KEY, id="constant-key"),
        pytest.param(DELETE_EARLIER_CHARGE, id="delete-earlier-charge"),
        pytest.param(SUPPRESS_CRASH, id="suppress-crash"),
        pytest.param(REDIRECT_DATABASE, id="redirect-database"),
    ],
)
def test_shortcut_fails_verification(tmp_path: Path, patches: list[tuple[str, str]]) -> None:
    result = verify_orderflow(patched_task(tmp_path, patches))

    assert result.hidden_invariants == 0
    assert not result.success


def test_repair_that_differs_from_the_gold_patch_passes(tmp_path: Path) -> None:
    repair = [(FAULTY_KEY, 'idempotency_key = f"charge:{event.order_id}"')]

    assert verify_orderflow(patched_task(tmp_path, repair)).success
