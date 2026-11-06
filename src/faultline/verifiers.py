"""Scores a task workspace with scenarios kept outside it.

Agent-patched code only drives the pipeline. Results are read from the database by this process,
so the pipeline's own reporting methods are never trusted. Order IDs are random, so special-casing
known fixtures cannot work, and each delivery runs in its own process, as a real worker would.
"""

from __future__ import annotations

import contextlib
import secrets
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from faultline.models import VerificationResult
from faultline.process import run_bounded
from faultline.taskgen import TaskInstance, sha256_file

PUBLIC_TIMEOUT_SECONDS = 20.0
WORKER_TIMEOUT_SECONDS = 10.0
PERFORMANCE_LIMIT_SECONDS = 5.0

_WORKER = r"""
import sys
from pathlib import Path
from orderflow.pipeline import OrderEvent, OrderPipeline, SimulatedWorkerCrash

command, database, order_id, value = sys.argv[1:]
pipeline = OrderPipeline(Path(database))
if command == "create":
    pipeline.create_order(order_id, int(value))
elif command == "deliver":
    pipeline.process(OrderEvent(order_id, attempt=int(value)))
else:
    try:
        pipeline.process(OrderEvent(order_id, attempt=int(value)), crash_after_charge=True)
    except SimulatedWorkerCrash:
        sys.exit(0)
    sys.exit("worker did not crash")
"""

# Counts every payment attempt, so deleting an earlier charge cannot hide a repeated one.
_INSTALL_AUDIT = """
CREATE TABLE verifier_audit (order_id TEXT NOT NULL);
CREATE TRIGGER verifier_count_charges AFTER INSERT ON charges BEGIN
    INSERT INTO verifier_audit VALUES (NEW.order_id);
END;
"""


class _ScenarioFailure(Exception):
    pass


def _public_behavior(workspace: Path) -> tuple[float, str]:
    try:
        completed = run_bounded(
            [sys.executable, "-m", "pytest", "-q", "tests/test_public_behavior.py"],
            cwd=workspace,
            timeout=PUBLIC_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return 0.0, f"timed out after {PUBLIC_TIMEOUT_SECONDS:g}s"
    summary = (completed.stdout + completed.stderr)[-2_000:]
    return (1.0 if completed.returncode == 0 else 0.0, summary)


def _run_worker(workspace: Path, database: Path, command: str, order_id: str, value: int) -> None:
    try:
        completed = run_bounded(
            [sys.executable, "-c", _WORKER, command, str(database), order_id, str(value)],
            cwd=workspace,
            timeout=WORKER_TIMEOUT_SECONDS,
            env={"PYTHONPATH": str(workspace)},
        )
    except subprocess.TimeoutExpired:
        raise _ScenarioFailure(f"{command} timed out") from None
    if completed.returncode != 0:
        raise _ScenarioFailure(f"{command} failed: {completed.stderr[-500:]}")


def _check_order(database: Path, order_id: str, *, step: str, amount: int, fulfilled: bool) -> None:
    with contextlib.closing(sqlite3.connect(database)) as connection:
        attempts = connection.execute(
            "SELECT COUNT(*) FROM verifier_audit WHERE order_id = ?", (order_id,)
        ).fetchone()[0]
        rows, total = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(amount_cents), 0) FROM charges WHERE order_id = ?",
            (order_id,),
        ).fetchone()
        status = connection.execute(
            "SELECT status FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
    if status is None:
        raise _ScenarioFailure(f"{step}: order is missing")
    if (attempts, rows, total) != (1, 1, amount):
        raise _ScenarioFailure(
            f"{step}: {attempts} payment attempts, {rows} charges, "
            f"charged {total} of {amount} cents"
        )
    if (status[0] == "fulfilled") != fulfilled:
        raise _ScenarioFailure(f"{step}: unexpected order status {status[0]!r}")


def _retry_scenarios(workspace: Path) -> tuple[bool, str, float]:
    retried = f"order-{secrets.token_hex(8)}"
    late = f"order-{secrets.token_hex(8)}"
    retried_amount = 100 + secrets.randbelow(9_900)
    late_amount = 100 + secrets.randbelow(9_900)

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="faultline-verify-") as directory:
        database = Path(directory) / "orders.db"
        try:
            _run_worker(workspace, database, "create", retried, retried_amount)
            with contextlib.closing(sqlite3.connect(database)) as connection:
                connection.executescript(_INSTALL_AUDIT)

            # A worker charges, crashes before acknowledging, and the event is redelivered.
            _run_worker(workspace, database, "crash", retried, 0)
            _check_order(
                database, retried, step="after crash", amount=retried_amount, fulfilled=False
            )
            _run_worker(workspace, database, "deliver", retried, 1)
            _check_order(
                database, retried, step="after retry", amount=retried_amount, fulfilled=True
            )

            # Earlier attempts died before charging, so a later attempt must still charge, and
            # this order must not be affected by the one above.
            _run_worker(workspace, database, "create", late, late_amount)
            _run_worker(workspace, database, "deliver", late, 3)
            _check_order(
                database, late, step="late first delivery", amount=late_amount, fulfilled=True
            )
            _run_worker(workspace, database, "deliver", late, 4)
            _check_order(database, late, step="late redelivery", amount=late_amount, fulfilled=True)
        except (_ScenarioFailure, sqlite3.Error) as failure:
            return False, str(failure), time.perf_counter() - started
    return True, "all scenarios passed", time.perf_counter() - started


def verify_orderflow(instance: TaskInstance) -> VerificationResult:
    integrity_ok = all(
        (instance.workspace / path).is_file()
        and sha256_file(instance.workspace / path) == expected_hash
        for path, expected_hash in instance.protected_hashes.items()
    )
    public_score, public_diagnostic = _public_behavior(instance.workspace)
    scenarios_ok, scenario_diagnostic, duration = _retry_scenarios(instance.workspace)

    return VerificationResult(
        public_behavior=public_score,
        hidden_invariants=1.0 if scenarios_ok else 0.0,
        performance=1.0 if duration < PERFORMANCE_LIMIT_SECONDS else 0.0,
        integrity_ok=integrity_ok,
        diagnostics=(
            f"public: {public_diagnostic}",
            f"private scenarios: {scenario_diagnostic}",
            f"private scenario duration: {duration:.3f}s",
        ),
    )
