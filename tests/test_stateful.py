"""Checkpoint, restore and branch on real episodes."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from conftest import action, patch_action

from faultline.checkpoints import CheckpointError
from faultline.env import FaultlineEnv
from faultline.models import ActionKind
from faultline.taskgen import OrderflowTaskGenerator, TaskInstance

pytestmark = pytest.mark.docker


def task(variant: str = "duplicate_charge/attempt_in_key", seed: int = 5) -> TaskInstance:
    return OrderflowTaskGenerator().generate_variants([variant], level=2, seed=seed, rename=False)


@pytest.fixture
def env() -> Iterator[FaultlineEnv]:
    environment = FaultlineEnv()
    environment.reset(options={"instance": task()})
    yield environment
    environment.close()


def snapshot_view(environment: FaultlineEnv) -> dict[str, object]:
    observation = environment._last_observation_required()
    query = environment.runtime.execute(
        action_model(ActionKind.QUERY, "SELECT order_id, status FROM orders ORDER BY order_id")
    )
    return {
        "files": {
            path: entry.data for path, entry in environment.runtime.export_workspace().items()
        },
        "orders": query,
        "ledger": environment.runtime.stack.ledger(),
        "diff": observation.diff_summary,
        "steps": observation.remaining_steps,
        "verification": environment.verification.model_dump(),
    }


def action_model(kind: ActionKind, argument: str = ""):
    from faultline.models import ToolAction

    return ToolAction(kind=kind, argument=argument)


def test_snapshot_restore_is_state_equivalent(env: FaultlineEnv) -> None:
    env.step(action(ActionKind.READ, "orderflow/config.py"))
    checkpoint = env.checkpoint()
    before = snapshot_view(env)

    env.step(patch_action(env.instance.gold_patches[0]))
    env.step(action(ActionKind.RESTART))
    assert snapshot_view(env) != before

    env.restore(checkpoint)
    assert snapshot_view(env) == before


def test_restoring_twice_is_idempotent(env: FaultlineEnv) -> None:
    checkpoint = env.checkpoint()
    env.step(patch_action(env.instance.gold_patches[0]))

    env.restore(checkpoint)
    first = snapshot_view(env)
    env.restore(checkpoint)

    assert snapshot_view(env) == first


def test_two_branches_diverge_independently(env: FaultlineEnv) -> None:
    checkpoint = env.checkpoint()
    gold = env.instance.gold_patches[0]
    wrong = {"path": gold["path"], "old": gold["old"], "new": 'idempotency_key = "same-for-all"'}
    parent = snapshot_view(env)
    left = env.branch(checkpoint)
    right = env.branch(checkpoint)
    try:
        assert snapshot_view(left)["files"] == parent["files"]
        assert snapshot_view(left)["verification"] == parent["verification"]
        assert snapshot_view(left)["ledger"] == parent["ledger"]

        left.step(patch_action(gold))
        right.step(patch_action(wrong))

        assert left.verification.success
        assert not right.verification.success
        assert snapshot_view(left)["files"] != snapshot_view(right)["files"]
        assert snapshot_view(env)["files"] == parent["files"]
    finally:
        left.close()
        right.close()


def test_restore_rejects_different_task(env: FaultlineEnv) -> None:
    checkpoint = env.checkpoint()
    other = FaultlineEnv(replay_incidents=False, checkpoint_dir=env.checkpoints.root)
    try:
        other.reset(options={"instance": task("schema_skew/v1_dropped", seed=6)})

        with pytest.raises(CheckpointError, match="belongs to task"):
            other.restore(checkpoint)

        observation, *_ = other.step(action(ActionKind.RESTORE, checkpoint))
        assert "tool error" in observation["last_result"]
    finally:
        other.close()


def test_agent_restore_does_not_refund_the_budget(env: FaultlineEnv) -> None:
    observation, *_ = env.step(action(ActionKind.CHECKPOINT))
    checkpoint = observation["last_result"].split()[-1]
    remaining = int(observation["remaining_steps"])

    env.step(action(ActionKind.READ, "orderflow/config.py"))
    observation, *_ = env.step(action(ActionKind.RESTORE, checkpoint))

    assert int(observation["remaining_steps"]) == remaining - 2


def test_reward_cannot_be_replayed_after_restore(env: FaultlineEnv) -> None:
    gold = env.instance.gold_patches[0]
    observation, *_ = env.step(action(ActionKind.CHECKPOINT))
    checkpoint = observation["last_result"].split()[-1]

    _, repair, *_ = env.step(patch_action(gold))
    assert repair > 0

    total = 0.0
    for _ in range(2):
        _, undo, *_ = env.step(action(ActionKind.RESTORE, checkpoint))
        _, redo, *_ = env.step(patch_action(gold))
        total += undo + redo

    assert total <= 0


def test_experimenter_restore_also_rewinds_reward_state(env: FaultlineEnv) -> None:
    gold = env.instance.gold_patches[0]
    checkpoint = env.checkpoint()
    _, first, *_ = env.step(patch_action(gold))

    env.restore(checkpoint)
    _, second, *_ = env.step(patch_action(gold))

    assert second == pytest.approx(first)


def test_restored_budget_comes_from_the_checkpoint(env: FaultlineEnv) -> None:
    env.step(action(ActionKind.READ, "orderflow/config.py"))
    checkpoint = env.checkpoint()
    env.step(action(ActionKind.READ, "orderflow/config.py"))

    observation = env.restore(checkpoint)

    assert int(observation["remaining_steps"]) == env.instance.manifest.step_budget - 1


def test_gold_run_can_rewind_and_reproduce_the_original_failure(env: FaultlineEnv) -> None:
    original = env.verification
    broken = env.checkpoint()

    env.step(patch_action(env.instance.gold_patches[0]))
    _, _, terminated, _, info = env.step(action(ActionKind.SUBMIT))
    assert terminated
    assert info["success"]

    env.restore(broken)
    env.step(action(ActionKind.READ, "orderflow/worker.py"))
    _, _, terminated, _, info = env.step(action(ActionKind.SUBMIT))

    assert terminated
    assert not info["success"]
    assert env.verification.failure_categories == original.failure_categories
    assert "duplicate_charge" in info["failure_categories"]
