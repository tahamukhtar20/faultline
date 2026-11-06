import json

from gymnasium.utils.env_checker import check_env

from faultline.env import ACTION_KINDS, FaultlineEnv
from faultline.models import ActionKind


def action(kind: ActionKind, argument: str = "") -> dict[str, object]:
    return {"kind": ACTION_KINDS.index(kind), "argument": argument}


def test_environment_passes_gymnasium_checker() -> None:
    env = FaultlineEnv()
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_gold_trajectory_repairs_and_submits_task() -> None:
    env = FaultlineEnv()
    try:
        observation, info = env.reset(seed=23)
        assert "charged twice" in observation["issue"]
        assert info["task_id"] == "orderflow-duplicate-charge-23"

        patch = json.dumps(
            {
                "path": "orderflow/pipeline.py",
                "old": 'idempotency_key = f"{event.order_id}:{event.attempt}"',
                "new": "idempotency_key = event.order_id",
            }
        )
        _, patch_reward, terminated, truncated, _ = env.step(action(ActionKind.PATCH, patch))
        assert patch_reward > 0
        assert not terminated
        assert not truncated

        _, submit_reward, terminated, truncated, info = env.step(action(ActionKind.SUBMIT))
        assert submit_reward > 0
        assert terminated
        assert not truncated
        assert info["final_verification"]["hidden_invariants"] == 1
    finally:
        env.close()


def test_test_tampering_terminates_episode() -> None:
    env = FaultlineEnv()
    try:
        env.reset(seed=4)
        patch = json.dumps(
            {
                "path": "tests/test_public_behavior.py",
                "old": 'assert pipeline.charge_count("order-2") == 1',
                "new": "assert True",
            }
        )

        _, reward, terminated, _, info = env.step(action(ActionKind.PATCH, patch))

        assert terminated
        assert reward < 0
        assert not info["final_verification"]["integrity_ok"]
    finally:
        env.close()
