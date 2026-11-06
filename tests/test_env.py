from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from conftest import action, patch_action
from gymnasium.utils.env_checker import check_env

from faultline.env import FaultlineEnv
from faultline.models import ActionKind
from faultline.taskgen import OrderflowTaskGenerator, TaskInstance
from faultline.trajectories import load_trajectory, replay_trajectory

pytestmark = pytest.mark.docker


@pytest.fixture
def env() -> Iterator[FaultlineEnv]:
    environment = FaultlineEnv(replay_incidents=False)
    yield environment
    environment.close()


def single_fault(variant: str = "duplicate_charge/attempt_in_key") -> TaskInstance:
    return OrderflowTaskGenerator().generate_variants([variant], level=2, seed=5, rename=False)


def test_environment_passes_gymnasium_checker(env: FaultlineEnv) -> None:
    check_env(env, skip_render_check=True)


def test_gold_trajectory_repairs_and_submits(env: FaultlineEnv) -> None:
    observation, _ = env.reset(seed=23, options={"level": 3})
    assert observation["issue"]
    total = 0.0
    for patch in env.instance.gold_patches:
        _, reward, terminated, truncated, _ = env.step(patch_action(patch))
        assert not terminated
        assert not truncated
        total += reward
    assert total > 0

    _, reward, terminated, truncated, info = env.step(action(ActionKind.SUBMIT))

    assert reward > 0.3
    assert terminated
    assert not truncated
    assert info["success"]
    assert info["failure_categories"] == []


def test_submitting_without_repairing_fails(env: FaultlineEnv) -> None:
    env.reset(seed=1, options={"level": 2})
    _, reward, terminated, _, info = env.step(action(ActionKind.SUBMIT))

    assert terminated
    assert reward < 0
    assert not info["success"]
    assert info["failure_categories"]


def test_test_tampering_terminates_episode(env: FaultlineEnv) -> None:
    env.reset(seed=4)
    patch = {
        "path": "tests/test_public_behavior.py",
        "old": "assert created.status_code == 201",
        "new": "assert True",
    }

    _, reward, terminated, _, info = env.step(patch_action(patch))

    assert terminated
    assert reward < -0.9
    assert not info["final_verification"]["integrity_ok"]


def test_stepping_after_the_episode_ends_is_an_error(env: FaultlineEnv) -> None:
    env.reset(seed=1)
    env.step(action(ActionKind.SUBMIT))

    with pytest.raises(RuntimeError, match="episode has ended"):
        env.step(action(ActionKind.READ, "orderflow/config.py"))


def test_step_budget_truncates(env: FaultlineEnv) -> None:
    env.reset(seed=1)
    truncated = False
    for _ in range(env.instance.manifest.step_budget):
        _, _, _, truncated, _ = env.step(action(ActionKind.SEARCH, "zzz-no-such-text"))
    assert truncated


def test_observation_fields_report_real_state(env: FaultlineEnv) -> None:
    observation, _ = env.reset(options={"instance": single_fault()})
    assert observation["diff_summary"] == "no changes"
    assert observation["visible_test_summary"] == "not run"
    assert "api running" in observation["service_summary"]
    started = env.remaining_seconds

    observation, *_ = env.step(action(ActionKind.RUN, "pytest -q"))
    assert "passed" in observation["visible_test_summary"]
    assert env.remaining_seconds < started

    observation, *_ = env.step(patch_action(env.instance.gold_patches[0]))
    assert "orderflow/worker.py +1 -1" in observation["diff_summary"]


def test_non_ascii_output_stays_inside_the_observation_space(env: FaultlineEnv) -> None:
    env.reset(options={"instance": single_fault()})
    patch = {
        "path": "orderflow/config.py",
        "old": '"""Service settings."""',
        "new": '"""café ☕"""',
    }
    env.step(patch_action(patch))

    observation, *_ = env.step(action(ActionKind.READ, "orderflow/config.py"))

    assert env.observation_space.contains(observation)


def test_long_output_is_truncated_inside_the_observation_space(env: FaultlineEnv) -> None:
    env.reset(options={"instance": single_fault()})
    observation, *_ = env.step(action(ActionKind.SEARCH, "e"))

    assert env.observation_space.contains(observation)


def test_tools_reject_bad_arguments(env: FaultlineEnv) -> None:
    env.reset(options={"instance": single_fault()})
    for kind, argument in [
        (ActionKind.READ, "../../etc/passwd"),
        (ActionKind.READ, "missing.py"),
        (ActionKind.RUN, "rm -rf /"),
        (ActionKind.RUN, "pytest -q; rm -rf /"),
        (ActionKind.PATCH, "not json"),
        (ActionKind.QUERY, "DELETE FROM orders"),
        (ActionKind.LOGS, "../../etc"),
        (ActionKind.RESTORE, "not-a-checkpoint"),
    ]:
        observation, *_ = env.step(action(kind, argument))
        assert "tool error" in observation["last_result"], (kind, argument)


def test_logs_and_query_show_the_live_service() -> None:
    instance = single_fault()
    env = FaultlineEnv()
    try:
        env.reset(options={"instance": instance})
        logs, *_ = env.step(action(ActionKind.LOGS, "provider"))
        assert logs["last_result"].count("provider charge") >= 2

        rows, *_ = env.step(action(ActionKind.QUERY, "SELECT order_id, status FROM orders"))
        assert "fulfilled" in rows["last_result"]
    finally:
        env.close()


def test_restart_runs_the_patched_code(env: FaultlineEnv) -> None:
    env.reset(options={"instance": single_fault()})
    env.step(
        patch_action(
            {
                "path": "orderflow/worker.py",
                "old": 'logger.info("worker %s started", owner)',
                "new": 'logger.info("patched build %s started", owner)',
            }
        )
    )

    env.step(action(ActionKind.RESTART))
    observation, *_ = env.step(action(ActionKind.LOGS, "worker"))

    assert "patched build" in observation["last_result"]


def test_episode_survives_code_that_hangs(env: FaultlineEnv) -> None:
    env.reset(options={"instance": single_fault()})
    signature = "def handle(claim: outbox.Claim) -> None:\n"
    hang = "    while True:\n        pass\n"

    _, reward, terminated, truncated, _ = env.step(
        patch_action({"path": "orderflow/worker.py", "old": signature, "new": signature + hang})
    )
    assert reward < 0
    assert not terminated
    assert not truncated

    env.step(
        patch_action({"path": "orderflow/worker.py", "old": signature + hang, "new": signature})
    )
    for patch in env.instance.gold_patches:
        env.step(patch_action(patch))
    _, _, terminated, _, info = env.step(action(ActionKind.SUBMIT))
    assert terminated
    assert info["success"]


def test_code_that_crashes_on_import_scores_as_a_failure(env: FaultlineEnv) -> None:
    env.reset(options={"instance": single_fault()})
    patch = {"path": "orderflow/worker.py", "old": "import logging", "new": "import logging_typo"}

    _, reward, terminated, _, _ = env.step(patch_action(patch))

    assert reward < 0
    assert not terminated
    assert env.verification.hidden_invariants == 0


def test_planted_symlink_is_an_integrity_failure(env: FaultlineEnv) -> None:
    env.reset(options={"instance": single_fault()})
    env.runtime.solver.run(["ln", "-s", "/etc/passwd", "/workspace/orderflow/link"])

    _, reward, terminated, _, info = env.step(action(ActionKind.RUN, "pytest -q"))

    assert terminated
    assert reward < -0.9
    assert not info["final_verification"]["integrity_ok"]


def test_trajectory_records_every_step_and_replays(tmp_path: Path) -> None:
    env = FaultlineEnv(level=2, replay_incidents=False, trajectory_dir=tmp_path)
    try:
        env.reset(seed=11)
        env.step(action(ActionKind.READ, "orderflow/config.py"))
        for patch in env.instance.gold_patches:
            env.step(patch_action(patch))
        env.step(action(ActionKind.SUBMIT))
    finally:
        env.close()

    recorded = next(tmp_path.glob("*.jsonl"))
    header, steps = load_trajectory(recorded)
    assert header["manifest"]["seed"] == 11
    assert steps[-1]["terminated"]
    assert steps[-1]["verification"]["success"]
    assert all(json.dumps(step) for step in steps)

    assert replay_trajectory(recorded) == []
