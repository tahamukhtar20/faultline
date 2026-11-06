from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from conftest import action, patch_action
from pettingzoo.test import api_test

from faultline.adversarial import _FAMILY_NAMES, ARCHITECT, SOLVER, FaultlineAdversarialEnv
from faultline.models import ActionKind

pytestmark = pytest.mark.docker


def design(**families: int) -> dict[str, object]:
    return {
        "faults": np.array([families.get(name, 0) for name in _FAMILY_NAMES], dtype=np.int64),
        "level": 1,
    }


@pytest.fixture
def env() -> Iterator[FaultlineAdversarialEnv]:
    environment = FaultlineAdversarialEnv()
    environment.reset(seed=3)
    yield environment
    environment.close()


def finish(env: FaultlineAdversarialEnv) -> None:
    for agent in env.agent_iter():
        _, _, terminated, truncated, _ = env.last()
        if terminated or truncated:
            env.step(None)
        else:
            assert agent == SOLVER
            env.step(action(ActionKind.SUBMIT))


def test_environment_passes_the_pettingzoo_api_test() -> None:
    environment = FaultlineAdversarialEnv()
    try:
        api_test(environment, num_cycles=3, verbose_progress=False)
    finally:
        environment.close()


def test_architect_is_rewarded_when_a_valid_task_beats_the_solver(
    env: FaultlineAdversarialEnv,
) -> None:
    env.step(design(duplicate_charge=1))
    assert env.task_valid
    assert env.agent_selection == SOLVER

    env.step(action(ActionKind.SUBMIT))

    assert 0 < env._cumulative_rewards[ARCHITECT] < 1
    assert env._cumulative_rewards[SOLVER] < 0


def test_architect_gets_nothing_when_the_solver_repairs_the_task(
    env: FaultlineAdversarialEnv,
) -> None:
    env.step(design(duplicate_charge=1))
    assert env.task is not None
    for patch in env.task.gold_patches:
        env.step(patch_action(patch))

    env.step(action(ActionKind.SUBMIT))

    assert env._cumulative_rewards[ARCHITECT] < 0
    assert env.infos[SOLVER]["success"]


@pytest.mark.parametrize(
    ("faults", "reason"),
    [
        ({}, "1 to 3 faults"),
        (
            {
                "duplicate_charge": 1,
                "lost_outbox_event": 1,
                "schema_skew": 1,
                "poison_retry": 2,
            },
            "1 to 3 faults",
        ),
        ({"poison_retry": 1, "timeout_race": 2}, "edit the same code"),
    ],
)
def test_invalid_tasks_lose_immediately(
    env: FaultlineAdversarialEnv, faults: dict[str, int], reason: str
) -> None:
    env.step(design(**faults))

    assert env._cumulative_rewards[ARCHITECT] == -1
    assert reason in env.infos[ARCHITECT]["invalid_reason"]
    assert all(env.terminations.values())


def test_harder_tasks_cost_the_architect_more() -> None:
    rewards = []
    for faults in ({"duplicate_charge": 1}, {"duplicate_charge": 1, "schema_skew": 1}):
        environment = FaultlineAdversarialEnv()
        try:
            environment.reset(seed=3)
            environment.step(design(**faults))
            environment.step(action(ActionKind.SUBMIT))
            rewards.append(environment._cumulative_rewards[ARCHITECT])
        finally:
            environment.close()

    assert rewards[0] > rewards[1]
