"""PettingZoo mode: an architect designs the task, and a solver repairs it.

The architect wins when the solver fails, but only with a task that passes the validity gate: the
faults must really break the service, and the gold patch must really repair it. An impossible task
loses outright, and every added fault or difficulty level costs some of the prize, so the best
strategy is the simplest valid task the solver cannot do.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector

from faultline.env import FaultlineEnv
from faultline.faults import FAMILIES
from faultline.models import RewardMode
from faultline.taskgen import OrderflowTaskGenerator, TaskInstance, compatible
from faultline.verifiers import validate_instance

ARCHITECT = "architect"
SOLVER = "solver"
MAX_FAULTS = 3
_FAMILY_NAMES = tuple(sorted(FAMILIES))
_COMPLEXITY_COST = 0.05
_INVALID_TASK_REWARD = -1.0
_WIN_REWARD = 1.0


class FaultlineAdversarialEnv(AECEnv[str, dict[str, Any], dict[str, Any]]):  # type: ignore[misc]
    metadata = {
        "name": "faultline_adversarial_v0",
        "render_modes": ["ansi"],
        "is_parallelizable": False,
    }

    def __init__(
        self,
        *,
        reward_mode: RewardMode = RewardMode.SHAPED,
        replay_incidents: bool = False,
        checkpoint_dir: Path | None = None,
        trajectory_dir: Path | None = None,
    ):
        super().__init__()
        self.possible_agents = [ARCHITECT, SOLVER]
        self.generator = OrderflowTaskGenerator()
        self.solver_env = FaultlineEnv(
            reward_mode=reward_mode,
            replay_incidents=replay_incidents,
            checkpoint_dir=checkpoint_dir,
            trajectory_dir=trajectory_dir,
        )
        self._seed = 0
        self.task: TaskInstance | None = None
        self.invalid_reason = ""
        self.task_valid = False
        self._solver_observation: dict[str, Any] | None = None

    @cache  # noqa: B019 - PettingZoo requires the same space object every call
    def observation_space(self, agent: str) -> spaces.Space[Any]:
        if agent == ARCHITECT:
            return spaces.Dict({"catalogue": spaces.MultiDiscrete(self._catalogue_sizes())})
        return self.solver_env.observation_space

    @cache  # noqa: B019
    def action_space(self, agent: str) -> spaces.Space[Any]:
        if agent == ARCHITECT:
            return spaces.Dict(
                {
                    # For each family: 0 leaves it out, n picks its nth variant.
                    "faults": spaces.MultiDiscrete(self._catalogue_sizes()),
                    "level": spaces.Discrete(5),
                }
            )
        return self.solver_env.action_space

    @staticmethod
    def _catalogue_sizes() -> np.ndarray[Any, np.dtype[np.int64]]:
        return np.array(
            [len(FAMILIES[family].variants) + 1 for family in _FAMILY_NAMES], dtype=np.int64
        )

    def observe(self, agent: str) -> dict[str, Any]:
        if agent == ARCHITECT or self._solver_observation is None:
            if agent == ARCHITECT:
                return {"catalogue": self._catalogue_sizes() - 1}
            return self._blank_solver_observation()
        return self._solver_observation

    @staticmethod
    def _blank_solver_observation() -> dict[str, Any]:
        return {
            "issue": "",
            "last_result": "",
            "visible_test_summary": "",
            "service_summary": "",
            "diff_summary": "",
            "remaining_steps": np.asarray(0, dtype=np.int64),
            "remaining_seconds": np.asarray(0.0, dtype=np.float32),
        }

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        del options
        self._seed = 0 if seed is None else seed
        self.agents = list(self.possible_agents)
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos: dict[str, dict[str, Any]] = {agent: {} for agent in self.agents}
        self._selector = AgentSelector(self.agents)
        self.agent_selection = self._selector.next()
        self.task = None
        self.task_valid = False
        self.invalid_reason = ""
        self._solver_observation = None

    def step(self, action: dict[str, Any]) -> None:
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        self._cumulative_rewards[agent] = 0.0
        self.rewards = dict.fromkeys(self.agents, 0.0)
        if agent == ARCHITECT:
            self._architect_step(action)
        else:
            self._solver_step(action)
        self._accumulate_rewards()

    def _complexity(self, instance: TaskInstance) -> float:
        return _COMPLEXITY_COST * instance.manifest.difficulty.score

    def _architect_step(self, action: dict[str, Any]) -> None:
        instance = self._design_task(action)
        if instance is None:
            self.rewards[ARCHITECT] = _INVALID_TASK_REWARD
            self.infos[ARCHITECT] = {"invalid_reason": self.invalid_reason}
            self.terminations = dict.fromkeys(self.agents, True)
            self.agent_selection = self._selector.next()
            return
        self.task = instance
        self.task_valid = True
        self._solver_observation, _ = self.solver_env.reset(options={"instance": instance})
        self.infos[ARCHITECT] = {"task_id": instance.manifest.task_id}
        self.agent_selection = self._selector.next()

    def _design_task(self, action: dict[str, Any]) -> TaskInstance | None:
        picks = np.asarray(action["faults"], dtype=np.int64)
        level = int(action["level"]) + 1
        variants = [
            FAMILIES[family].variants[pick - 1]
            for family, pick in zip(_FAMILY_NAMES, picks.tolist(), strict=True)
            if pick > 0
        ]
        if not 1 <= len(variants) <= MAX_FAULTS:
            self.invalid_reason = f"a task needs 1 to {MAX_FAULTS} faults, got {len(variants)}"
            return None
        if not compatible(variants):
            self.invalid_reason = "the chosen faults edit the same code"
            return None
        instance = self.generator.generate_variants(
            [variant.variant_id for variant in variants], level=level, seed=self._seed
        )
        problem = validate_instance(instance)
        if problem:
            self.invalid_reason = problem
            return None
        return instance

    def _solver_step(self, action: dict[str, Any]) -> None:
        observation, reward, terminated, truncated, info = self.solver_env.step(action)
        self._solver_observation = observation
        self.rewards[SOLVER] = reward
        self.infos[SOLVER] = info
        if terminated or truncated:
            assert self.task is not None
            prize = 0.0 if info.get("success") else _WIN_REWARD
            self.rewards[ARCHITECT] = prize - self._complexity(self.task)
            self.terminations = dict.fromkeys(self.agents, terminated)
            self.truncations = dict.fromkeys(self.agents, truncated)
            self.terminations[ARCHITECT] = True
            self.truncations[ARCHITECT] = False

    def render(self) -> str:
        return self.solver_env.render() if self.task is not None else "architect is choosing a task"

    def close(self) -> None:
        self.solver_env.close()
