"""Gymnasium interface for Faultline."""

from __future__ import annotations

import tempfile
from pathlib import Path
from string import printable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from faultline.models import ActionKind, Observation, RewardMode, ToolAction
from faultline.rewards import RewardEngine
from faultline.runtime import LocalWorkspaceRuntime
from faultline.taskgen import OrderflowTaskGenerator, TaskInstance
from faultline.verifiers import verify_orderflow

ACTION_KINDS = tuple(ActionKind)


class FaultlineEnv(gym.Env[dict[str, Any], dict[str, Any]]):
    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(self, *, reward_mode: RewardMode = RewardMode.SHAPED):
        super().__init__()
        self.generator = OrderflowTaskGenerator()
        self.reward_engine = RewardEngine(mode=reward_mode)
        self.action_space = spaces.Dict(
            {
                "kind": spaces.Discrete(len(ACTION_KINDS)),
                "argument": spaces.Text(min_length=0, max_length=16_384, charset=printable),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "issue": spaces.Text(max_length=4_096, charset=printable),
                "last_result": spaces.Text(max_length=8_000, charset=printable),
                "remaining_steps": spaces.Box(0, 10_000, shape=(), dtype=np.int64),
            }
        )
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._instance: TaskInstance | None = None
        self._runtime: LocalWorkspaceRuntime | None = None
        self._steps = 0
        self._last_observation: Observation | None = None

    @property
    def instance(self) -> TaskInstance:
        if self._instance is None:
            raise RuntimeError("environment must be reset before use")
        return self._instance

    @property
    def runtime(self) -> LocalWorkspaceRuntime:
        if self._runtime is None:
            raise RuntimeError("environment must be reset before use")
        return self._runtime

    def _observation(self, last_result: str) -> Observation:
        remaining_steps = max(0, self.instance.manifest.step_budget - self._steps)
        return Observation(
            issue=self.instance.manifest.issue,
            last_result=last_result,
            remaining_steps=remaining_steps,
        )

    @staticmethod
    def _as_space_value(observation: Observation) -> dict[str, Any]:
        payload = observation.model_dump()
        payload["remaining_steps"] = np.asarray(payload["remaining_steps"], dtype=np.int64)
        return payload

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self.close()
        task_seed = 0 if seed is None else seed
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="faultline-")
        destination = Path(self._temporary_directory.name) / "task"
        self._instance = self.generator.generate_duplicate_charge(destination, seed=task_seed)
        self._runtime = LocalWorkspaceRuntime(self._instance)
        self._steps = 0
        self.reward_engine.reset()
        initial_verification = verify_orderflow(self._instance)
        self.reward_engine.prime(initial_verification)
        self._last_observation = self._observation("episode initialized")
        return self._as_space_value(self._last_observation), {
            "task_id": self._instance.manifest.task_id
        }

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        kind_index = int(action["kind"])
        argument = str(action["argument"])
        if not 0 <= kind_index < len(ACTION_KINDS):
            raise ValueError(f"invalid action kind index: {kind_index}")
        tool_action = ToolAction(kind=ACTION_KINDS[kind_index], argument=argument)

        self._steps += 1
        tool_result = self.runtime.execute(tool_action)
        verification = verify_orderflow(self.instance)
        submitted = tool_action.kind is ActionKind.SUBMIT
        breakdown = self.reward_engine.step(verification, submitted=submitted)

        integrity_failure = not verification.integrity_ok
        terminated = submitted or integrity_failure
        truncated = self._steps >= self.instance.manifest.step_budget and not terminated
        self._last_observation = self._observation(tool_result)
        info: dict[str, Any] = {
            "reward_breakdown": {
                "progress": breakdown.progress,
                "terminal": breakdown.terminal,
                "action_cost": breakdown.action_cost,
                "integrity_penalty": breakdown.integrity_penalty,
            }
        }
        if terminated or truncated:
            info["final_verification"] = verification.model_dump()
        return (
            self._as_space_value(self._last_observation),
            breakdown.total,
            terminated,
            truncated,
            info,
        )

    def render(self) -> str:
        if self._last_observation is None:
            return "FaultlineEnv(uninitialized)"
        return (
            f"Issue: {self._last_observation.issue}\n"
            f"Last result: {self._last_observation.last_result}\n"
            f"Remaining steps: {self._last_observation.remaining_steps}"
        )

    def close(self) -> None:
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
        self._temporary_directory = None
        self._instance = None
        self._runtime = None
