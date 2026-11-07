"""FaultlineEnv behind OpenEnv's reset, step and state.

Each episode starts Docker containers from this process, so the server runs beside a Docker engine.
It does not run on Hugging Face Spaces, which offer no Docker engine to a Space. To grade without
one, use the Harbor, Inspect or Verifiers formats.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from faultline.env import ACTION_KINDS, FaultlineEnv
from faultline.models import ActionKind
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import FaultlineAction, FaultlineObservation
except ImportError:
    from models import FaultlineAction, FaultlineObservation


class FaultlineEnvironment(Environment[FaultlineAction, FaultlineObservation, State]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._env = FaultlineEnv()
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        level: int = 1,
        **kwargs: Any,
    ) -> FaultlineObservation:
        observation, _ = self._env.reset(seed=seed, options={"level": level})
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        return self._observation(observation, reward=0.0, done=False)

    def step(
        self, action: FaultlineAction, timeout_s: float | None = None, **kwargs: Any
    ) -> FaultlineObservation:
        kind = ActionKind(action.kind)
        observation, reward, terminated, truncated, _ = self._env.step(
            {"kind": ACTION_KINDS.index(kind), "argument": action.argument}
        )
        self._state.step_count += 1
        return self._observation(observation, reward=float(reward), done=terminated or truncated)

    @property
    def state(self) -> State:
        return self._state

    def close(self) -> None:
        self._env.close()

    @staticmethod
    def _observation(
        observation: dict[str, Any], *, reward: float, done: bool
    ) -> FaultlineObservation:
        return FaultlineObservation(
            issue=observation["issue"],
            last_result=observation["last_result"],
            visible_test_summary=observation["visible_test_summary"],
            service_summary=observation["service_summary"],
            diff_summary=observation["diff_summary"],
            remaining_steps=int(observation["remaining_steps"]),
            reward=reward,
            done=done,
        )
