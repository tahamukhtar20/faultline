from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

from .models import FaultlineAction, FaultlineObservation


class FaultlineClient(EnvClient[FaultlineAction, FaultlineObservation, State]):
    def _step_payload(self, action: FaultlineAction) -> dict[str, Any]:
        return {"kind": action.kind, "argument": action.argument}

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[FaultlineObservation]:
        observation = FaultlineObservation(
            **payload.get("observation", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
