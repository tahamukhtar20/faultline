"""The action and observation an OpenEnv client exchanges with Faultline."""

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class FaultlineAction(Action):
    kind: str = Field(
        ...,
        description="read, search, run, logs, query, patch, restart, checkpoint, restore or submit",
    )
    argument: str = Field(default="", description="The action's argument, such as a file path")


class FaultlineObservation(Observation):
    issue: str = ""
    last_result: str = ""
    visible_test_summary: str = ""
    service_summary: str = ""
    diff_summary: str = ""
    remaining_steps: int = 0
