"""Typed contracts shared by generators, environments, runtimes, and verifiers."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ActionKind(StrEnum):
    READ = "read"
    SEARCH = "search"
    RUN = "run"
    PATCH = "patch"
    SUBMIT = "submit"


class RewardMode(StrEnum):
    SPARSE = "sparse"
    SHAPED = "shaped"


class ToolAction(FrozenModel):
    kind: ActionKind
    argument: str = Field(default="", max_length=16_384)


class Observation(FrozenModel):
    issue: str
    last_result: str = ""
    remaining_steps: int = Field(ge=0)


class TaskManifest(FrozenModel):
    task_id: str = Field(min_length=1)
    seed: int
    issue: str = Field(min_length=1)
    fault_ids: tuple[str, ...] = Field(min_length=1)
    step_budget: int = Field(default=80, ge=1)
    protected_paths: tuple[str, ...] = ()


class VerificationResult(FrozenModel):
    public_behavior: float = Field(ge=0, le=1)
    hidden_invariants: float = Field(ge=0, le=1)
    performance: float = Field(ge=0, le=1)
    integrity_ok: bool
    diagnostics: tuple[str, ...] = ()

    @property
    def success(self) -> bool:
        return (
            self.integrity_ok
            and self.public_behavior == 1
            and self.hidden_invariants == 1
            and self.performance == 1
        )
