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
    LOGS = "logs"
    QUERY = "query"
    PATCH = "patch"
    RESTART = "restart"
    CHECKPOINT = "checkpoint"
    RESTORE = "restore"
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
    visible_test_summary: str = "not run"
    service_summary: str = "unknown"
    diff_summary: str = "no changes"
    remaining_steps: int = Field(ge=0)


class Difficulty(FrozenModel):
    level: int = Field(ge=1, le=5)
    fault_count: int = Field(ge=1)
    interaction_depth: int = Field(ge=0)
    observation_noise: int = Field(ge=0, le=3)
    verifier_gap: int = Field(ge=0, le=3)
    delayed_effects: int = Field(ge=0)
    budget_pressure: int = Field(ge=0, le=3)

    @property
    def score(self) -> int:
        return (
            self.fault_count
            + self.interaction_depth
            + self.observation_noise
            + self.verifier_gap
            + self.delayed_effects
            + self.budget_pressure
        )


class TaskManifest(FrozenModel):
    task_id: str = Field(min_length=1)
    seed: int
    issue: str = Field(min_length=1)
    fault_ids: tuple[str, ...] = Field(min_length=1)
    variant_ids: tuple[str, ...] = Field(min_length=1)
    difficulty: Difficulty
    step_budget: int = Field(ge=1)
    time_budget_seconds: int = Field(ge=1)
    visible_incidents: tuple[str, ...] = ()


class ScenarioResult(FrozenModel):
    name: str
    component: str
    passed: bool
    critical: bool = False
    category: str = ""
    detail: str = ""


class VerificationResult(FrozenModel):
    public_behavior: float = Field(ge=0, le=1)
    hidden_invariants: float = Field(ge=0, le=1)
    resilience: float = Field(ge=0, le=1)
    performance: float = Field(ge=0, le=1)
    integrity_ok: bool
    critical_invariants_ok: bool
    scenarios: tuple[ScenarioResult, ...] = ()
    diagnostics: tuple[str, ...] = ()

    @property
    def success(self) -> bool:
        return (
            self.integrity_ok
            and self.critical_invariants_ok
            and self.public_behavior == 1
            and self.hidden_invariants == 1
            and self.resilience == 1
            and self.performance == 1
        )

    @property
    def failure_categories(self) -> tuple[str, ...]:
        categories = {scenario.category for scenario in self.scenarios if not scenario.passed}
        if not self.integrity_ok:
            categories.add("integrity_violation")
        if self.public_behavior < 1:
            categories.add("public_tests")
        return tuple(sorted(categories))
