"""Reward accounting with sparse and verifier-derived shaped modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from faultline.models import RewardMode, VerificationResult


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    progress: float
    terminal: float
    action_cost: float
    integrity_penalty: float

    @property
    def total(self) -> float:
        return self.progress + self.terminal + self.action_cost + self.integrity_penalty


class RewardEngine:
    """Pays only for changes in verifier state, never for what the agent claims.

    Shaped progress is the change in a potential, so any sequence of actions that returns to the
    same state, such as repairing, restoring and repairing again, nets to zero.
    """

    PUBLIC_WEIGHT = 0.10
    INVARIANT_WEIGHT = 0.25
    RESILIENCE_WEIGHT = 0.15
    PERFORMANCE_WEIGHT = 0.10
    TERMINAL_WEIGHT = 0.40

    def __init__(self, mode: RewardMode = RewardMode.SHAPED, action_cost: float = 0.001):
        if action_cost < 0:
            raise ValueError("action_cost must be non-negative")
        self.mode = mode
        self.action_cost = action_cost
        self._previous_potential = 0.0
        self._terminal_paid = False

    def reset(self) -> None:
        self._previous_potential = 0.0
        self._terminal_paid = False

    def prime(self, result: VerificationResult) -> None:
        """Set the episode baseline without rewarding pre-existing passing behavior."""
        self._previous_potential = self.potential(result)
        self._terminal_paid = False

    def state(self) -> dict[str, Any]:
        return {
            "previous_potential": self._previous_potential,
            "terminal_paid": self._terminal_paid,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._previous_potential = float(state["previous_potential"])
        self._terminal_paid = bool(state["terminal_paid"])

    @classmethod
    def potential(cls, result: VerificationResult) -> float:
        if not result.integrity_ok:
            return 0.0
        return (
            cls.PUBLIC_WEIGHT * result.public_behavior
            + cls.INVARIANT_WEIGHT * result.hidden_invariants
            + cls.RESILIENCE_WEIGHT * result.resilience
            + cls.PERFORMANCE_WEIGHT * result.performance
        )

    def step(self, result: VerificationResult, *, submitted: bool = False) -> RewardBreakdown:
        integrity_penalty = -1.0 if not result.integrity_ok else 0.0
        current_potential = self.potential(result)
        progress = 0.0
        if self.mode is RewardMode.SHAPED and result.integrity_ok:
            progress = current_potential - self._previous_potential
        self._previous_potential = current_potential

        terminal = 0.0
        if submitted and result.success and not self._terminal_paid:
            terminal = self.TERMINAL_WEIGHT
            self._terminal_paid = True

        return RewardBreakdown(
            progress=progress,
            terminal=terminal,
            action_cost=-self.action_cost,
            integrity_penalty=integrity_penalty,
        )
