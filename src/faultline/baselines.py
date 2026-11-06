"""Reference policies: the gold repair, random actions, and fixed repair rules.

Every policy gets the same interface as a real agent, and only the gold policy reads the answer.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from faultline.env import ACTION_KINDS, FaultlineEnv
from faultline.models import ActionKind


@dataclass(slots=True)
class EpisodeResult:
    task_id: str
    level: int
    difficulty_score: int
    policy: str
    success: bool = False
    total_reward: float = 0.0
    steps: int = 0
    seconds: float = 0.0
    ended_by: str = "submit"
    components: dict[str, float] = field(default_factory=dict)
    failure_categories: list[str] = field(default_factory=list)


Step = Callable[[ActionKind, str], tuple[dict[str, Any], bool]]
Policy = Callable[[FaultlineEnv, dict[str, Any], Step], None]


def gold_policy(env: FaultlineEnv, _: dict[str, Any], step: Step) -> None:
    for patch in env.instance.gold_patches:
        step(ActionKind.PATCH, json.dumps(patch))
    step(ActionKind.SUBMIT, "")


def random_policy(env: FaultlineEnv, observation: dict[str, Any], step: Step) -> None:
    """Random but legal actions, including small random edits, then a submission."""
    rng = random.Random(env.instance.manifest.seed)
    observation, done = step(ActionKind.SEARCH, "")
    paths = [line for line in observation["last_result"].splitlines() if line.endswith(".py")]
    sources = [path for path in paths if not path.startswith("tests/")]
    for _ in range(12):
        if done:
            return
        kind = rng.choice([ActionKind.READ, ActionKind.SEARCH, ActionKind.LOGS, ActionKind.PATCH])
        if kind is ActionKind.READ:
            _, done = step(kind, rng.choice(paths))
        elif kind is ActionKind.SEARCH:
            _, done = step(kind, rng.choice(["order", "lease", "retry", "charge", "event"]))
        elif kind is ActionKind.LOGS:
            _, done = step(kind, rng.choice(["all", "api", "worker", "provider"]))
        else:
            path = rng.choice(sources)
            observation, done = step(ActionKind.READ, path)
            lines = [
                line.partition(": ")[2]
                for line in observation["last_result"].splitlines()
                if " = " in line
            ]
            if lines and not done:
                old = rng.choice(lines)
                edit = json.dumps({"path": path, "old": old, "new": f"{old}  # reviewed"})
                _, done = step(ActionKind.PATCH, edit)
    if not done:
        step(ActionKind.SUBMIT, "")


_RULES: tuple[tuple[str, str], ...] = (
    # Payment keys built from something that changes between deliveries.
    (r'= f"\{claim\.order_id\}:\{[^"]+\}"', "= claim.order_id"),
    # Lease shorter than the provider's timeout.
    (r"= 0\.5$", "= 3.0"),
    # Amounts converted twice.
    (r'\["amount_cents"\] \* 100', '["amount_cents"]'),
    # No delay between retries.
    (r"^(\s*)delay = 0\.0$", r"\1delay = config.RETRY_DELAY_SECONDS * 2**claim.attempts"),
)


def scripted_policy(env: FaultlineEnv, _: dict[str, Any], step: Step) -> None:
    """A checklist of known bad patterns, without any understanding of the code."""
    observation, done = step(ActionKind.SEARCH, "")
    paths = [
        path
        for path in observation["last_result"].splitlines()
        if path.endswith(".py") and not path.startswith("tests/")
    ]
    _, done = step(ActionKind.RUN, "pytest -q")
    for path in paths:
        if done:
            return
        observation, done = step(ActionKind.READ, path)
        if done:
            return
        for numbered in observation["last_result"].splitlines():
            line = numbered.partition(": ")[2]
            for pattern, replacement in _RULES:
                fixed = re.sub(pattern, replacement, line)
                if fixed != line and not done:
                    edit = json.dumps({"path": path, "old": line, "new": fixed})
                    _, done = step(ActionKind.PATCH, edit)
    if not done:
        step(ActionKind.SUBMIT, "")


POLICIES: dict[str, Policy] = {
    "gold": gold_policy,
    "random": random_policy,
    "scripted": scripted_policy,
}


def run_episode(
    env: FaultlineEnv, policy_name: str, policy: Policy, *, level: int, seed: int
) -> EpisodeResult:
    observation, info = env.reset(seed=seed, options={"level": level})
    result = EpisodeResult(
        task_id=info["task_id"],
        level=level,
        difficulty_score=env.instance.manifest.difficulty.score,
        policy=policy_name,
    )
    finished = {"done": False}

    def step(kind: ActionKind, argument: str) -> tuple[dict[str, Any], bool]:
        if finished["done"]:
            raise RuntimeError("the policy acted after the episode ended")
        next_observation, reward, terminated, truncated, step_info = env.step(
            {"kind": ACTION_KINDS.index(kind), "argument": argument}
        )
        result.total_reward += reward
        result.steps += 1
        if terminated or truncated:
            finished["done"] = True
            result.ended_by = "budget" if truncated else "submit"
            if not step_info["final_verification"]["integrity_ok"]:
                result.ended_by = "integrity_violation"
            result.success = bool(step_info["success"])
            result.failure_categories = list(step_info["failure_categories"])
            verification = step_info["final_verification"]
            result.components = {
                name: float(verification[name])
                for name in ("public_behavior", "hidden_invariants", "resilience", "performance")
            }
        return next_observation, finished["done"]

    policy(env, observation, step)
    if not finished["done"]:
        step(ActionKind.SUBMIT, "")
    result.seconds = env.instance.manifest.time_budget_seconds - env.remaining_seconds
    return result
