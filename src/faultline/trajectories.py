"""Complete episode records in JSONL, and deterministic replay of a recorded episode."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, TextIO

from faultline.models import Observation, TaskManifest, ToolAction, VerificationResult

_CHECKPOINT_REPLY = re.compile(r"checkpoint ([0-9a-f]{64})")


class TrajectoryWriter:
    def __init__(self, directory: Path, manifest: TaskManifest, level: int):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{manifest.task_id}.jsonl"
        serial = 0
        while self.path.exists():
            serial += 1
            self.path = directory / f"{manifest.task_id}-{serial}.jsonl"
        self._file: TextIO | None = self.path.open("w")
        self._write(
            {"type": "manifest", "level": level, "manifest": manifest.model_dump(mode="json")}
        )

    def _write(self, record: dict[str, Any]) -> None:
        if self._file is not None:
            self._file.write(json.dumps(record, sort_keys=True) + "\n")
            self._file.flush()

    def record_step(
        self,
        *,
        step: int,
        action: ToolAction,
        observation: Observation,
        reward: float,
        breakdown: dict[str, float],
        verification: VerificationResult,
        workspace_hash: str,
        seconds: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self._write(
            {
                "type": "step",
                "step": step,
                "action": action.model_dump(mode="json"),
                "observation": observation.model_dump(mode="json"),
                "reward": reward,
                "reward_breakdown": breakdown,
                "verification": {
                    "public_behavior": verification.public_behavior,
                    "hidden_invariants": verification.hidden_invariants,
                    "resilience": verification.resilience,
                    "performance": verification.performance,
                    "integrity_ok": verification.integrity_ok,
                    "critical_invariants_ok": verification.critical_invariants_ok,
                    "success": verification.success,
                    "failure_categories": list(verification.failure_categories),
                },
                "workspace_hash": workspace_hash,
                "seconds": round(seconds, 3),
                "terminated": terminated,
                "truncated": truncated,
            }
        )

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def load_trajectory(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    return records[0], records[1:]


def replay_trajectory(path: Path) -> list[str]:
    """Re-run a recorded episode and return every difference from the recording.

    Workspace hashes, rewards and verifier components must match. Checkpoint IDs can differ,
    because a live database's bytes are not reproducible, so recorded IDs are mapped to new ones.
    """
    from faultline.env import ACTION_KINDS, FaultlineEnv
    from faultline.models import ActionKind

    header, steps = load_trajectory(path)
    manifest = TaskManifest.model_validate(header["manifest"])
    env = FaultlineEnv(level=header["level"], replay_incidents=False)
    differences: list[str] = []
    checkpoint_ids: dict[str, str] = {}
    try:
        env.reset(seed=manifest.seed)
        if env.instance.manifest != manifest:
            return ["the task generated from this seed differs from the recorded task"]
        for record in steps:
            kind = ActionKind(record["action"]["kind"])
            argument = record["action"]["argument"]
            if kind is ActionKind.RESTORE:
                argument = checkpoint_ids.get(argument.strip(), argument)
            _, reward, terminated, truncated, info = env.step(
                {"kind": ACTION_KINDS.index(kind), "argument": argument}
            )
            step = record["step"]
            if kind is ActionKind.CHECKPOINT:
                recorded = _CHECKPOINT_REPLY.search(record["observation"]["last_result"])
                replayed = env._last_observation_required().last_result
                current = _CHECKPOINT_REPLY.search(replayed)
                if recorded and current:
                    checkpoint_ids[recorded.group(1)] = current.group(1)
            if info["workspace_hash"] != record["workspace_hash"]:
                differences.append(f"step {step}: workspace differs")
            if abs(reward - record["reward"]) > 1e-9:
                differences.append(f"step {step}: reward {reward} != {record['reward']}")
            if (terminated, truncated) != (record["terminated"], record["truncated"]):
                differences.append(f"step {step}: termination differs")
            recorded_verification = record["verification"]
            verification = env.verification
            for field in ("public_behavior", "hidden_invariants", "resilience", "performance"):
                if getattr(verification, field) != recorded_verification[field]:
                    differences.append(f"step {step}: {field} differs")
    finally:
        env.close()
    return differences
