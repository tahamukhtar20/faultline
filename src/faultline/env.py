"""Gymnasium interface for Faultline."""

from __future__ import annotations

import difflib
import tempfile
import time
from pathlib import Path
from string import printable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from faultline.checkpoints import CheckpointError, CheckpointState, CheckpointStore
from faultline.models import ActionKind, Observation, RewardMode, ToolAction, VerificationResult
from faultline.rewards import RewardEngine
from faultline.runtime import MAX_OUTPUT, EpisodeRuntime
from faultline.sandbox.archive import ArchiveError, Tree, tree_hash
from faultline.taskgen import OrderflowTaskGenerator, TaskInstance
from faultline.trajectories import TrajectoryWriter
from faultline.verifiers import verify_tree

ACTION_KINDS = tuple(ActionKind)
_PRINTABLE = frozenset(printable)
_READ_ONLY = frozenset({ActionKind.READ, ActionKind.SEARCH, ActionKind.LOGS, ActionKind.QUERY})


def _printable(text: str, limit: int) -> str:
    """Tool output is untrusted: keep it inside the observation space, whatever it contains."""
    cleaned = "".join(char if char in _PRINTABLE else "?" for char in text)
    if len(cleaned) <= limit:
        return cleaned
    marker = f"\n... [{len(cleaned) - limit} characters truncated] ...\n"
    keep = limit - len(marker)
    return cleaned[: keep // 2] + marker + cleaned[-(keep - keep // 2) :]


def _diff_summary(baseline: Tree, current: Tree) -> str:
    changes = []
    for path in sorted(set(baseline) | set(current)):
        old = baseline[path].data if path in baseline else b""
        new = current[path].data if path in current else b""
        if old == new:
            continue
        diff = difflib.ndiff(
            old.decode(errors="replace").splitlines(), new.decode(errors="replace").splitlines()
        )
        added = removed = 0
        for line in diff:
            added += line.startswith("+ ")
            removed += line.startswith("- ")
        changes.append(f"{path} +{added} -{removed}")
    return "; ".join(changes) if changes else "no changes"


class FaultlineEnv(gym.Env[dict[str, Any], dict[str, Any]]):
    """A repair episode on a generated task.

    Use reset(seed=..., options={"level": 1-5}) to choose a task, or options={"instance": ...} to
    run a specific one. The checkpoint, restore and branch methods let an experimenter rewind and
    fork whole episodes, including the budget and reward state. The agent's own checkpoint and
    restore actions only move files and data: the budget keeps counting and a repair paid for
    before a restore cannot be collected again.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        *,
        level: int = 1,
        reward_mode: RewardMode = RewardMode.SHAPED,
        replay_incidents: bool = True,
        checkpoint_dir: Path | None = None,
        trajectory_dir: Path | None = None,
    ):
        super().__init__()
        self.level = level
        self.reward_mode = reward_mode
        self.replay_incidents = replay_incidents
        self.generator = OrderflowTaskGenerator()
        self.reward_engine = RewardEngine(mode=reward_mode)
        self.trajectory_dir = trajectory_dir
        self.action_space = spaces.Dict(
            {
                "kind": spaces.Discrete(len(ACTION_KINDS)),
                "argument": spaces.Text(min_length=0, max_length=16_384, charset=printable),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "issue": spaces.Text(max_length=4_096, charset=printable),
                "last_result": spaces.Text(max_length=MAX_OUTPUT, charset=printable),
                "visible_test_summary": spaces.Text(max_length=2_000, charset=printable),
                "service_summary": spaces.Text(max_length=1_000, charset=printable),
                "diff_summary": spaces.Text(max_length=1_000, charset=printable),
                "remaining_steps": spaces.Box(0, 10_000, shape=(), dtype=np.int64),
                "remaining_seconds": spaces.Box(0, 100_000, shape=(), dtype=np.float32),
            }
        )
        self._store_directory: tempfile.TemporaryDirectory[str] | None = None
        if checkpoint_dir is None:
            self._store_directory = tempfile.TemporaryDirectory(prefix="faultline-checkpoints-")
            checkpoint_dir = Path(self._store_directory.name)
        self.checkpoints = CheckpointStore(checkpoint_dir)
        self._instance: TaskInstance | None = None
        self._runtime: EpisodeRuntime | None = None
        self._trajectory: TrajectoryWriter | None = None
        self._steps = 0
        self._spent_seconds = 0.0
        self._episode_started = 0.0
        self._finished = False
        self._integrity_failed = False
        self._parent_checkpoint: str | None = None
        self._workspace_hash = ""
        self._verification: VerificationResult | None = None
        self._last_observation: Observation | None = None

    @property
    def instance(self) -> TaskInstance:
        if self._instance is None:
            raise RuntimeError("environment must be reset before use")
        return self._instance

    @property
    def runtime(self) -> EpisodeRuntime:
        if self._runtime is None:
            raise RuntimeError("environment must be reset before use")
        return self._runtime

    @property
    def verification(self) -> VerificationResult:
        if self._verification is None:
            raise RuntimeError("environment must be reset before use")
        return self._verification

    def _elapsed(self) -> float:
        return self._spent_seconds + (time.monotonic() - self._episode_started)

    def _observation(self, last_result: str, workspace: Tree | None = None) -> Observation:
        manifest = self.instance.manifest
        diff = (
            self._last_observation.diff_summary
            if workspace is None and self._last_observation is not None
            else _diff_summary(self.instance.files, workspace or self.instance.files)
        )
        return Observation(
            issue=manifest.issue,
            last_result=_printable(last_result, MAX_OUTPUT),
            visible_test_summary=_printable(self.runtime.last_test_summary, 2_000),
            service_summary=_printable(self.runtime.service_summary(), 1_000),
            diff_summary=_printable(diff, 1_000),
            remaining_steps=max(0, manifest.step_budget - self._steps),
            remaining_seconds=max(0.0, manifest.time_budget_seconds - self._elapsed()),
        )

    @staticmethod
    def _as_space_value(observation: Observation) -> dict[str, Any]:
        payload = observation.model_dump()
        payload["remaining_steps"] = np.asarray(payload["remaining_steps"], dtype=np.int64)
        payload["remaining_seconds"] = np.asarray(payload["remaining_seconds"], dtype=np.float32)
        return payload

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self._close_episode()
        options = options or {}
        instance = options.get("instance")
        if instance is None:
            instance = self.generator.generate(
                level=int(options.get("level", self.level)), seed=0 if seed is None else seed
            )
        self._start_episode(instance)
        return self._as_space_value(self._last_observation_required()), {
            "task_id": instance.manifest.task_id,
            "difficulty": instance.manifest.difficulty.model_dump(),
        }

    def _start_episode(self, instance: TaskInstance) -> None:
        self._instance = instance
        self._verification = verify_tree(instance, instance.files)
        self.reward_engine.prime(self._verification)
        self._runtime = EpisodeRuntime(instance, replay_incidents=self.replay_incidents)
        self._steps = 0
        self._spent_seconds = 0.0
        self._episode_started = time.monotonic()
        self._finished = False
        self._integrity_failed = False
        self._parent_checkpoint = None
        self._workspace_hash = tree_hash(instance.files)
        self._last_observation = None
        self._last_observation = self._observation("episode initialized", instance.files)
        if self.trajectory_dir is not None:
            self._trajectory = TrajectoryWriter(self.trajectory_dir, instance.manifest, self.level)

    def _last_observation_required(self) -> Observation:
        if self._last_observation is None:
            raise RuntimeError("environment must be reset before use")
        return self._last_observation

    def _refresh_verification(self) -> Tree | None:
        """Export the workspace and rescore it if anything changed, including from agent code.

        A workspace that cannot be exported safely, for example because agent code planted a
        symlink, is an integrity violation.
        """
        try:
            workspace = self.runtime.export_workspace()
        except ArchiveError as error:
            self._verification = VerificationResult(
                public_behavior=0,
                hidden_invariants=0,
                resilience=0,
                performance=0,
                integrity_ok=False,
                critical_invariants_ok=False,
                diagnostics=(f"unsafe workspace: {error}",),
            )
            return None
        current_hash = tree_hash(workspace)
        if current_hash != self._workspace_hash:
            self._workspace_hash = current_hash
            self._verification = verify_tree(self.instance, workspace)
        return workspace

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._instance is None:
            raise RuntimeError("environment must be reset before use")
        if self._finished:
            raise RuntimeError("episode has ended: call reset, or restore a checkpoint")
        kind_index = int(action["kind"])
        if not 0 <= kind_index < len(ACTION_KINDS):
            raise ValueError(f"invalid action kind index: {kind_index}")
        tool_action = ToolAction(kind=ACTION_KINDS[kind_index], argument=str(action["argument"]))
        started = time.monotonic()

        self._steps += 1
        tool_result = self._execute(tool_action)
        workspace = None
        if tool_action.kind not in _READ_ONLY:
            workspace = self._refresh_verification()
        verification = self.verification
        submitted = tool_action.kind is ActionKind.SUBMIT
        breakdown = self.reward_engine.step(verification, submitted=submitted)

        integrity_failure = not verification.integrity_ok
        self._integrity_failed = self._integrity_failed or integrity_failure
        terminated = submitted or integrity_failure
        out_of_steps = self._steps >= self.instance.manifest.step_budget
        out_of_time = self._elapsed() >= self.instance.manifest.time_budget_seconds
        truncated = (out_of_steps or out_of_time) and not terminated
        self._finished = terminated or truncated
        self._last_observation = self._observation(tool_result, workspace)

        info: dict[str, Any] = {
            "reward_breakdown": {
                "progress": breakdown.progress,
                "terminal": breakdown.terminal,
                "action_cost": breakdown.action_cost,
                "integrity_penalty": breakdown.integrity_penalty,
            },
            "workspace_hash": self._workspace_hash,
        }
        if self._finished:
            info["final_verification"] = verification.model_dump()
            info["failure_categories"] = list(verification.failure_categories)
            info["success"] = verification.success
        if self._trajectory is not None:
            self._trajectory.record_step(
                step=self._steps,
                action=tool_action,
                observation=self._last_observation,
                reward=breakdown.total,
                breakdown=info["reward_breakdown"],
                verification=verification,
                workspace_hash=self._workspace_hash,
                seconds=time.monotonic() - started,
                terminated=terminated,
                truncated=truncated,
            )
        return (
            self._as_space_value(self._last_observation),
            breakdown.total,
            terminated,
            truncated,
            info,
        )

    def _execute(self, action: ToolAction) -> str:
        if action.kind is ActionKind.CHECKPOINT:
            return f"checkpoint {self.checkpoint()}"
        if action.kind is ActionKind.RESTORE:
            try:
                state = self.checkpoints.load(
                    action.argument.strip(), task_id=self.instance.manifest.task_id
                )
            except CheckpointError as error:
                return f"tool error: {error}"
            self.runtime.restore_state(state.workspace, state.data, state.ledger)
            self.runtime.noise_counter = int(state.controller["noise_counter"])
            self._parent_checkpoint = action.argument.strip()
            return f"restored checkpoint {action.argument.strip()}"
        return self.runtime.execute(action)

    def checkpoint(self) -> str:
        """Save the episode. The ID depends only on the saved state."""
        workspace, data, ledger = self.runtime.capture_state()
        manifest = self.instance.manifest
        checkpoint_id = self.checkpoints.save(
            CheckpointState(
                task_id=manifest.task_id,
                seed=manifest.seed,
                protected_hashes=self.instance.protected_hashes,
                controller={
                    "steps": self._steps,
                    "spent_seconds": round(self._elapsed(), 3),
                    "integrity_failed": self._integrity_failed,
                    "reward_engine": self.reward_engine.state(),
                    "noise_counter": self.runtime.noise_counter,
                    "last_test_summary": self.runtime.last_test_summary,
                    "verification": self.verification.model_dump(mode="json"),
                },
                ledger=ledger,
                workspace=workspace,
                data=data,
                parent=self._parent_checkpoint,
            )
        )
        self._parent_checkpoint = checkpoint_id
        return checkpoint_id

    def restore(self, checkpoint_id: str) -> dict[str, Any]:
        """Rewind the whole episode, including budgets and reward state, and reopen it."""
        state = self.checkpoints.load(checkpoint_id, task_id=self.instance.manifest.task_id)
        if state.protected_hashes != self.instance.protected_hashes:
            raise CheckpointError("protected hashes do not match this task")
        self.runtime.restore_state(state.workspace, state.data, state.ledger)
        self._load_controller(state.controller, state.workspace)
        self._parent_checkpoint = checkpoint_id
        self._finished = False
        self._last_observation = self._observation(
            f"restored checkpoint {checkpoint_id}", state.workspace
        )
        return self._as_space_value(self._last_observation)

    def _load_controller(self, controller: dict[str, Any], workspace: Tree) -> None:
        self._steps = int(controller["steps"])
        self._spent_seconds = float(controller["spent_seconds"])
        self._episode_started = time.monotonic()
        self._integrity_failed = bool(controller["integrity_failed"])
        self.reward_engine.load_state(controller["reward_engine"])
        self.runtime.noise_counter = int(controller["noise_counter"])
        self.runtime.last_test_summary = str(controller["last_test_summary"])
        self._verification = VerificationResult.model_validate(controller["verification"])
        self._workspace_hash = tree_hash(workspace)

    def branch(self, checkpoint_id: str) -> FaultlineEnv:
        """Create an independent environment that continues from a checkpoint."""
        child = FaultlineEnv(
            level=self.level,
            reward_mode=self.reward_mode,
            replay_incidents=False,
            checkpoint_dir=self.checkpoints.root,
            trajectory_dir=self.trajectory_dir,
        )
        child._start_episode(self.instance)
        child.replay_incidents = self.replay_incidents
        child.restore(checkpoint_id)
        return child

    def render(self) -> str:
        if self._last_observation is None:
            return "FaultlineEnv(uninitialized)"
        return (
            f"Issue: {self._last_observation.issue}\n"
            f"Last result: {self._last_observation.last_result}\n"
            f"Remaining steps: {self._last_observation.remaining_steps}"
        )

    def _close_episode(self) -> None:
        if self._trajectory is not None:
            self._trajectory.close()
            self._trajectory = None
        if self._runtime is not None:
            self._runtime.close()
        self._runtime = None
        self._instance = None

    def close(self) -> None:
        self._close_episode()
        if self._store_directory is not None:
            self._store_directory.cleanup()
            self._store_directory = None
