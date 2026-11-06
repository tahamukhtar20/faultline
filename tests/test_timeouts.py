"""Agent code that hangs must fail the step, not crash or stall the environment."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from faultline import runtime, verifiers
from faultline.env import ACTION_KINDS, FaultlineEnv
from faultline.models import ActionKind
from faultline.process import run_bounded

PIPELINE = "orderflow/pipeline.py"
PROCESS_SIGNATURE = (
    "def process(self, event: OrderEvent, *, crash_after_charge: bool = False) -> None:\n"
)
HANG = "        while True:\n            pass\n"


def action(kind: ActionKind, argument: str = "") -> dict[str, object]:
    return {"kind": ACTION_KINDS.index(kind), "argument": argument}


def patch(old: str, new: str) -> dict[str, object]:
    return action(ActionKind.PATCH, json.dumps({"path": PIPELINE, "old": old, "new": new}))


def test_episode_survives_a_hang_and_can_still_be_solved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(verifiers, "PUBLIC_TIMEOUT_SECONDS", 3.0)
    monkeypatch.setattr(verifiers, "WORKER_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(runtime, "COMMAND_TIMEOUT_SECONDS", 3.0)
    env = FaultlineEnv()
    try:
        env.reset(seed=1)

        _, reward, terminated, truncated, _ = env.step(
            patch(PROCESS_SIGNATURE, PROCESS_SIGNATURE + HANG)
        )
        assert reward < 0
        assert not terminated
        assert not truncated

        observation, *_ = env.step(action(ActionKind.RUN, "pytest -q"))
        assert "timed out after 3s" in observation["last_result"]

        env.step(patch(PROCESS_SIGNATURE + HANG, PROCESS_SIGNATURE))
        env.step(
            patch(
                'idempotency_key = f"{event.order_id}:{event.attempt}"',
                "idempotency_key = event.order_id",
            )
        )
        _, _, terminated, _, info = env.step(action(ActionKind.SUBMIT))
        assert terminated
        assert info["final_verification"]["hidden_invariants"] == 1
    finally:
        env.close()


def test_timeout_also_kills_child_processes(tmp_path: Path) -> None:
    program = (
        "import subprocess, time\n"
        "subprocess.Popen(['sh', '-c', 'sleep 2; touch survived'])\n"
        "time.sleep(60)\n"
    )

    with pytest.raises(subprocess.TimeoutExpired):
        run_bounded([sys.executable, "-c", program], cwd=tmp_path, timeout=1)

    time.sleep(2)
    assert not (tmp_path / "survived").exists()
