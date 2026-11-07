"""The OpenEnv server, driven over its network protocol. No model is called.

Needs `integrations/openenv` installed, so it is skipped in the main environment.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from collections.abc import Iterator

import pytest

pytest.importorskip("faultline_env")

from faultline_env import FaultlineAction, FaultlineClient

from faultline.taskgen import OrderflowTaskGenerator

pytestmark = pytest.mark.docker

LEVEL = 2
SEED = 1


@pytest.fixture(scope="module")
def server_url() -> Iterator[str]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "faultline_env.server.app", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            with socket.socket() as connection:
                if connection.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.1)
        yield f"http://127.0.0.1:{port}"
    finally:
        process.terminate()
        process.wait(30)


def test_the_reference_repair_succeeds_over_the_protocol(server_url: str) -> None:
    gold = OrderflowTaskGenerator().generate(level=LEVEL, seed=SEED).gold_patches

    with FaultlineClient(base_url=server_url, message_timeout_s=300).sync() as client:
        started = client.reset(seed=SEED, level=LEVEL)
        assert started.observation.issue
        assert not started.done

        total = 0.0
        for patch in gold:
            result = client.step(FaultlineAction(kind="patch", argument=json.dumps(patch)))
            total += result.reward or 0.0
        finished = client.step(FaultlineAction(kind="submit"))
        total += finished.reward or 0.0

    assert finished.done
    assert total > 0.4


def test_submitting_the_broken_code_does_not_pay_the_terminal_reward(server_url: str) -> None:
    with FaultlineClient(base_url=server_url, message_timeout_s=300).sync() as client:
        client.reset(seed=SEED, level=LEVEL)
        finished = client.step(FaultlineAction(kind="submit"))

    assert finished.done
    assert (finished.reward or 0.0) < 0.4
