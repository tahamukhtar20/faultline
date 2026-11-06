"""The boundary around agent-written code, attacked from inside a real sandbox."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from faultline.runtime import EpisodeRuntime
from faultline.sandbox.docker import Container, docker, remove_sandboxes
from faultline.taskgen import OrderflowTaskGenerator

pytestmark = pytest.mark.docker


@pytest.fixture(scope="module")
def runtime() -> Iterator[EpisodeRuntime]:
    instance = OrderflowTaskGenerator().generate_variants(
        ["duplicate_charge/attempt_in_key"], level=2, seed=2
    )
    episode = EpisodeRuntime(instance, replay_incidents=False)
    yield episode
    episode.close()


def inspect(container: Container) -> dict[str, Any]:
    details: dict[str, Any] = json.loads(docker("inspect", container.name).stdout)[0]
    return details


def run_python(container: Container, program: str, timeout: float = 20) -> str:
    result = container.text(["python", "-c", program], timeout=timeout)
    return result.stdout + result.stderr


CONNECT = """
import socket
for host, port in [("1.1.1.1", 53), ("8.8.8.8", 443), ("host.docker.internal", 22), ("172.17.0.1", 22)]:
    connection = socket.socket()
    connection.settimeout(2)
    try:
        connection.connect((host, port))
        print("REACHED", host, port)
    except OSError:
        pass
print("done")
"""


def test_containers_run_with_a_locked_down_configuration(runtime: EpisodeRuntime) -> None:
    assert runtime.stack.api is not None
    for container in (runtime.solver, runtime.stack.api, runtime.stack.workers[0]):
        config = inspect(container)
        host = config["HostConfig"]
        assert host["ReadonlyRootfs"] is True
        assert host["CapDrop"] == ["ALL"] or "ALL" in host["CapDrop"]
        assert host["Privileged"] is False
        assert "no-new-privileges" in host["SecurityOpt"]
        assert host["PidsLimit"] == 128
        assert host["Memory"] == host["MemorySwap"] > 0
        assert config["Config"]["User"] == "10001:10001"
        sources = [mount.get("Source", "") for mount in config["Mounts"]]
        assert not any("docker.sock" in source for source in sources)
        assert all(mount["Type"] == "volume" for mount in config["Mounts"])


def test_agent_code_is_not_root_and_cannot_write_outside_the_workspace(
    runtime: EpisodeRuntime,
) -> None:
    output = run_python(
        runtime.solver,
        """
import os
print("uid", os.getuid())
for path in ("/usr/evil", "/etc/evil", "/opt/faultline/toolbox.py"):
    try:
        open(path, "w")
        print("WROTE", path)
    except OSError:
        pass
""",
    )
    assert "uid 10001" in output
    assert "WROTE" not in output


def test_solver_has_no_network_at_all(runtime: EpisodeRuntime) -> None:
    assert inspect(runtime.solver)["HostConfig"]["NetworkMode"] == "none"
    output = run_python(runtime.solver, CONNECT)
    assert "REACHED" not in output
    assert "done" in output


def test_service_can_reach_neither_the_internet_nor_the_host(runtime: EpisodeRuntime) -> None:
    assert runtime.stack.api is not None
    output = run_python(runtime.stack.api, CONNECT)
    assert "REACHED" not in output
    assert "done" in output
    gateway = docker(
        "network",
        "inspect",
        "--format",
        "{{json .IPAM.Config}}",
        runtime.stack.api.name.rsplit("-", 3)[0],
    ).stdout.decode()
    assert '"Gateway":"' not in gateway or "invalid" in gateway or '"Gateway":""' in gateway


def test_service_cannot_control_the_ledger_over_the_network(runtime: EpisodeRuntime) -> None:
    assert runtime.stack.api is not None
    host = runtime.stack.provider.host
    output = run_python(
        runtime.stack.api,
        f"""
import httpx
for method, path in [("GET", "/ledger"), ("POST", "/ledger"), ("POST", "/load"), ("GET", "/"),
                     ("DELETE", "/charges"), ("POST", "/release"), ("POST", "/hold")]:
    print(method, path, httpx.request(method, "http://{host}:8100" + path, json={{}}).status_code)
""",
    )
    assert "200" not in output
    assert "/ledger" in output


def test_verifier_code_is_not_visible_to_agent_code(runtime: EpisodeRuntime) -> None:
    output = run_python(
        runtime.solver,
        """
import os
hits = []
for root, _, names in os.walk("/"):
    if root.startswith(("/proc", "/sys", "/usr", "/dev")):
        continue
    hits += [os.path.join(root, name) for name in names if name in ("scenarios.py", "verifiers.py")]
print("hits", hits)
""",
        timeout=30,
    )
    assert "hits []" in output


def test_process_and_memory_limits_hold(runtime: EpisodeRuntime) -> None:
    forks = run_python(
        runtime.solver,
        """
import os, time
count = 0
try:
    for _ in range(2000):
        if os.fork() == 0:
            time.sleep(3)
            os._exit(0)
        count += 1
except OSError:
    print("fork refused after", count)
""",
    )
    assert "fork refused" in forks

    memory = runtime.solver.text(
        ["python", "-c", "x = bytearray(2 * 1024 ** 3); print('ALLOCATED')"]
    )
    assert "ALLOCATED" not in memory.stdout


def test_workspace_size_is_capped(runtime: EpisodeRuntime) -> None:
    output = run_python(
        runtime.solver,
        """
try:
    with open("/workspace/big.bin", "wb") as file:
        for _ in range(512):
            file.write(b"x" * 1024 * 1024)
    print("WROTE ALL")
except OSError as error:
    print("capped", error.errno)
""",
        timeout=40,
    )
    runtime.solver.run(["rm", "-f", "/workspace/big.bin"])
    assert "WROTE ALL" not in output
    assert "capped" in output


def test_workspace_code_cannot_shadow_the_toolbox(runtime: EpisodeRuntime) -> None:
    runtime.solver.run(["sh", "-c", "echo 'raise SystemExit(99)' > /workspace/json.py"])
    try:
        assert "pipeline" not in runtime._toolbox("read", "orderflow/config.py")
        assert "LEASE" in runtime._toolbox("read", "orderflow/config.py").upper()
    finally:
        runtime.solver.run(["rm", "-f", "/workspace/json.py"])


def test_closing_removes_every_resource() -> None:
    instance = OrderflowTaskGenerator().generate_variants(
        ["schema_skew/v1_dropped"], level=2, seed=3
    )
    episode = EpisodeRuntime(instance, replay_incidents=False)
    selector = f"label=faultline.sandbox={episode._sandbox.sandbox_id}"

    assert docker("ps", "-q", "--filter", selector).stdout.strip()
    episode.close()

    assert not docker("ps", "-aq", "--filter", selector).stdout.strip()
    assert not docker("volume", "ls", "-q", "--filter", selector).stdout.strip()
    assert not docker("network", "ls", "-q", "--filter", selector).stdout.strip()


def test_orphans_can_be_cleaned_up() -> None:
    # Safe alongside other tests only with a scoped ID, so this removes a sandbox that never existed.
    remove_sandboxes("does-not-exist")
