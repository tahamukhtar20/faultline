"""The grader exported tasks bundle, run the way a framework runs it: alone in one container."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from conftest import apply_patches, gold_tree

from faultline.exporters.harbor import export_task
from faultline.sandbox.archive import FileEntry, Tree
from faultline.sandbox.docker import docker
from faultline.taskgen import OrderflowTaskGenerator, TaskInstance

pytestmark = pytest.mark.docker


@pytest.fixture(scope="module")
def task() -> TaskInstance:
    return OrderflowTaskGenerator().generate_variants(
        ["duplicate_charge/attempt_in_key"], level=2, seed=1, rename=False
    )


@pytest.fixture(scope="module")
def grader_image(task: TaskInstance, tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    directory = export_task(task, tmp_path_factory.mktemp("harbor"))
    tag = f"faultline-grader-test:{task.manifest.task_id}"
    docker("build", "--quiet", "--tag", tag, str(directory / "tests"), timeout=900)
    yield tag
    docker("image", "rm", "--force", tag, check=False)


def grade(image: str, tree: Tree, tmp_path: Path) -> dict[str, Any]:
    app = tmp_path / "app"
    out = tmp_path / "out"
    out.mkdir()
    for name, entry in tree.items():
        target = app / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(entry.data)
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--mount",
            f"type=bind,source={app},target=/submitted,readonly",
            "--mount",
            f"type=bind,source={out},target=/out",
            image,
            "sh",
            "-c",
            "cp -r /submitted /app && cd /tests && python -m faultline.grader "
            "--workspace /app --manifest /tests/manifest.json --out /out; "
            "status=$?; chmod -R a+rX /out; exit $status",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2_000:]
    details: dict[str, Any] = json.loads((out / "details.json").read_text())
    details["reward"] = json.loads((out / "reward.json").read_text())["reward"]
    return details


def test_the_gold_patch_earns_full_reward(
    task: TaskInstance, grader_image: str, tmp_path: Path
) -> None:
    result = grade(grader_image, gold_tree(task), tmp_path)

    assert result["success"], result["scenarios"]
    assert result["reward"] == 1.0


def test_the_injected_fault_fails(task: TaskInstance, grader_image: str, tmp_path: Path) -> None:
    result = grade(grader_image, task.files, tmp_path)

    assert not result["success"]
    assert "duplicate_charge" in result["failure_categories"]
    assert result["reward"] < 0.6


def test_changing_a_protected_test_earns_nothing(
    task: TaskInstance, grader_image: str, tmp_path: Path
) -> None:
    tree = dict(gold_tree(task))
    tree["tests/test_public_behavior.py"] = FileEntry(b"def test_nothing():\n    pass\n")

    result = grade(grader_image, tree, tmp_path)

    assert not result["integrity_ok"]
    assert result["reward"] == 0.0


_DETACH = """
import os, time
if os.fork() == 0:
    os.setsid()
    if os.fork() == 0:
        open("/tmp/survivor", "w").write(str(os.getpid()))
        time.sleep(60)
    os._exit(0)
time.sleep(float(os.environ["SERVICE_SECONDS"]))
"""
_SUPERVISE = """
import os, subprocess, sys, time
service = subprocess.Popen(
    ["python", "-I", "/opt/faultline/supervisor.py", "20000", "python", "-c", sys.argv[1]],
    cwd="/",
    env={**os.environ, "SERVICE_SECONDS": sys.argv[2]},
)
while not os.path.exists("/tmp/survivor"):
    time.sleep(0.05)
survivor = int(open("/tmp/survivor").read())
assert os.path.exists(f"/proc/{survivor}")
if sys.argv[3] == "kill":
    service.terminate()
service.wait()
assert not os.path.exists(f"/proc/{survivor}"), "a detached process survived"
"""


@pytest.mark.parametrize(("service_seconds", "how"), [("60", "kill"), ("1", "exit")])
def test_a_detached_process_dies_with_its_service(
    grader_image: str, service_seconds: str, how: str
) -> None:
    completed = docker(
        "run",
        "--rm",
        "--network=none",
        grader_image,
        "python",
        "-c",
        _SUPERVISE,
        _DETACH,
        service_seconds,
        how,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr.decode()[-1_000:]


def test_agent_code_cannot_read_the_private_scenarios_or_write_the_reward(
    task: TaskInstance, grader_image: str, tmp_path: Path
) -> None:
    probe = (
        "import os\n"
        "paths = [('r', '/tests/faultline/scenarios.py'), ('a', '/out/reward.json')]\n"
        "for action, path in paths:\n"
        "    try:\n"
        "        open(path, action).close()\n"
        "    except OSError:\n"
        "        continue\n"
        "    os.kill(os.getpid(), 9)\n"
    )
    tree = apply_patches(
        gold_tree(task),
        [{"path": "orderflow/config.py", "old": "import os\n", "new": probe}],
    )

    assert grade(grader_image, tree, tmp_path)["success"]


def test_a_correct_repair_that_writes_at_startup_passes(
    task: TaskInstance, grader_image: str, tmp_path: Path
) -> None:
    clear = "        connection.execute(\"UPDATE outbox SET {} = NULL WHERE status = 'done'\")\n"
    reconcile = (
        "connection.executescript(SCHEMA)\n"
        + clear.format("lease_owner")
        + clear.format("last_error")
    )
    tree = apply_patches(
        gold_tree(task),
        [
            {
                "path": "orderflow/db.py",
                "old": "connection.executescript(SCHEMA)\n",
                "new": reconcile,
            }
        ],
    )

    assert grade(grader_image, tree, tmp_path)["success"]
