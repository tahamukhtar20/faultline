"""Writes seeded tasks as Harbor task directories.

Each task has the broken service and a provider sidecar for the agent, and a separate verifier
image holding the private scenarios. Only the agent's /app directory crosses between them.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from importlib.metadata import version
from pathlib import Path

import faultline
from faultline.sandbox.docker import BUILD_CONTEXT
from faultline.taskgen import LEVELS, OrderflowTaskGenerator, TaskInstance
from faultline.verifiers import count_public_tests

PYTHON_IMAGE = (
    "python:3.13-slim@sha256:09a17cad02668ae61ece136b4fd36ac3849e14f670ebedb637ea3e15580525bc"
)
_PROVIDER_DIR = Path(__file__).parent / "provider"
# Everything the grader imports. An exported task must work with nothing else installed.
_GRADER_FILES = (
    "__init__.py",
    "grader/__init__.py",
    "grader/__main__.py",
    "grader/stack.py",
    "integrity.py",
    "models.py",
    "rewards.py",
    "sandbox/__init__.py",
    "sandbox/archive.py",
    "sandbox/crash_api.py",
    "sandbox/errors.py",
    "sandbox/provider.py",
    "sandbox/supervisor.py",
    "scenarios.py",
    "service_stack.py",
)
_AGENT_SECONDS = 3600
_VERIFIER_SECONDS = 900

_INSTRUCTION = """\
{issue}

## The service

The code is in `/app`: a FastAPI order API and a worker that charges and ships orders through an
outbox. Start them from `/app` with:

```
python -m uvicorn orderflow.api:app --port 8000
python -m orderflow.worker
```

Data lives in `/data`. Delete `/data/orders.db*` for a clean start.

The payment and shipping provider is running at `http://provider:8100`. It takes `POST /charges`
and `POST /shipments`, and `GET /ledger` shows every charge and shipment it has recorded.

Public tests run with `python -m pytest tests` from `/app`.

## Rules

- Repair the code in `/app`. Only those files are kept, so nothing you leave running carries over.
- Do not change anything in `tests/` or `orderflow/legacy_producer.py`, which is the previous
  release and stays as it is. Do not add test configuration such as `conftest.py` or
  `pyproject.toml`. Any of these fails the task.
- Your repair is checked by private scenarios against a separate copy of the service, including
  real process crashes. The public tests passing does not mean the repair is complete.
"""

_TASK_TOML = """\
schema_version = "1.4"
artifacts = ["/app"]

[task]
name = "faultline/{task_id}"
description = "Repair a faulty order service so that it survives crashes and bad events."
keywords = ["debugging", "distributed-systems", "reliability", "python"]

[metadata]
category = "programming"
difficulty_explanation = "Level {level} of 5, computed difficulty {score}."
faultline_level = {level}
faultline_seed = {seed}
faultline_difficulty = {score}

[agent]
timeout_sec = {agent_seconds}

[verifier]
environment_mode = "separate"
timeout_sec = {verifier_seconds}
network_mode = "no-network"

[verifier.environment]
cpus = 4
memory_mb = 4096

[environment]
cpus = 2
memory_mb = 4096
"""

_AGENT_DOCKERFILE = """\
FROM {image}

RUN pip install --no-cache-dir {pins}

COPY workspace/ /app/
RUN mkdir -p /data
WORKDIR /app
"""

_COMPOSE = """\
services:
  main:
    depends_on:
      provider:
        condition: service_healthy
  provider:
    build:
      context: ./provider
    healthcheck:
      test:
        - CMD
        - python
        - -c
        - import urllib.request; urllib.request.urlopen('http://localhost:8100/ledger')
      interval: 1s
      retries: 30
"""

_PROVIDER_DOCKERFILE = """\
FROM {image}

COPY provider.py dev_provider.py /opt/provider/
WORKDIR /opt/provider
CMD ["python", "dev_provider.py"]
"""

_VERIFIER_DOCKERFILE = """\
FROM {image}

RUN pip install --no-cache-dir {pins} {pydantic}

COPY . /tests/
RUN mkdir -p /opt/faultline \\
    && cd /tests/faultline/sandbox \\
    && cp crash_api.py provider.py supervisor.py /opt/faultline/ \\
    && chmod -R go-rwx /tests
"""

_TEST_SH = """\
#!/bin/sh
cd /tests
exec python -m faultline.grader --workspace /app --manifest /tests/manifest.json \\
    --out /logs/verifier
"""

_SOLVE_SH = """\
#!/bin/sh
exec python3 /solution/apply_patches.py
"""

_APPLY_PATCHES = """\
import json
from pathlib import Path

for patch in json.loads(Path("/solution/patches.json").read_text()):
    path = Path("/app") / patch["path"]
    text = path.read_text()
    assert text.count(patch["old"]) == 1, patch["old"]
    path.write_text(text.replace(patch["old"], patch["new"]))
"""


def _service_pins() -> str:
    """The same versions the container sandbox uses, read from its Dockerfile."""
    dockerfile = (BUILD_CONTEXT / "Dockerfile").read_text()
    return " ".join(re.findall(r"^([\w-]+==[\w.]+)$", dockerfile, re.MULTILINE))


def _write(path: Path, content: str | bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        content = content.encode()
    path.write_bytes(content)
    path.chmod(0o755 if executable else 0o644)


def _write_environment(directory: Path, instance: TaskInstance) -> None:
    for name, entry in instance.files.items():
        _write(directory / "workspace" / name, entry.data)
    _write(
        directory / "Dockerfile",
        _AGENT_DOCKERFILE.format(image=PYTHON_IMAGE, pins=_service_pins()),
    )
    _write(directory / "docker-compose.yaml", _COMPOSE)
    _write(
        directory / "provider" / "Dockerfile",
        _PROVIDER_DOCKERFILE.format(image=PYTHON_IMAGE),
    )
    _write(directory / "provider" / "provider.py", (BUILD_CONTEXT / "provider.py").read_bytes())
    _write(
        directory / "provider" / "dev_provider.py",
        (_PROVIDER_DIR / "dev_provider.py").read_bytes(),
    )


def _write_tests(directory: Path, instance: TaskInstance) -> None:
    package = Path(faultline.__file__).parent
    for name in _GRADER_FILES:
        _write(directory / "faultline" / name, (package / name).read_bytes())
    _write(
        directory / "manifest.json",
        json.dumps(
            {
                "task_id": instance.manifest.task_id,
                "protected_hashes": instance.protected_hashes,
                "public_tests": count_public_tests(instance),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _write(
        directory / "Dockerfile",
        _VERIFIER_DOCKERFILE.format(
            image=PYTHON_IMAGE,
            pins=_service_pins(),
            pydantic=f"pydantic=={version('pydantic')}",
        ),
    )
    _write(directory / "test.sh", _TEST_SH, executable=True)


def _write_solution(directory: Path, instance: TaskInstance) -> None:
    _write(directory / "solve.sh", _SOLVE_SH, executable=True)
    _write(directory / "apply_patches.py", _APPLY_PATCHES)
    _write(directory / "patches.json", json.dumps(list(instance.gold_patches), indent=2) + "\n")


def export_task(instance: TaskInstance, out: Path) -> Path:
    manifest = instance.manifest
    directory = out / manifest.task_id
    if directory.exists():
        shutil.rmtree(directory)
    _write(directory / "instruction.md", _INSTRUCTION.format(issue=manifest.issue))
    _write(
        directory / "task.toml",
        _TASK_TOML.format(
            task_id=manifest.task_id,
            level=manifest.difficulty.level,
            seed=manifest.seed,
            score=manifest.difficulty.score,
            agent_seconds=_AGENT_SECONDS,
            verifier_seconds=_VERIFIER_SECONDS,
        ),
    )
    _write_environment(directory / "environment", instance)
    _write_tests(directory / "tests", instance)
    _write_solution(directory / "solution", instance)
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("harbor-tasks"))
    parser.add_argument("--levels", default=",".join(str(level) for level in LEVELS))
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    generator = OrderflowTaskGenerator()
    for level in (int(value) for value in args.levels.split(",")):
        for seed in range(args.seeds):
            directory = export_task(generator.generate(level=level, seed=seed), args.out)
            print(directory)


if __name__ == "__main__":
    main()
