"""Faultline as an Inspect AI task.

The agent works in the `default` service, beside the provider sidecar, on a network with no route
out. The scorer carries a tar of /app to the `grader` service, which holds the private scenarios,
has no network, and refuses anything in the archive that is not a plain file.

    inspect eval src/faultline/exporters/inspect_task.py@orderflow \
        --model anthropic/claude-haiku-5-5 \
        -T levels=1,2 -T seeds=1
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import bash, text_editor
from inspect_ai.util import sandbox

from faultline.exporters.harbor import export_task
from faultline.taskgen import LEVELS, OrderflowTaskGenerator

_COMPOSE = """\
services:
  default:
    build: ./environment
    init: true
    command: tail -f /dev/null
    working_dir: /app
    networks: [sandbox]
    depends_on:
      provider:
        condition: service_healthy
  provider:
    build: ./environment/provider
    init: true
    networks: [sandbox]
    healthcheck:
      test:
        - CMD
        - python
        - -c
        - import socket; socket.create_connection(('localhost', 8100), 1)
      interval: 1s
      retries: 30
  grader:
    build: ./tests
    init: true
    command: tail -f /dev/null
    network_mode: none
networks:
  sandbox:
    internal: true
"""
_SUBMISSION = "/tmp/faultline-submission.tar"
_RESULTS = "/logs/verifier"
_GRADER_SECONDS = 900
_COMMAND_SECONDS = 180


def _samples(levels: list[int], seeds: int, directory: Path) -> list[Sample]:
    generator = OrderflowTaskGenerator()
    samples = []
    for level in levels:
        for seed in range(seeds):
            instance = generator.generate(level=level, seed=seed)
            task_dir = export_task(instance, directory)
            compose = task_dir / "compose.yaml"
            compose.write_text(_COMPOSE)
            samples.append(
                Sample(
                    id=instance.manifest.task_id,
                    input=(task_dir / "instruction.md").read_text(),
                    sandbox=("docker", str(compose)),
                    metadata={
                        "level": level,
                        "seed": seed,
                        "difficulty": instance.manifest.difficulty.score,
                        "gold_patches": list(instance.gold_patches),
                    },
                )
            )
    return samples


@scorer(metrics=[mean()])
def private_scenarios() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        agent = sandbox()
        grader = sandbox("grader")
        packed = await agent.exec(["tar", "-cf", _SUBMISSION, "-C", "/app", "."], timeout=120)
        if not packed.success:
            return Score(value=0.0, explanation=f"could not collect /app: {packed.stderr}")
        await grader.write_file(_SUBMISSION, await agent.read_file(_SUBMISSION, text=False))
        graded = await grader.exec(
            [
                "sh",
                "-c",
                "cd /tests && exec python -m faultline.grader "
                f"--tar {_SUBMISSION} --manifest /tests/manifest.json --out {_RESULTS}",
            ],
            timeout=_GRADER_SECONDS,
        )
        if not graded.success:
            raise RuntimeError(f"the grader itself failed: {graded.stderr[-2_000:]}")
        reward = json.loads(await grader.read_file(f"{_RESULTS}/reward.json"))
        details = json.loads(await grader.read_file(f"{_RESULTS}/details.json"))
        return Score(
            value=float(reward["reward"]),
            answer="pass" if details["success"] else "fail",
            explanation=graded.stdout,
            metadata={
                "failure_categories": details["failure_categories"],
                "public_behavior": details["public_behavior"],
                "hidden_invariants": details["hidden_invariants"],
                "resilience": details["resilience"],
                "performance": details["performance"],
            },
        )

    return score


@solver
def gold_patch() -> Solver:
    """Applies the reference repair, to check that a task can be solved. Needs no model."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        agent = sandbox()
        for patch in state.metadata["gold_patches"]:
            path = f"/app/{patch['path']}"
            text = await agent.read_file(path)
            await agent.write_file(path, text.replace(patch["old"], patch["new"]))
        return state

    return solve


@solver
def no_change() -> Solver:
    """Submits the broken code untouched, to check that a task cannot pass by doing nothing."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return state

    return solve


@task
def orderflow(levels: str = "1,2,3,4,5", seeds: int = 3) -> Task:
    selected = [int(level) for level in str(levels).split(",")]
    if unknown := set(selected) - set(LEVELS):
        raise ValueError(f"unknown levels: {sorted(unknown)}")
    directory = Path(tempfile.mkdtemp(prefix="faultline-inspect-"))
    return Task(
        dataset=MemoryDataset(_samples(selected, seeds, directory)),
        solver=react(tools=[bash(_COMMAND_SECONDS), text_editor(_COMMAND_SECONDS)]),
        scorer=private_scenarios(),
    )
