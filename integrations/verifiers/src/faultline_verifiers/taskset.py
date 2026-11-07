"""Faultline's seeded tasks as a Verifiers taskset, built on its Harbor support.

Tasks are generated on the evaluator and graded in a separate image holding the private scenarios.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTask, HarborTaskset
from verifiers.v1.tasksets.harbor.taskset import make_tar, parse_task
from verifiers.v1.utils.artifacts import collect

from faultline.exporters.harbor import export_task
from faultline.taskgen import LEVELS, OrderflowTaskGenerator


class FaultlineConfig(HarborConfig):
    levels: list[int] = list(LEVELS)
    seeds: int = 3


def _build_verifier_image(task_dir: Path) -> str:
    tag = f"faultline-verifier:{task_dir.name}"
    subprocess.run(
        ["docker", "build", "--quiet", "--tag", tag, str(task_dir / "tests")],
        check=True,
        capture_output=True,
        timeout=900,
    )
    return tag


def _use_verifier_image(task_dir: Path, image: str) -> None:
    """Verifiers runs prebuilt images only, so the task must name one instead of a Dockerfile."""
    config = task_dir / "task.toml"
    section = "[verifier.environment]\n"
    config.write_text(config.read_text().replace(section, f'{section}docker_image = "{image}"\n'))


class FaultlineTask(HarborTask):
    async def validate(self, runtime: vf.Runtime, trace: vf.Trace) -> bool | None:
        """Apply the reference repair and grade it, for `vf-validate`. Needs no model."""
        await runtime.write("/tmp/solution.tgz", make_tar(Path(self.data.task_dir) / "solution"))
        applied = await runtime.run(
            [
                "sh",
                "-c",
                "mkdir -p /solution && tar -xzf /tmp/solution.tgz -C /solution "
                "&& sh /solution/solve.sh",
            ],
            {},
        )
        if applied.exit_code:
            return False
        trace.state.artifacts = await collect(
            runtime, self.data.artifacts, max_bytes=self.data.artifact_max_bytes
        )
        await vf.grade_submission(self, trace, runtime)
        return trace.rewards["reward"].score == 1.0


class FaultlineTaskset(HarborTaskset, vf.Taskset[FaultlineTask, FaultlineConfig]):
    def load(self) -> Iterator[FaultlineTask]:
        directory = Path(tempfile.mkdtemp(prefix="faultline-verifiers-"))
        generator = OrderflowTaskGenerator()
        index = 0
        for level in self.config.levels:
            for seed in range(self.config.seeds):
                task_dir = export_task(generator.generate(level=level, seed=seed), directory)
                _use_verifier_image(task_dir, _build_verifier_image(task_dir))
                yield FaultlineTask(parse_task(task_dir, index, self.config), self.config.task)
                index += 1
