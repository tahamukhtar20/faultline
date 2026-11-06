"""Task generation from exact, reversible fault templates."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from faultline.models import TaskManifest


@dataclass(frozen=True, slots=True)
class TextMutation:
    mutation_id: str
    relative_path: Path
    clean_text: str
    faulty_text: str

    def _replace_exactly_once(self, root: Path, old: str, new: str) -> None:
        path = root / self.relative_path
        content = path.read_text()
        count = content.count(old)
        if count != 1:
            raise ValueError(f"expected one mutation anchor in {path}, found {count}")
        path.write_text(content.replace(old, new, 1))

    def inject(self, root: Path) -> None:
        self._replace_exactly_once(root, self.clean_text, self.faulty_text)

    def repair(self, root: Path) -> None:
        self._replace_exactly_once(root, self.faulty_text, self.clean_text)


@dataclass(frozen=True, slots=True)
class TaskInstance:
    manifest: TaskManifest
    workspace: Path
    protected_hashes: dict[str, str]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OrderflowTaskGenerator:
    """Injects faults into a copy of the clean orderflow template."""

    DUPLICATE_CHARGE = TextMutation(
        mutation_id="duplicate_charge",
        relative_path=Path("orderflow/pipeline.py"),
        clean_text="idempotency_key = event.order_id",
        faulty_text='idempotency_key = f"{event.order_id}:{event.attempt}"',
    )

    def __init__(self, template_root: Path | None = None):
        self.template_root = (
            template_root or Path(__file__).parent / "worlds" / "orderflow" / "template"
        )

    def generate_duplicate_charge(self, destination: Path, *, seed: int = 0) -> TaskInstance:
        if destination.exists():
            raise FileExistsError(f"task destination already exists: {destination}")
        shutil.copytree(self.template_root, destination)
        self.DUPLICATE_CHARGE.inject(destination)

        protected_paths = ("tests/test_public_behavior.py",)
        protected_hashes = {path: sha256_file(destination / path) for path in protected_paths}
        manifest = TaskManifest(
            task_id=f"orderflow-duplicate-charge-{seed}",
            seed=seed,
            issue=(
                "Customers are occasionally charged twice when a worker crashes and an order "
                "is retried. Repair the pipeline without disabling retries or skipping payment "
                "processing. Existing public behavior must remain compatible."
            ),
            fault_ids=(self.DUPLICATE_CHARGE.mutation_id,),
            step_budget=60,
            protected_paths=protected_paths,
        )
        return TaskInstance(
            manifest=manifest,
            workspace=destination,
            protected_hashes=protected_hashes,
        )

    def apply_gold_solution(self, instance: TaskInstance) -> None:
        self.DUPLICATE_CHARGE.repair(instance.workspace)
