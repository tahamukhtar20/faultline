"""Seeded task generation: a level and a seed fully determine a task."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from faultline.faults import FAMILIES, INTERACTING_PAIRS, VARIANTS, FaultVariant, TextEdit
from faultline.models import Difficulty, TaskManifest
from faultline.sandbox.archive import FileEntry, Tree

WORLD_ROOT = Path(__file__).parent / "worlds" / "orderflow"
FORBIDDEN_NAMES = frozenset(
    {"conftest.py", "pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini", "sitecustomize.py"}
)
_ALWAYS_PROTECTED = ("orderflow/failpoints.py", "orderflow/legacy_producer.py")

# Seeded renames, so an agent cannot rely on memorized identifiers or patches.
_RENAMES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("outbox", ("outbox", "event_queue", "dispatch_log")),
    ("lease_until", ("lease_until", "claimed_until", "visible_after")),
    ("lease_owner", ("lease_owner", "claimed_by", "holder")),
    ("LEASE_SECONDS", ("LEASE_SECONDS", "CLAIM_TIMEOUT_SECONDS", "VISIBILITY_SECONDS")),
    ("idempotency_key", ("idempotency_key", "payment_key", "dedupe_key")),
)
_DISTRACTIONS = (
    (
        "orderflow/worker.py",
        "def handle(",
        "# NOTE: charge before ship, see the payments runbook.\n",
    ),
    ("orderflow/outbox.py", "def claim_next(", "# Claim order matters for fairness reviews.\n"),
    ("orderflow/api.py", "def read_order(", "# Reads are served from the primary for now.\n"),
    ("orderflow/events.py", "def encode_event(", "# Version 3 is being discussed for tax lines.\n"),
)


@dataclass(frozen=True, slots=True)
class LevelConfig:
    fault_count: int
    interacting: bool
    noise: int
    verifier_gap: int
    budget_pressure: int
    step_budget: int
    time_budget_seconds: int
    vague_issue: bool
    distractions: int
    visible_incidents: int


LEVELS: dict[int, LevelConfig] = {
    1: LevelConfig(1, False, 0, 0, 0, 60, 1_800, False, 0, 1),
    2: LevelConfig(1, False, 1, 1, 0, 50, 1_500, False, 0, 1),
    3: LevelConfig(2, False, 1, 1, 1, 50, 1_500, False, 1, 2),
    4: LevelConfig(2, True, 2, 2, 2, 40, 1_200, True, 2, 1),
    5: LevelConfig(3, True, 3, 3, 3, 35, 1_200, True, 4, 1),
}


@dataclass(frozen=True, slots=True)
class TaskInstance:
    manifest: TaskManifest
    files: Tree
    protected_hashes: dict[str, str]
    gold_patches: tuple[dict[str, str], ...]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compatible(variants: list[FaultVariant]) -> bool:
    """Whether the variants can be applied together: no edit may disturb another's text."""
    try:
        _apply_edits(_load_template(), variants)
    except ValueError:
        return False
    return True


def _load_template() -> dict[str, str]:
    root = WORLD_ROOT / "template"
    return {
        str(path.relative_to(root)): path.read_text()
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def _apply_edit(files: dict[str, str], edit: TextEdit) -> None:
    content = files[edit.path]
    if content.count(edit.clean) != 1:
        raise ValueError(f"edit anchor must occur once in {edit.path}: {edit.clean!r}")
    files[edit.path] = content.replace(edit.clean, edit.faulty)


def _apply_edits(files: dict[str, str], variants: list[FaultVariant]) -> None:
    applied: set[TextEdit] = set()
    for variant in variants:
        for edit in variant.edits:
            if edit not in applied:
                _apply_edit(files, edit)
                applied.add(edit)


def _rename(text: str, renames: dict[str, str]) -> str:
    for old, new in renames.items():
        text = text.replace(old, new)
    return text


def _choose_families(rng: random.Random, config: LevelConfig) -> list[str]:
    if config.interacting:
        chosen = list(rng.choice(INTERACTING_PAIRS))
        others = [family for family in FAMILIES if family not in chosen]
        rng.shuffle(others)
        chosen.extend(others[: config.fault_count - len(chosen)])
    else:
        chosen = rng.sample(sorted(FAMILIES), config.fault_count)
    return chosen


def _choose_variants(rng: random.Random, families: list[str]) -> list[FaultVariant]:
    for _ in range(100):
        variants = [rng.choice(FAMILIES[family].variants) for family in families]
        if compatible(variants):
            return variants
    raise ValueError(f"no compatible variants for {families}")


def _gold_patches(
    variants: list[FaultVariant], renames: dict[str, str]
) -> tuple[dict[str, str], ...]:
    patches: list[dict[str, str]] = []
    seen: set[TextEdit] = set()
    for variant in variants:
        for edit in variant.edits:
            if edit in seen:
                continue
            seen.add(edit)
            faulty = _rename(edit.faulty, renames)
            clean = _rename(edit.clean, renames)
            if faulty:
                patches.append({"path": _rename(edit.path, renames), "old": faulty, "new": clean})
            else:
                patches.append(_reinsertion_patch(edit, renames))
    return tuple(patches)


def _reinsertion_patch(edit: TextEdit, renames: dict[str, str]) -> dict[str, str]:
    """A fault that deleted code is repaired by anchoring on the line that followed it."""
    template = _load_template()[edit.path]
    following = template[template.index(edit.clean) + len(edit.clean) :].splitlines(keepends=True)[
        0
    ]
    following = _rename(following, renames)
    return {
        "path": _rename(edit.path, renames),
        "old": following,
        "new": _rename(edit.clean, renames) + following,
    }


def _build_issue(rng: random.Random, config: LevelConfig, families: list[str]) -> str:
    lines = [
        rng.choice(
            FAMILIES[family].vague_issues if config.vague_issue else FAMILIES[family].direct_issues
        )
        for family in families[: max(1, len(families) - config.verifier_gap // 2)]
    ]
    lines.append(
        "Do not disable payments, retries or shipping, and keep the public API compatible."
    )
    return " ".join(lines)


def _distract(files: dict[str, str], rng: random.Random, count: int) -> None:
    for path, anchor, comment in rng.sample(_DISTRACTIONS, count):
        files[path] = files[path].replace(anchor, comment + anchor, 1)


class OrderflowTaskGenerator:
    """Builds tasks from the clean orderflow world."""

    def generate(self, *, level: int = 1, seed: int = 0) -> TaskInstance:
        config = LEVELS[level]
        rng = random.Random(f"faultline:{level}:{seed}")
        families = _choose_families(rng, config)
        variants = _choose_variants(rng, families)
        return self.build(
            task_id=f"orderflow-l{level}-{seed}",
            level=level,
            seed=seed,
            config=config,
            rng=rng,
            variants=variants,
        )

    def generate_variants(
        self, variant_ids: list[str], *, level: int = 1, seed: int = 0, rename: bool = True
    ) -> TaskInstance:
        """Build a task with chosen faults at a level's settings, for tests and the architect."""
        variants = [VARIANTS[variant_id] for variant_id in variant_ids]
        rng = random.Random(f"faultline:{level}:{seed}:{','.join(variant_ids)}")
        digest = sha256_bytes(",".join(variant_ids).encode())[:8]
        return self.build(
            task_id=f"orderflow-custom-{digest}-l{level}-{seed}",
            level=level,
            seed=seed,
            config=LEVELS[level],
            rng=rng,
            variants=variants,
            rename=rename,
        )

    def build(
        self,
        *,
        task_id: str,
        level: int,
        seed: int,
        config: LevelConfig,
        rng: random.Random,
        variants: list[FaultVariant],
        rename: bool = True,
    ) -> TaskInstance:
        families = [variant.family for variant in variants]
        renames = {old: rng.choice(choices) for old, choices in _RENAMES} if rename else {}

        texts = _load_template()
        _apply_edits(texts, variants)
        _distract(texts, rng, config.distractions)
        for family in self._regression_families(families, config):
            source = WORLD_ROOT / "regressions" / f"test_{family}.py"
            texts[f"tests/test_{family}.py"] = source.read_text()
        files: Tree = {
            _rename(path, renames): FileEntry(_rename(text, renames).encode())
            for path, text in texts.items()
        }

        protected = sorted(
            path for path in files if path.startswith("tests/") or path in _ALWAYS_PROTECTED
        )
        issue = _build_issue(rng, config, families)
        visible = tuple(FAMILIES[family].incident for family in families)[
            : config.visible_incidents
        ]
        manifest = TaskManifest(
            task_id=task_id,
            seed=seed,
            issue=issue,
            fault_ids=tuple(families),
            variant_ids=tuple(variant.variant_id for variant in variants),
            difficulty=Difficulty(
                level=level,
                fault_count=len(families),
                interaction_depth=self._interaction_depth(families),
                observation_noise=config.noise,
                verifier_gap=config.verifier_gap,
                delayed_effects=sum(FAMILIES[family].delayed_effect for family in families),
                budget_pressure=config.budget_pressure,
            ),
            step_budget=config.step_budget,
            time_budget_seconds=config.time_budget_seconds,
            visible_incidents=visible,
        )
        return TaskInstance(
            manifest=manifest,
            files=files,
            protected_hashes={path: sha256_bytes(files[path].data) for path in protected},
            gold_patches=_gold_patches(variants, renames),
        )

    @staticmethod
    def _regression_families(families: list[str], config: LevelConfig) -> list[str]:
        if config.verifier_gap == 0:
            return families
        if config.verifier_gap == 1 and len(families) > 1:
            return families[:1]
        return []

    @staticmethod
    def _interaction_depth(families: list[str]) -> int:
        chosen = set(families)
        return sum(1 for pair in INTERACTING_PAIRS if chosen.issuperset(pair))


def manifest_digest(instance: TaskInstance) -> str:
    """A fingerprint of a generated task, used to prove that generation is deterministic."""
    payload = {
        "manifest": instance.manifest.model_dump(mode="json"),
        "files": {path: sha256_bytes(entry.data) for path, entry in sorted(instance.files.items())},
        "gold": list(instance.gold_patches),
    }
    return sha256_bytes(json.dumps(payload, sort_keys=True).encode())
