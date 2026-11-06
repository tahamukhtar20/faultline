import pytest
from conftest import gold_tree

from faultline.faults import FAMILIES, VARIANTS
from faultline.taskgen import LEVELS, OrderflowTaskGenerator, compatible, manifest_digest

SEEDS = range(4)


def test_same_seed_generates_the_same_task() -> None:
    generator = OrderflowTaskGenerator()
    for level in LEVELS:
        first = generator.generate(level=level, seed=3)
        second = generator.generate(level=level, seed=3)
        assert manifest_digest(first) == manifest_digest(second)


def test_seeds_produce_different_code() -> None:
    generator = OrderflowTaskGenerator()
    digests = {
        manifest_digest(generator.generate(level=level, seed=seed))
        for level in LEVELS
        for seed in SEEDS
    }
    workspaces = {
        tuple(sorted((path, entry.data) for path, entry in instance.files.items()))
        for instance in (
            generator.generate(level=level, seed=seed) for level in LEVELS for seed in SEEDS
        )
    }

    assert len(digests) == len(LEVELS) * len(SEEDS)
    assert len(workspaces) >= 12


@pytest.mark.parametrize("level", sorted(LEVELS))
@pytest.mark.parametrize("seed", SEEDS)
def test_gold_patches_apply_exactly_and_change_every_fault(level: int, seed: int) -> None:
    instance = OrderflowTaskGenerator().generate(level=level, seed=seed)

    repaired = gold_tree(instance)

    assert repaired != instance.files
    assert all(path in repaired for path in instance.protected_hashes)


def test_every_variant_builds_and_every_family_has_several() -> None:
    generator = OrderflowTaskGenerator()
    assert set(FAMILIES) == {
        "duplicate_charge",
        "lost_outbox_event",
        "poison_retry",
        "schema_skew",
        "timeout_race",
    }
    assert all(len(family.variants) >= 2 for family in FAMILIES.values())
    for variant_id in VARIANTS:
        gold_tree(generator.generate_variants([variant_id], seed=1))


def test_difficulty_grows_with_level() -> None:
    generator = OrderflowTaskGenerator()
    scores = [
        sum(generator.generate(level=level, seed=seed).manifest.difficulty.score for seed in SEEDS)
        for level in sorted(LEVELS)
    ]
    assert scores == sorted(scores)
    assert scores[0] < scores[-1]


def test_higher_levels_have_more_faults_and_smaller_budgets() -> None:
    assert LEVELS[1].fault_count < LEVELS[5].fault_count
    assert LEVELS[1].step_budget > LEVELS[5].step_budget


def test_conflicting_variants_are_detected() -> None:
    assert not compatible(
        [VARIANTS["poison_retry/strict_fifo"], VARIANTS["timeout_race/lease_ignored"]]
    )
    assert compatible(
        [VARIANTS["duplicate_charge/attempt_in_key"], VARIANTS["schema_skew/v1_dropped"]]
    )


def test_private_scenarios_never_enter_the_workspace() -> None:
    instance = OrderflowTaskGenerator().generate(level=1, seed=0)
    assert not any("scenario" in path or "verifier" in path for path in instance.files)
    assert not any(b"verifier" in entry.data for entry in instance.files.values())
