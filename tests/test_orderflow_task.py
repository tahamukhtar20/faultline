from faultline.taskgen import OrderflowTaskGenerator
from faultline.verifiers import verify_orderflow


def test_generated_fault_passes_public_tests_but_fails_private_invariant(tmp_path) -> None:
    generator = OrderflowTaskGenerator()
    instance = generator.generate_duplicate_charge(tmp_path / "task", seed=17)

    result = verify_orderflow(instance)

    assert result.public_behavior == 1
    assert result.integrity_ok
    assert result.hidden_invariants == 0
    assert not result.success


def test_gold_solution_restores_private_invariant(tmp_path) -> None:
    generator = OrderflowTaskGenerator()
    instance = generator.generate_duplicate_charge(tmp_path / "task", seed=17)

    generator.apply_gold_solution(instance)
    result = verify_orderflow(instance)

    assert result.success


def test_modifying_public_tests_is_an_integrity_failure(tmp_path) -> None:
    generator = OrderflowTaskGenerator()
    instance = generator.generate_duplicate_charge(tmp_path / "task")
    test_path = instance.workspace / "tests/test_public_behavior.py"
    test_path.write_text("def test_nothing():\n    assert True\n")

    result = verify_orderflow(instance)

    assert not result.integrity_ok
    assert not result.success
