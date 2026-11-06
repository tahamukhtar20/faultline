"""The admission gate over the whole curriculum: every generated task is broken and solvable."""

import pytest

from faultline.taskgen import LEVELS, OrderflowTaskGenerator
from faultline.verifiers import validate_instance

pytestmark = pytest.mark.docker


@pytest.mark.parametrize("level", sorted(LEVELS))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_generated_task_passes_the_gate(level: int, seed: int) -> None:
    instance = OrderflowTaskGenerator().generate(level=level, seed=seed)

    assert validate_instance(instance) == ""
