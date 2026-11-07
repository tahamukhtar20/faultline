"""The Inspect AI task, run with the reference repair and with no repair. No model is called."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai import eval as inspect_eval

from faultline.exporters.inspect_task import gold_patch, no_change, orderflow

pytestmark = pytest.mark.docker


def run(solver: Callable[[], Any], level: str) -> Any:
    logs = inspect_eval(
        orderflow(levels=level, seeds=1),
        model="mockllm/model",
        solver=solver(),
        display="none",
        log_dir=None,
        fail_on_error=False,
    )
    assert logs[0].status == "success", logs[0].error
    assert logs[0].samples is not None
    return logs[0].samples[0].scores["private_scenarios"]


def test_the_reference_repair_passes() -> None:
    score = run(gold_patch, "3")

    assert score.value == 1.0
    assert score.answer == "pass"


def test_doing_nothing_fails_with_categories() -> None:
    score = run(no_change, "3")

    assert score.value == 0.0
    assert score.metadata["failure_categories"]
