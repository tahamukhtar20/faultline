from faultline.models import RewardMode, VerificationResult
from faultline.rewards import RewardEngine


def result(
    *,
    public: float = 0,
    invariants: float = 0,
    performance: float = 0,
    integrity: bool = True,
) -> VerificationResult:
    return VerificationResult(
        public_behavior=public,
        hidden_invariants=invariants,
        performance=performance,
        integrity_ok=integrity,
    )


def test_shaped_reward_only_pays_for_verified_progress() -> None:
    engine = RewardEngine(action_cost=0)
    first = engine.step(result(public=1))
    repeated = engine.step(result(public=1))

    assert first.total == 0.10
    assert repeated.total == 0


def test_regression_removes_previous_progress() -> None:
    engine = RewardEngine(action_cost=0)
    engine.step(result(public=1, invariants=0.8))

    regression = engine.step(result(public=0, invariants=0.4))

    assert regression.progress < 0


def test_sparse_mode_only_pays_terminal_success() -> None:
    engine = RewardEngine(mode=RewardMode.SPARSE, action_cost=0)
    complete = result(public=1, invariants=1, performance=1)

    before_submission = engine.step(complete)
    submission = engine.step(complete, submitted=True)
    repeated_submission = engine.step(complete, submitted=True)

    assert before_submission.total == 0
    assert submission.total == 0.40
    assert repeated_submission.total == 0


def test_integrity_violation_dominates_partial_progress() -> None:
    engine = RewardEngine(action_cost=0)

    hacked = engine.step(result(public=1, invariants=1, integrity=False), submitted=True)

    assert hacked.total == -1
