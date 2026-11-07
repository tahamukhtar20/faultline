"""Grade a workspace and write the result. Run as root in a disposable container.

Exits non-zero, writing no reward, if the grader itself failed: that says nothing about the agent.
"""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import sys
from pathlib import Path, PurePosixPath

from faultline.grader.stack import (
    ProcessStack,
    clear_scratch,
    kill_user,
    next_user,
    prepare_data_directory,
    run_to_completion,
)
from faultline.integrity import public_score, violations
from faultline.models import ScenarioResult, VerificationResult
from faultline.rewards import RewardEngine
from faultline.sandbox.archive import MAX_TREE_BYTES, MAX_TREE_FILES, ArchiveError
from faultline.scenarios import HIDDEN, PERFORMANCE, RESILIENCE, SCENARIOS, run_scenario

_PUBLIC_TIMEOUT_SECONDS = 60
_REPORT = "public-report.xml"
_RUN_DIR = Path("/var/faultline")
_IGNORED_PARTS = {"__pycache__", ".pytest_cache"}
_MAX_USER_PROCESSES = 512
_MAX_FILE_BYTES = 256 * 1024 * 1024


def read_workspace(root: Path) -> dict[str, bytes]:
    """Read a submitted tree. Links and device files are refused rather than followed."""
    files: dict[str, bytes] = {}
    total = 0
    if not root.is_dir():
        raise ArchiveError(f"workspace is missing: {root}")
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if _IGNORED_PARTS.intersection(PurePosixPath(name).parts):
            continue
        if path.is_symlink():
            raise ArchiveError(f"workspace contains a link: {name!r}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ArchiveError(f"workspace member is not a regular file: {name!r}")
        total += path.stat().st_size
        if total > MAX_TREE_BYTES or len(files) >= MAX_TREE_FILES:
            raise ArchiveError("workspace is too large")
        files[name] = path.read_bytes()
    return files


def install_workspace(files: dict[str, bytes]) -> Path:
    """Copy the tree somewhere the services can read but never change."""
    workspace = _RUN_DIR / "workspace"
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True)
    _RUN_DIR.chmod(0o755)
    workspace.chmod(0o755)
    for name, data in files.items():
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(0o644)
    for directory in workspace.rglob("*"):
        if directory.is_dir():
            directory.chmod(0o755)
    return workspace


def run_public_tests(workspace: Path, expected: int) -> float:
    uid = next_user()
    prepare_data_directory(uid)
    report = Path("/data") / _REPORT
    completed = run_to_completion(
        ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={report}", "tests"],
        workspace=workspace,
        uid=uid,
        timeout=_PUBLIC_TIMEOUT_SECONDS,
    )
    kill_user(uid)
    text = report.read_text(errors="replace") if report.is_file() else ""
    clear_scratch()
    return public_score(completed.returncode, text, expected)


def run_scenarios(workspace: Path) -> list[ScenarioResult]:
    """One at a time: the stacks share a loopback interface and fixed ports."""
    results = []
    for scenario in SCENARIOS:
        stack = ProcessStack(workspace)
        try:
            results.append(run_scenario(scenario, stack))
        finally:
            stack.stop_services()
            stack.close()
    return results


def _fraction(results: list[ScenarioResult], component: str) -> float:
    selected = [result for result in results if result.component == component]
    return sum(result.passed for result in selected) / len(selected)


def failed_integrity(reasons: list[str]) -> VerificationResult:
    return VerificationResult(
        public_behavior=0,
        hidden_invariants=0,
        resilience=0,
        performance=0,
        integrity_ok=False,
        critical_invariants_ok=False,
        diagnostics=tuple(reasons),
    )


def grade(root: Path, manifest: dict[str, object]) -> VerificationResult:
    protected = manifest["protected_hashes"]
    expected = manifest["public_tests"]
    assert isinstance(protected, dict) and isinstance(expected, int)
    try:
        files = read_workspace(root)
    except (ArchiveError, OSError) as error:
        return failed_integrity([str(error)])
    found = violations(protected, files)
    if found:
        return failed_integrity(found)

    workspace = install_workspace(files)
    public = run_public_tests(workspace, expected)
    results = run_scenarios(workspace)
    broken = [item for item in results if item.category == "infrastructure"]
    if broken:
        raise RuntimeError("; ".join(f"{item.name}: {item.detail}" for item in broken))
    return VerificationResult(
        public_behavior=public,
        hidden_invariants=_fraction(results, HIDDEN),
        resilience=_fraction(results, RESILIENCE),
        performance=_fraction(results, PERFORMANCE),
        integrity_ok=True,
        critical_invariants_ok=not any(item.critical for item in results if not item.passed),
        scenarios=tuple(results),
    )


def reward(result: VerificationResult) -> float:
    """Full marks need every check to pass. Partial credit stays below the terminal bonus."""
    terminal = RewardEngine.TERMINAL_WEIGHT if result.success else 0.0
    return round(RewardEngine.potential(result) + terminal, 6)


def write_results(result: VerificationResult, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    (directory / "details.json").write_text(
        json.dumps(
            {
                "success": result.success,
                "failure_categories": result.failure_categories,
                **result.model_dump(mode="json"),
            },
            indent=2,
        )
        + "\n"
    )
    (directory / "reward.json").write_text(
        json.dumps(
            {
                "reward": reward(result),
                "public_behavior": result.public_behavior,
                "hidden_invariants": result.hidden_invariants,
                "resilience": result.resilience,
                "performance": result.performance,
            }
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/app"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # Limits inherited by every service. Root, which runs the grader, is exempt from the first.
    resource.setrlimit(resource.RLIMIT_NPROC, (_MAX_USER_PROCESSES, _MAX_USER_PROCESSES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (_MAX_FILE_BYTES, _MAX_FILE_BYTES))
    args.out.mkdir(parents=True, exist_ok=True)
    args.out.chmod(0o700)

    result = grade(args.workspace, json.loads(args.manifest.read_text()))
    write_results(result, args.out)
    print(f"reward {reward(result)}, failures: {list(result.failure_categories) or 'none'}")
    for scenario in result.scenarios:
        print(f"  {scenario.name}: {'pass' if scenario.passed else scenario.category}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
