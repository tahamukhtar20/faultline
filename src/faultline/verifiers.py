"""Scores a workspace in fresh containers that share nothing with the agent's sandbox."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath

from faultline.models import ScenarioResult, VerificationResult
from faultline.sandbox.archive import FileEntry, Tree, tree_hash
from faultline.sandbox.docker import Sandbox, docker, ensure_image
from faultline.scenarios import HIDDEN, PERFORMANCE, RESILIENCE, SCENARIOS, run_scenario
from faultline.stack import ServiceStack
from faultline.taskgen import FORBIDDEN_NAMES, TaskInstance, sha256_bytes

PUBLIC_TIMEOUT_SECONDS = 60


class VerificationError(RuntimeError):
    """The sandbox itself failed, so this run says nothing about the agent's code."""


_cache: dict[str, VerificationResult] = {}
_cache_lock = threading.Lock()


def integrity_violations(instance: TaskInstance, tree: Tree) -> list[str]:
    """Protected files must be untouched, and nothing may add test configuration."""
    violations = [
        f"protected file changed: {path}"
        for path, expected in instance.protected_hashes.items()
        if path not in tree or sha256_bytes(tree[path].data) != expected
    ]
    violations.extend(
        f"forbidden file added: {path}"
        for path in tree
        if path not in instance.protected_hashes
        and (PurePosixPath(path).name in FORBIDDEN_NAMES or PurePosixPath(path).parts[0] == "tests")
    )
    return violations


def _public_tests(sandbox: Sandbox, volume: str) -> tuple[float, str]:
    args = sandbox.container_args("public", mounts={"/workspace": f"{volume}:ro"}, networked=False)
    completed = docker(
        "run",
        "--rm",
        *args,
        sandbox.image,
        "timeout",
        "--signal=KILL",
        str(PUBLIC_TIMEOUT_SECONDS),
        "python",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests",
        timeout=PUBLIC_TIMEOUT_SECONDS + 30,
        check=False,
    )
    summary = (completed.stdout + completed.stderr).decode(errors="replace")[-2_000:]
    return (1.0 if completed.returncode == 0 else 0.0, summary)


def _run_one(sandbox: Sandbox, volume: str, index: int) -> ScenarioResult:
    stack = ServiceStack(sandbox, volume, f"s{index}")
    try:
        return run_scenario(SCENARIOS[index], stack)
    finally:
        stack.stop_services()
        stack.close()


def _fraction(results: list[ScenarioResult], component: str) -> float:
    selected = [result for result in results if result.component == component]
    return sum(result.passed for result in selected) / len(selected)


def verify_tree(instance: TaskInstance, tree: Tree) -> VerificationResult:
    violations = integrity_violations(instance, tree)
    if violations:
        return VerificationResult(
            public_behavior=0,
            hidden_invariants=0,
            resilience=0,
            performance=0,
            integrity_ok=False,
            critical_invariants_ok=False,
            diagnostics=tuple(violations),
        )

    key = tree_hash(tree)
    with _cache_lock:
        cached = _cache.get(key)
    if cached is not None:
        return cached

    sandbox = Sandbox(ensure_image())
    try:
        volume = sandbox.volume("workspace")
        sandbox.write_volume(volume, tree)
        with ThreadPoolExecutor(max_workers=len(SCENARIOS) + 1) as pool:
            public = pool.submit(_public_tests, sandbox, volume)
            futures = [
                pool.submit(_run_one, sandbox, volume, index) for index in range(len(SCENARIOS))
            ]
            results = [future.result() for future in futures]
            public_score, public_summary = public.result()
    finally:
        sandbox.close()

    broken = [item for item in results if item.category == "infrastructure"]
    if broken:
        raise VerificationError("; ".join(f"{item.name}: {item.detail}" for item in broken))

    result = VerificationResult(
        public_behavior=public_score,
        hidden_invariants=_fraction(results, HIDDEN),
        resilience=_fraction(results, RESILIENCE),
        performance=_fraction(results, PERFORMANCE),
        integrity_ok=True,
        critical_invariants_ok=not any(item.critical for item in results if not item.passed),
        scenarios=tuple(results),
        diagnostics=(f"public: {public_summary}",),
    )
    with _cache_lock:
        _cache[key] = result
    return result


def validate_instance(instance: TaskInstance) -> str:
    """The solvability gate: faults must break the world and the gold patches must repair it.

    Returns the reason a task is invalid, or an empty string if it may be used.
    """
    if verify_tree(instance, instance.files).success:
        return "the injected faults do not fail verification"
    repaired = dict(instance.files)
    for patch in instance.gold_patches:
        text = repaired[patch["path"]].data.decode()
        if text.count(patch["old"]) != 1:
            return f"a gold patch does not apply to {patch['path']}"
        entry = repaired[patch["path"]]
        repaired[patch["path"]] = FileEntry(
            text.replace(patch["old"], patch["new"]).encode(), entry.mode
        )
    result = verify_tree(instance, repaired)
    if not result.success:
        return f"the gold patches do not pass verification: {list(result.failure_categories)}"
    return ""


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()
