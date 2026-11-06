"""Scores a workspace in fresh containers that share nothing with the agent's sandbox."""

from __future__ import annotations

import re
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath

from faultline.models import ScenarioResult, VerificationResult
from faultline.sandbox.archive import FileEntry, Tree, tree_hash
from faultline.sandbox.docker import Sandbox, capped, docker, ensure_image
from faultline.scenarios import HIDDEN, PERFORMANCE, RESILIENCE, SCENARIOS, run_scenario
from faultline.stack import ServiceStack
from faultline.taskgen import FORBIDDEN_NAMES, TaskInstance, sha256_bytes

PUBLIC_TIMEOUT_SECONDS = 60
_REPORT_MARKER = "=====faultline-report====="


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


def _public_tests(sandbox: Sandbox, volume: str, expected: int) -> tuple[float, str]:
    """Run the protected tests and require a complete report, not only a clean exit code.

    A patch can make the process exit early with status 0, so every expected test must be reported
    as having run and passed. The output is capped, because this runs agent code.
    """
    report = "/tmp/public-report.xml"
    script = (
        f"python -m pytest -q -p no:cacheprovider --junitxml={report} tests; "
        f'status=$?; echo "{_REPORT_MARKER}"; cat {report}; exit $status'
    )
    args = sandbox.container_args(
        f"public-{secrets.token_hex(4)}", mounts={"/workspace": f"{volume}:ro"}, networked=False
    )
    completed = docker(
        "run",
        "--rm",
        *args,
        sandbox.image,
        *capped(["timeout", "--signal=KILL", str(PUBLIC_TIMEOUT_SECONDS), "sh", "-c", script]),
        timeout=PUBLIC_TIMEOUT_SECONDS + 30,
        check=False,
    )
    output = completed.stdout.decode(errors="replace")
    summary, _, report_text = output.partition(_REPORT_MARKER)
    counts = {
        name: int(value)
        for name, value in re.findall(r'\b(tests|failures|errors|skipped)="(\d+)"', report_text)
    }
    passed = (
        completed.returncode == 0
        and counts.get("tests") == expected
        and not any(counts.get(name, 1) for name in ("failures", "errors", "skipped"))
    )
    return (1.0 if passed else 0.0, summary[-2_000:])


def _count_tests(instance: TaskInstance) -> int:
    return sum(
        len(re.findall(r"^def test_", instance.files[path].data.decode(), re.MULTILINE))
        + _parameter_count(instance.files[path].data.decode())
        for path in instance.protected_hashes
        if path.startswith("tests/test_")
    )


def _parameter_count(source: str) -> int:
    """Extra test cases from parametrize lists, which are the only parametrization used."""
    return sum(
        len(re.findall(r'"[^"]+"', values)) - 1
        for values in re.findall(r"parametrize\([^,]+, \[([^\]]*)\]", source)
    )


def _run_one(sandbox: Sandbox, volume: str, index: int) -> ScenarioResult:
    stack = ServiceStack(sandbox, volume)
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
            public = pool.submit(_public_tests, sandbox, volume, _count_tests(instance))
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
    # A time budget can be missed because the machine is busy, so that result is not kept.
    if "too_slow" not in result.failure_categories:
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
