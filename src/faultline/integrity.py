"""Checks on a submitted workspace that need no sandbox, shared by every way of scoring it."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import PurePosixPath

FORBIDDEN_NAMES = frozenset(
    {"conftest.py", "pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini", "sitecustomize.py"}
)


def violations(protected_hashes: Mapping[str, str], files: Mapping[str, bytes]) -> list[str]:
    """Protected files must be untouched, and nothing may add test configuration."""
    found = [
        f"protected file changed: {path}"
        for path, expected in protected_hashes.items()
        if path not in files or hashlib.sha256(files[path]).hexdigest() != expected
    ]
    found.extend(
        f"forbidden file added: {path}"
        for path in files
        if path not in protected_hashes
        and (PurePosixPath(path).name in FORBIDDEN_NAMES or PurePosixPath(path).parts[0] == "tests")
    )
    return found


def public_score(returncode: int, report: str, expected_tests: int) -> float:
    """Every expected test must be reported as run and passed, not only a clean exit code.

    A patch can make the process exit early with status 0, so the exit code alone proves nothing.
    """
    counts = {
        name: int(value)
        for name, value in re.findall(r'\b(tests|failures|errors|skipped)="(\d+)"', report)
    }
    passed = (
        returncode == 0
        and counts.get("tests") == expected_tests
        and not any(counts.get(name, 1) for name in ("failures", "errors", "skipped"))
    )
    return 1.0 if passed else 0.0
