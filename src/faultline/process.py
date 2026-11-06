"""Runs commands that execute agent-patched code, with a hard time limit."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
from collections.abc import Mapping
from pathlib import Path


def run_bounded(
    command: list[str],
    *,
    cwd: Path,
    timeout: float,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command in its own process group and kill the whole group when it ends.

    Raises subprocess.TimeoutExpired if the command outlives the timeout. Bytecode caching is off
    so a quick patch of the same size can never be shadowed by a stale .pyc file.
    """
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(env or {})},
        text=True,
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(process.pid)
        process.communicate(timeout=1)
        raise
    _kill_group(process.pid)
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _kill_group(process_group: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process_group, signal.SIGKILL)
