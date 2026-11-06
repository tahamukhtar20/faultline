"""Runs structured tool actions against a task workspace."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from faultline.models import ActionKind, ToolAction
from faultline.process import run_bounded
from faultline.taskgen import TaskInstance

COMMAND_TIMEOUT_SECONDS = 30.0


class LocalWorkspaceRuntime:
    """Limits the agent to allowlisted actions. Not a sandbox: patched code runs on the host."""

    _MAX_OUTPUT = 8_000

    def __init__(self, instance: TaskInstance):
        self.instance = instance

    def _path(self, relative_path: str) -> Path:
        candidate = (self.instance.workspace / relative_path).resolve()
        root = self.instance.workspace.resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("path escapes the task workspace")
        return candidate

    def execute(self, action: ToolAction) -> str:
        try:
            result = self._execute(action)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            result = f"tool error: {error}"
        except subprocess.TimeoutExpired as error:
            result = f"tool error: command timed out after {error.timeout:g}s"
        return result[-self._MAX_OUTPUT :]

    def _execute(self, action: ToolAction) -> str:
        if action.kind is ActionKind.READ:
            path = self._path(action.argument)
            if not path.is_file():
                raise ValueError(f"not a file: {action.argument}")
            return path.read_text()

        if action.kind is ActionKind.SEARCH:
            needle = action.argument
            if not needle:
                raise ValueError("search text cannot be empty")
            matches: list[str] = []
            for path in sorted(self.instance.workspace.rglob("*.py")):
                for line_number, line in enumerate(path.read_text().splitlines(), start=1):
                    if needle in line:
                        relative = path.relative_to(self.instance.workspace)
                        matches.append(f"{relative}:{line_number}:{line}")
            return "\n".join(matches) if matches else "no matches"

        if action.kind is ActionKind.RUN:
            allowed = {"pytest -q", "python -m pytest -q"}
            if action.argument not in allowed:
                raise ValueError(f"command is not allowlisted: {action.argument}")
            completed = run_bounded(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=self.instance.workspace,
                timeout=COMMAND_TIMEOUT_SECONDS,
            )
            return completed.stdout + completed.stderr

        if action.kind is ActionKind.PATCH:
            payload = json.loads(action.argument)
            if not isinstance(payload, dict):
                raise ValueError("patch argument must be a JSON object")
            path_value = payload.get("path")
            old = payload.get("old")
            new = payload.get("new")
            if not (isinstance(path_value, str) and isinstance(old, str) and isinstance(new, str)):
                raise ValueError("patch requires string path, old, and new fields")
            path = self._path(path_value)
            content = path.read_text()
            count = content.count(old)
            if count != 1:
                raise ValueError(f"patch anchor must occur exactly once; found {count}")
            path.write_text(content.replace(old, new, 1))
            return f"patched {path_value}"

        if action.kind is ActionKind.SUBMIT:
            return "solution submitted for private verification"

        raise ValueError(f"unsupported action: {action.kind}")
