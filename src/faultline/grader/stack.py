"""A running copy of the order service as plain processes in one disposable container.

The grader runs as root, and each scenario's services share one unprivileged user. A root
supervisor per service gives the guarantees separate containers would: agent code cannot signal
the provider, the grader or the supervisor, and a kill reaches every process a service started,
even one that detached itself. The provider is still controlled only through its stdin and stdout.
"""

from __future__ import annotations

import itertools
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from faultline.sandbox.errors import ControlChannelError
from faultline.service_stack import BaseStack

DATA_DIR = Path("/data")
TOOLS_DIR = Path("/opt/faultline")
_API_PORT = 8000
_PROVIDER_PORT = 8100
_API_COMMAND = [
    "python",
    "-m",
    "uvicorn",
    "orderflow.api:app",
    "--host",
    "0.0.0.0",
    "--port",
    str(_API_PORT),
    "--log-level",
    "warning",
]
_CRASH_TEST_API_COMMAND = ["python", str(TOOLS_DIR / "crash_api.py")]
_WORKER_COMMAND = ["python", "-m", "orderflow.worker"]
_SCRATCH_DIRS = (Path("/tmp"), Path("/var/tmp"), Path("/dev/shm"))
_KILL_ALL = "import os, signal; os.kill(-1, signal.SIGKILL)"
_user_ids = itertools.count(20_000)


def kill_user(uid: int) -> None:
    """Kill every process a user owns."""
    subprocess.run(
        [sys.executable, "-I", "-c", _KILL_ALL],
        cwd="/",
        user=uid,
        group=uid,
        extra_groups=[],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
        check=False,
    )


def service_environment(workspace: Path) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(DATA_DIR),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(workspace),
        "ORDERFLOW_DATA": str(DATA_DIR),
        "ORDERFLOW_PROVIDER_URL": f"http://127.0.0.1:{_PROVIDER_PORT}",
    }


@dataclass(frozen=True, slots=True)
class CompletedCommand:
    returncode: int
    stdout: str
    stderr: str


def _supervised(command: list[str], uid: int) -> list[str]:
    return ["python", "-I", str(TOOLS_DIR / "supervisor.py"), str(uid), *command]


def run_to_completion(
    command: list[str], *, workspace: Path, uid: int, timeout: float
) -> CompletedCommand:
    process = subprocess.Popen(
        _supervised(command, uid),
        cwd=workspace,
        env=service_environment(workspace),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.terminate()
        stdout, stderr = process.communicate()
        return CompletedCommand(124, stdout, stderr)
    return CompletedCommand(process.returncode, stdout, stderr)


class ProcessService:
    def __init__(self, command: list[str], *, workspace: Path, uid: int):
        self._workspace = workspace
        self._uid = uid
        self._process = subprocess.Popen(
            _supervised(command, uid),
            cwd=workspace,
            env=service_environment(workspace),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def is_running(self) -> bool:
        return self._process.poll() is None

    def wait_exit(self, timeout: float) -> bool:
        try:
            self._process.wait(timeout)
        except subprocess.TimeoutExpired:
            return False
        return True

    def text(self, command: list[str], *, timeout: float) -> CompletedCommand:
        return run_to_completion(command, workspace=self._workspace, uid=self._uid, timeout=timeout)

    def kill(self) -> None:
        self._process.terminate()
        self._process.wait()


class ProcessProvider:
    """The trusted provider, running as its own user and reachable from the grader by pipe only."""

    def __init__(self, uid: int, gid: int):
        self._process = subprocess.Popen(
            ["python", "-I", str(TOOLS_DIR / "provider.py")],
            cwd="/",
            user=uid,
            group=gid,
            extra_groups=[],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._lines: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        threading.Thread(target=self._read_lines, daemon=True).start()
        self._next_line(timeout=30)

    def _read_lines(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            self._lines.put(line)
        self._lines.put("")

    def _next_line(self, timeout: float) -> dict[str, Any]:
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            raise ControlChannelError("provider did not respond") from None
        if not line:
            raise ControlChannelError("provider exited")
        reply: dict[str, Any] = json.loads(line)
        return reply

    def call(self, operation: str, *, timeout: float = 15, **arguments: Any) -> dict[str, Any]:
        assert self._process.stdin is not None
        with self._lock:
            try:
                self._process.stdin.write(json.dumps({"op": operation, **arguments}) + "\n")
                self._process.stdin.flush()
            except OSError as error:
                raise ControlChannelError("provider exited") from error
            return self._next_line(timeout)

    def close(self) -> None:
        self._process.kill()
        self._process.wait()


def clear_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def next_user() -> int:
    return next(_user_ids)


def prepare_data_directory(uid: int) -> None:
    """Empty the data directory and give it to this run's user."""
    DATA_DIR.mkdir(exist_ok=True)
    clear_directory(DATA_DIR)
    os.chown(DATA_DIR, uid, uid)
    DATA_DIR.chmod(0o700)


def clear_scratch() -> None:
    """Nothing one scenario's code wrote may reach the next one."""
    for path in _SCRATCH_DIRS:
        clear_directory(path)


class ProcessStack(BaseStack):
    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._uid = next_user()
        prepare_data_directory(self._uid)
        provider_uid = next_user()
        self.provider: ProcessProvider = ProcessProvider(provider_uid, provider_uid)
        self.api: ProcessService | None = None
        self.workers: list[ProcessService] = []

    def _start(self, command: list[str]) -> ProcessService:
        return ProcessService(command, workspace=self._workspace, uid=self._uid)

    def _api_url(self, path: str) -> str:
        return f"http://127.0.0.1:{_API_PORT}{path}"

    def start_api(self, *, crash_test: bool = False) -> ProcessService:
        self.stop_api()
        self.api = self._start(_CRASH_TEST_API_COMMAND if crash_test else _API_COMMAND)
        return self.api

    def start_worker(self) -> ProcessService:
        worker = self._start(_WORKER_COMMAND)
        self.workers.append(worker)
        return worker

    def stop_api(self) -> None:
        if self.api is not None:
            self.api.kill()
            self._wait_for_port_to_close()
            self.api = None

    def _wait_for_port_to_close(self) -> None:
        """The next API must be able to bind, and a stale reply must not look like its own."""
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if "connection_error" in self.request("GET", "/health", timeout=0.5):
                return
            time.sleep(0.05)

    def kill_worker(self, worker: ProcessService) -> None:
        self.workers.remove(worker)
        worker.kill()

    def stop_services(self) -> None:
        self.stop_api()
        for worker in list(self.workers):
            self.kill_worker(worker)

    def close(self) -> None:
        kill_user(self._uid)
        self.provider.close()
        clear_scratch()
        clear_directory(DATA_DIR)
