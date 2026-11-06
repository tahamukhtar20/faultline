"""Docker sandboxes for agent-written code.

Every container runs as a non-root user with a read-only root filesystem, no capabilities, and
memory, CPU and process limits. Containers that need to talk share a network with no gateway:
they can reach each other by name, but not the host or the internet.
"""

from __future__ import annotations

import hashlib
import json
import queue
import secrets
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from faultline.sandbox.archive import ArchiveError, Tree, build_tar, read_tar

BUILD_CONTEXT = Path(__file__).parent
_BUILD_FILES = ("Dockerfile", "provider.py", "toolbox.py", "crash_api.py")
_LABEL = "faultline.sandbox"
_VOLUME_SIZE = "128m"
_HARDENING = (
    "--read-only",
    "--tmpfs=/tmp:size=64m,mode=1777",
    "--cap-drop=ALL",
    "--security-opt=no-new-privileges",
    "--user=10001:10001",
    "--init",
    "--pids-limit=128",
    "--memory=512m",
    "--memory-swap=512m",
    "--cpus=1",
    "--workdir=/workspace",
    "--log-driver=local",
    "--log-opt=max-size=5m",
    "--log-opt=max-file=1",
    "--log-opt=compress=false",
)
MAX_COMMAND_OUTPUT_BYTES = 1024 * 1024


class DockerError(RuntimeError):
    pass


def docker(
    *args: str, stdin: bytes | None = None, timeout: float = 60, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            ["docker", *args], input=stdin, capture_output=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as error:
        raise DockerError(f"docker {args[0]} timed out after {timeout:g}s") from error
    if check and completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()
        raise DockerError(f"docker {' '.join(args[:2])} failed: {detail}")
    return completed


def image_tag() -> str:
    digest = hashlib.sha256()
    for name in _BUILD_FILES:
        digest.update((BUILD_CONTEXT / name).read_bytes())
    return f"faultline-sandbox:{digest.hexdigest()[:12]}"


def ensure_image() -> str:
    """Build the sandbox image if this version of the build files has not been built yet."""
    tag = image_tag()
    if docker("image", "inspect", tag, check=False).returncode != 0:
        docker("build", "--quiet", "--tag", tag, str(BUILD_CONTEXT), timeout=900)
    return tag


def remove_sandboxes(sandbox_id: str | None = None) -> None:
    """Remove the resources of one sandbox, or of every sandbox left over from crashed runs."""
    selector = f"label={_LABEL}" + (f"={sandbox_id}" if sandbox_id else "")
    containers = docker("ps", "-aq", "--filter", selector, check=False).stdout.split()
    if containers:
        docker("rm", "-f", "-v", *[c.decode() for c in containers], check=False, timeout=120)
    networks = docker("network", "ls", "-q", "--filter", selector, check=False).stdout.split()
    if networks:
        docker("network", "rm", *[n.decode() for n in networks], check=False)
    volumes = docker("volume", "ls", "-q", "--filter", selector, check=False).stdout.split()
    if volumes:
        docker("volume", "rm", "-f", *[v.decode() for v in volumes], check=False)


def capped(command: list[str]) -> list[str]:
    """Wrap a command that runs agent code, so its output cannot fill the host's memory."""
    limit = str(MAX_COMMAND_OUTPUT_BYTES)
    script = f'set -o pipefail; "$@" 2>&1 | head --bytes={limit}'
    return ["bash", "-c", script, "bash", *command]


@dataclass(frozen=True, slots=True)
class ExecResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class Container:
    name: str

    def run(
        self,
        command: list[str],
        *,
        stdin: bytes | None = None,
        timeout: float = 30,
        user: str | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        """Run a command in the container, killing it inside the container on timeout."""
        user_args = ("--user", user) if user else ()
        stdin_args = ("--interactive",) if stdin is not None else ()
        bounded = ["timeout", "--signal=KILL", f"{timeout:g}", *command]
        return docker(
            "exec",
            *stdin_args,
            *user_args,
            self.name,
            *bounded,
            stdin=stdin,
            timeout=timeout + 15,
            check=False,
        )

    def text(self, command: list[str], **kwargs: Any) -> ExecResult:
        completed = self.run(command, **kwargs)
        return ExecResult(
            completed.returncode,
            completed.stdout.decode(errors="replace"),
            completed.stderr.decode(errors="replace"),
        )

    def is_running(self) -> bool:
        completed = docker(
            "inspect", "--format", "{{.State.Running}}", self.name, check=False, timeout=20
        )
        return completed.stdout.strip() == b"true"

    def wait_exit(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_running():
                return True
            time.sleep(0.05)
        return False

    def kill(self) -> None:
        docker("kill", "--signal=KILL", self.name, check=False, timeout=20)

    def logs(self, tail: int = 500) -> str:
        completed = docker("logs", "--tail", str(tail), self.name, check=False, timeout=20)
        return (completed.stdout + completed.stderr).decode(errors="replace")

    def remove(self) -> None:
        docker("rm", "-f", "-v", self.name, check=False, timeout=30)


@dataclass(slots=True)
class Sandbox:
    """A group of containers, volumes and one network that are created and removed together."""

    image: str
    sandbox_id: str = field(default_factory=lambda: secrets.token_hex(6))
    _network: str | None = None
    _network_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def _labels(self) -> tuple[str, ...]:
        return (f"--label={_LABEL}={self.sandbox_id}",)

    @property
    def network(self) -> str:
        with self._network_lock:
            if self._network is None:
                name = f"fl-{self.sandbox_id}"
                docker(
                    "network",
                    "create",
                    "--internal",
                    "--opt=com.docker.network.bridge.inhibit_ipv4=true",
                    *self._labels,
                    name,
                )
                self._network = name
            return self._network

    def name(self, suffix: str) -> str:
        return f"fl-{self.sandbox_id}-{suffix}"

    def volume(self, suffix: str) -> str:
        """A size-limited, memory-backed volume, so agent code cannot fill the host's disk.

        A tmpfs volume is emptied when nothing has it mounted, so an idle container holds it open
        for as long as the sandbox lives, whatever happens to the containers that use it.
        """
        name = self.name(suffix)
        docker(
            "volume",
            "create",
            "--driver=local",
            "--opt=type=tmpfs",
            "--opt=device=tmpfs",
            f"--opt=o=size={_VOLUME_SIZE},uid=10001,gid=10001,mode=0755",
            *self._labels,
            name,
        )
        self.start(
            f"{suffix}-holder", ["sleep", "infinity"], mounts={"/data": name}, networked=False
        )
        return name

    def write_volume(self, volume: str, tree: Tree) -> None:
        """Replace a volume's contents from a throwaway container that runs no agent code."""
        script = "find /data -mindepth 1 -delete && tar --extract --no-same-owner --directory=/data"
        self._throwaway(volume, ["sh", "-c", script], stdin=build_tar(tree))

    def read_volume(self, volume: str) -> Tree:
        """Read a volume from a throwaway container, unaffected by anything agent code is doing."""
        try:
            archive = self._throwaway(f"{volume}:ro", ["tar", "--create", "--directory=/data", "."])
        except DockerError as error:
            raise ArchiveError(f"volume cannot be read: {error}") from error
        return read_tar(archive)

    def _throwaway(self, source: str, command: list[str], stdin: bytes | None = None) -> bytes:
        args = self.container_args(
            f"io-{secrets.token_hex(4)}", mounts={"/data": source}, networked=False
        )
        interactive = ("--interactive",) if stdin is not None else ()
        completed = docker("run", "--rm", *interactive, *args, self.image, *command, stdin=stdin)
        return completed.stdout

    def container_args(
        self,
        suffix: str,
        *,
        mounts: dict[str, str],
        env: dict[str, str] | None = None,
        networked: bool = True,
    ) -> list[str]:
        network = self.network if networked else "none"
        return [
            f"--name={self.name(suffix)}",
            f"--network={network}",
            *_HARDENING,
            *self._labels,
            *[_volume_flag(target, source) for target, source in mounts.items()],
            *[f"--env={key}={value}" for key, value in (env or {}).items()],
        ]

    def start(
        self,
        suffix: str,
        command: list[str],
        *,
        mounts: dict[str, str],
        env: dict[str, str] | None = None,
        networked: bool = True,
    ) -> Container:
        args = self.container_args(suffix, mounts=mounts, env=env, networked=networked)
        docker("run", "--detach", *args, self.image, *command)
        return Container(self.name(suffix))

    def close(self) -> None:
        remove_sandboxes(self.sandbox_id)


def _volume_flag(target: str, source: str) -> str:
    """Mounts are written as {target: "volume"} or {target: "volume:ro"}."""
    volume, _, mode = source.partition(":")
    return f"--volume={volume}:{target}" + (f":{mode}" if mode else "")


def cleanup_command() -> None:
    """Remove sandboxes left behind by crashed runs. Do not run this while episodes are active."""
    remove_sandboxes()


class ControlChannelError(RuntimeError):
    pass


class Provider:
    """The trusted provider sidecar, controlled only through its stdin and stdout."""

    def __init__(self, sandbox: Sandbox, suffix: str):
        self.host = sandbox.name(suffix)
        args = sandbox.container_args(suffix, mounts={})
        self._process = subprocess.Popen(
            [
                "docker",
                "run",
                "--interactive",
                *args,
                sandbox.image,
                "python",
                "-I",
                "/opt/faultline/provider.py",
            ],
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
        if self._process.stdin is not None:
            self._process.stdin.close()
        self._process.terminate()
