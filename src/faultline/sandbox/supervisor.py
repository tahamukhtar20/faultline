"""Runs one service the way a container's init does, for graders with no container per service.

It stays root, so the service cannot signal it, and adopts every orphan, so a process cannot escape
by detaching itself. When the service ends, or this process gets SIGTERM, everything below it dies.
Started with `python -I` so nothing in the workspace can shadow it. Standard library only.

Usage: supervisor.py UID COMMAND...
"""

import contextlib
import ctypes
import os
import signal
import sys
import time
from collections import defaultdict

_PR_SET_CHILD_SUBREAPER = 36


def _descendants() -> list[int]:
    children: dict[int, list[int]] = defaultdict(list)
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as stat:
                fields = stat.read().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        children[int(fields[1])].append(int(entry))
    found = []
    pending = [os.getpid()]
    while pending:
        for child in children[pending.pop()]:
            found.append(child)
            pending.append(child)
    return found


def _reap() -> None:
    with contextlib.suppress(ChildProcessError):
        while os.waitpid(-1, os.WNOHANG)[0] != 0:
            pass


def _kill_everything() -> None:
    while True:
        found = _descendants()
        if not found:
            return
        for pid in found:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        _reap()
        time.sleep(0.005)


def _stop(_signal: int, _frame: object) -> None:
    _kill_everything()
    os._exit(137)


def main() -> None:
    uid = int(sys.argv[1])
    command = sys.argv[2:]
    ctypes.CDLL(None).prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    signal.signal(signal.SIGTERM, _stop)
    service = os.fork()
    if service == 0:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.setsid()
        os.setgroups([])
        os.setgid(uid)
        os.setuid(uid)
        os.umask(0o077)
        os.execvp(command[0], command)
    while True:
        pid, status = os.wait()
        if pid == service:
            break
    _kill_everything()
    sys.exit(os.waitstatus_to_exitcode(status) & 0xFF)


if __name__ == "__main__":
    main()
