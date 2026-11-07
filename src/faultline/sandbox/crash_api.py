"""Runs the API under a crash test: the process dies just before its second order commit.

This sits outside the agent's code, so it cannot be moved or removed. Commits only count once an
order request has arrived, so database work at startup is left alone. Code that writes an order in
one transaction never reaches a second commit, and is not interrupted, which is correct: there is
no moment at which a crash could leave it half done.
"""

import os
import sqlite3
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

import uvicorn

_WRITES = ("INSERT", "UPDATE", "DELETE", "REPLACE")
_real_connect = sqlite3.connect
_armed = False
_commits = 0


def _connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
    connection: sqlite3.Connection = _real_connect(*args, **kwargs)

    def watch(statement: str) -> None:
        global _commits
        if not _armed:
            return
        text = statement.lstrip().upper()
        autocommit_write = text.startswith(_WRITES) and not connection.in_transaction
        if text.startswith(("COMMIT", "END")) or autocommit_write:
            _commits += 1
            if _commits >= 2:
                os._exit(137)

    connection.set_trace_callback(watch)
    return connection


sqlite3.connect = _connect  # type: ignore[assignment]

from orderflow.api import app  # noqa: E402  # must import after sqlite3.connect is wrapped


async def crash_test_app(
    scope: MutableMapping[str, Any],
    receive: Callable[[], Awaitable[Any]],
    send: Callable[[Any], Awaitable[None]],
) -> None:
    global _armed
    if scope["type"] == "http" and scope["method"] == "POST" and scope["path"] == "/orders":
        _armed = True
    await app(scope, receive, send)


uvicorn.run(crash_test_app, host="0.0.0.0", port=8000, log_level="warning")
