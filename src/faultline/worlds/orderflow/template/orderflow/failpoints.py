"""Crash injection for resilience testing. Set ORDERFLOW_FAILPOINT to a point's name."""

import os


def failpoint(name: str) -> None:
    if os.environ.get("ORDERFLOW_FAILPOINT") == name:
        os._exit(137)
