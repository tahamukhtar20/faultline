"""The previous release's order writer, still running during the rolling upgrade.

It is frozen at that release, so it shares no code with the current one.
"""

import json
import sqlite3
import sys

from orderflow import config


def write_order(order_id: str, amount_cents: int) -> None:
    connection = sqlite3.connect(config.DATABASE_PATH, timeout=10)
    try:
        with connection:
            connection.execute(
                "INSERT INTO orders(order_id, amount_cents, currency, status) "
                "VALUES (?, ?, 'USD', 'pending')",
                (order_id, amount_cents),
            )
            connection.execute(
                "INSERT INTO outbox(order_id, payload) VALUES (?, ?)",
                (order_id, json.dumps({"schema_version": 1, "amount_cents": amount_cents})),
            )
    finally:
        connection.close()


if __name__ == "__main__":
    write_order(sys.argv[1], int(sys.argv[2]))
