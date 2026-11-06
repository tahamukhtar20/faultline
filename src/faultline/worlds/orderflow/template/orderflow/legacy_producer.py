"""The previous release's order writer, still running during the rolling upgrade."""

import sys

from orderflow import db, outbox


def write_order(order_id: str, amount_cents: int) -> None:
    db.initialize()
    with db.transaction() as connection:
        connection.execute(
            "INSERT INTO orders(order_id, amount_cents, currency, status) "
            "VALUES (?, ?, 'USD', 'pending')",
            (order_id, amount_cents),
        )
        outbox.enqueue(connection, order_id, {"schema_version": 1, "amount_cents": amount_cents})


if __name__ == "__main__":
    write_order(sys.argv[1], int(sys.argv[2]))
