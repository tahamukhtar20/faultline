"""The transactional outbox: at-least-once delivery of order events to workers."""

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

from orderflow import config, db


@dataclass(frozen=True)
class Claim:
    event_id: int
    order_id: str
    payload: dict[str, Any]
    attempts: int
    owner: str


def enqueue(connection: sqlite3.Connection, order_id: str, payload: dict[str, Any]) -> None:
    connection.execute(
        "INSERT INTO outbox(order_id, payload) VALUES (?, ?)", (order_id, json.dumps(payload))
    )


def claim_next(owner: str) -> Claim | None:
    now = time.time()
    with db.transaction() as connection:
        row = connection.execute(
            "SELECT * FROM outbox WHERE status = 'pending' AND lease_until <= ? "
            "ORDER BY event_id LIMIT 1",
            (now,),
        ).fetchone()
        if row is None:
            return None
        attempts = row["attempts"] + 1
        connection.execute(
            "UPDATE outbox SET attempts = ?, lease_until = ?, lease_owner = ? WHERE event_id = ?",
            (attempts, now + config.LEASE_SECONDS, owner, row["event_id"]),
        )
    return Claim(
        event_id=row["event_id"],
        order_id=row["order_id"],
        payload=json.loads(row["payload"]),
        attempts=attempts,
        owner=owner,
    )


def complete(claim: Claim) -> None:
    with db.transaction() as connection:
        connection.execute(
            "UPDATE outbox SET status = 'done' WHERE event_id = ?", (claim.event_id,)
        )
        connection.execute(
            "UPDATE orders SET status = 'fulfilled' WHERE order_id = ?", (claim.order_id,)
        )


def mark_dead(claim: Claim, error: str) -> None:
    with db.transaction() as connection:
        connection.execute(
            "UPDATE outbox SET status = 'dead', last_error = ? WHERE event_id = ?",
            (error, claim.event_id),
        )
        connection.execute(
            "UPDATE orders SET status = 'failed' WHERE order_id = ?", (claim.order_id,)
        )


def retry(claim: Claim, error: str) -> None:
    if claim.attempts >= config.MAX_ATTEMPTS:
        mark_dead(claim, error)
        return
    delay = config.RETRY_DELAY_SECONDS * 2**claim.attempts
    with db.transaction() as connection:
        connection.execute(
            "UPDATE outbox SET lease_until = ?, last_error = ? WHERE event_id = ?",
            (time.time() + delay, error, claim.event_id),
        )
