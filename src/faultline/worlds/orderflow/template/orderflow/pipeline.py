"""A small durable order pipeline."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


class SimulatedWorkerCrash(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OrderEvent:
    order_id: str
    attempt: int = 0


class OrderPipeline:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    amount_cents INTEGER NOT NULL,
                    status TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS charges (
                    idempotency_key TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL,
                    amount_cents INTEGER NOT NULL
                );
                """
            )

    def create_order(self, order_id: str, amount_cents: int) -> None:
        if amount_cents <= 0:
            raise ValueError("amount_cents must be positive")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO orders(order_id, amount_cents, status) VALUES (?, ?, 'pending')",
                (order_id, amount_cents),
            )

    def process(self, event: OrderEvent, *, crash_after_charge: bool = False) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT amount_cents FROM orders WHERE order_id = ?",
                (event.order_id,),
            ).fetchone()
            if row is None:
                raise KeyError(event.order_id)
            amount_cents = int(row[0])

        # A payment is an external side effect and commits independently of worker acknowledgement.
        idempotency_key = event.order_id
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO charges"
                "(idempotency_key, order_id, amount_cents) VALUES (?, ?, ?)",
                (idempotency_key, event.order_id, amount_cents),
            )

        if crash_after_charge:
            raise SimulatedWorkerCrash("worker stopped before acknowledging the event")

        with self._connect() as connection:
            connection.execute(
                "UPDATE orders SET status = 'fulfilled' WHERE order_id = ?",
                (event.order_id,),
            )

    def charge_count(self, order_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM charges WHERE order_id = ?",
                (order_id,),
            ).fetchone()
        assert row is not None
        return int(row[0])

    def total_charged(self, order_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(amount_cents), 0) FROM charges WHERE order_id = ?",
                (order_id,),
            ).fetchone()
        assert row is not None
        return int(row[0])

    def order_status(self, order_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM orders WHERE order_id = ?",
                (order_id,),
            ).fetchone()
        if row is None:
            raise KeyError(order_id)
        return str(row[0])
