"""The order API."""

import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from orderflow import db, outbox
from orderflow.events import encode_event


class OrderRequest(BaseModel):
    order_id: str = Field(min_length=1, max_length=64)
    amount_cents: int = Field(gt=0)
    currency: str = Field(default="USD", min_length=3, max_length=3)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    db.initialize()
    yield


app = FastAPI(title="orderflow", lifespan=lifespan)


def insert_order(connection: sqlite3.Connection, order: OrderRequest) -> None:
    connection.execute(
        "INSERT INTO orders(order_id, amount_cents, currency, status) VALUES (?, ?, ?, 'pending')",
        (order.order_id, order.amount_cents, order.currency),
    )


def insert_event(connection: sqlite3.Connection, order: OrderRequest) -> None:
    outbox.enqueue(connection, order.order_id, encode_event(order.amount_cents, order.currency))


@app.post("/orders", status_code=201)
def create_order(order: OrderRequest) -> dict[str, str]:
    try:
        with db.transaction() as connection:
            insert_order(connection, order)
            insert_event(connection, order)
    except sqlite3.IntegrityError as error:
        raise HTTPException(status_code=409, detail="order already exists") from error
    return {"order_id": order.order_id, "status": "pending"}


@app.get("/orders/{order_id}")
def read_order(order_id: str) -> dict[str, object]:
    connection = db.connect()
    try:
        row = connection.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    finally:
        connection.close()
    if row is None:
        raise HTTPException(status_code=404, detail="order not found")
    return dict(row)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
