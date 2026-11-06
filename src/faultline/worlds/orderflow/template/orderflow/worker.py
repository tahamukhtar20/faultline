"""Delivers order events: charges the customer, then ships the order."""

import logging
import os
import socket
import time

from orderflow import config, db, outbox, providers
from orderflow.events import parse_event
from orderflow.providers import PermanentFailure

logger = logging.getLogger("orderflow.worker")


def order_is_fulfilled(order_id: str) -> bool:
    connection = db.connect()
    try:
        row = connection.execute(
            "SELECT status FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
    finally:
        connection.close()
    return row is not None and row["status"] == "fulfilled"


def handle(claim: outbox.Claim) -> None:
    if order_is_fulfilled(claim.order_id):
        outbox.complete(claim)
        return
    event = parse_event(claim.payload)
    idempotency_key = claim.order_id
    providers.charge(idempotency_key, claim.order_id, event.amount_cents, event.currency)
    providers.ship(claim.order_id)
    outbox.complete(claim)


def run_once(owner: str) -> bool:
    claim = outbox.claim_next(owner)
    if claim is None:
        return False
    logger.info(
        "event=%s order=%s attempt=%s claimed", claim.event_id, claim.order_id, claim.attempts
    )
    try:
        handle(claim)
    except PermanentFailure as error:
        logger.error("event=%s order=%s rejected: %s", claim.event_id, claim.order_id, error)
        outbox.mark_dead(claim, str(error))
    except Exception as error:
        logger.warning("event=%s order=%s failed: %r", claim.event_id, claim.order_id, error)
        outbox.retry(claim, repr(error))
    else:
        logger.info("event=%s order=%s done", claim.event_id, claim.order_id)
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    db.initialize()
    owner = f"{socket.gethostname()}-{os.getpid()}"
    logger.info("worker %s started", owner)
    while True:
        if not run_once(owner):
            time.sleep(config.POLL_SECONDS)


if __name__ == "__main__":
    main()
