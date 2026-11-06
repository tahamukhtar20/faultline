"""The fault catalogue. Each variant is an exact, reversible edit of the clean world."""

from __future__ import annotations

from dataclasses import dataclass

WORKER = "orderflow/worker.py"
OUTBOX = "orderflow/outbox.py"
API = "orderflow/api.py"
EVENTS = "orderflow/events.py"
CONFIG = "orderflow/config.py"


@dataclass(frozen=True, slots=True)
class TextEdit:
    path: str
    clean: str
    faulty: str


@dataclass(frozen=True, slots=True)
class FaultVariant:
    variant_id: str
    family: str
    edits: tuple[TextEdit, ...]


@dataclass(frozen=True, slots=True)
class FaultFamily:
    family: str
    incident: str
    delayed_effect: bool
    direct_issues: tuple[str, ...]
    vague_issues: tuple[str, ...]
    variants: tuple[FaultVariant, ...]


_CLEAN_KEY = "idempotency_key = claim.order_id"
_COMMIT_TOGETHER = """            insert_order(connection, order)
            insert_event(connection, order)
        failpoint("after_order_commit")
"""
_DEAD_ON_REJECT = TextEdit(
    WORKER,
    "outbox.mark_dead(claim, str(error))",
    "outbox.retry(claim, str(error))",
)
_BOUNDED_RETRY = TextEdit(
    OUTBOX,
    """    if claim.attempts >= config.MAX_ATTEMPTS:
        mark_dead(claim, error)
        return
""",
    "",
)
_CLAIM_RESPECTS_LEASE = """            "SELECT * FROM outbox WHERE status = 'pending' AND lease_until <= ? "
            "ORDER BY event_id LIMIT 1",
            (now,),
        ).fetchone()
        if row is None:
            return None
"""  # noqa: E501
_V1_BRANCH = """    if payload.get("schema_version", 1) == 1:
        return OrderCreated(amount_cents=payload["amount_cents"], currency="USD")
"""


def _family(
    family: str,
    *,
    incident: str,
    delayed_effect: bool,
    direct: tuple[str, ...],
    vague: tuple[str, ...],
    variants: dict[str, tuple[TextEdit, ...]],
) -> FaultFamily:
    return FaultFamily(
        family=family,
        incident=incident,
        delayed_effect=delayed_effect,
        direct_issues=direct,
        vague_issues=vague,
        variants=tuple(
            FaultVariant(f"{family}/{name}", family, edits) for name, edits in variants.items()
        ),
    )


FAMILIES: dict[str, FaultFamily] = {
    family.family: family
    for family in (
        _family(
            "duplicate_charge",
            incident="crash_after_charge",
            delayed_effect=True,
            direct=(
                "Customers are charged twice when a worker crashes after charging and the order "
                "event is redelivered. Repair this without disabling retries.",
                "A worker that dies between the payment and the acknowledgement causes a second "
                "payment when the event is picked up again. Make redelivery safe.",
            ),
            vague=(
                "Finance found customers with two payments for one order, always around worker "
                "restarts.",
                "Support reports double payments that we can't reproduce on a healthy system.",
            ),
            variants={
                "attempt_in_key": (
                    TextEdit(
                        WORKER,
                        _CLEAN_KEY,
                        'idempotency_key = f"{claim.order_id}:{claim.attempts}"',
                    ),
                ),
                "owner_in_key": (
                    TextEdit(
                        WORKER, _CLEAN_KEY, 'idempotency_key = f"{claim.order_id}:{claim.owner}"'
                    ),
                ),
                "clock_in_key": (
                    TextEdit(
                        WORKER, _CLEAN_KEY, 'idempotency_key = f"{claim.order_id}:{time.time_ns()}"'
                    ),
                ),
            },
        ),
        _family(
            "lost_outbox_event",
            incident="crash_on_commit",
            delayed_effect=True,
            direct=(
                "If the API crashes while creating an order, the order row and its outbox event "
                "can disagree. They must commit atomically.",
                "A crash during order creation leaves orders and events inconsistent. Creating "
                "an order must be all or nothing.",
            ),
            vague=(
                "After last night's API restarts, a few orders look wrong and nobody can say why.",
                "Reconciliation is off by a handful of orders on days when the API restarts.",
            ),
            variants={
                "order_first": (
                    TextEdit(
                        API,
                        _COMMIT_TOGETHER,
                        """            insert_order(connection, order)
        failpoint("after_order_commit")
        with db.transaction() as connection:
            insert_event(connection, order)
""",
                    ),
                ),
                "event_first": (
                    TextEdit(
                        API,
                        _COMMIT_TOGETHER,
                        """            insert_event(connection, order)
        failpoint("after_order_commit")
        with db.transaction() as connection:
            insert_order(connection, order)
""",
                    ),
                ),
            },
        ),
        _family(
            "poison_retry",
            incident="poison_queue",
            delayed_effect=False,
            direct=(
                "One event the payment provider permanently rejects stops every order behind it. "
                "Rejected events must be failed, and later orders must keep moving.",
                "A single undeliverable event blocks the queue. Handle it without giving up on "
                "temporary provider errors.",
            ),
            vague=(
                "Orders sometimes stop moving entirely until someone intervenes.",
                "The queue stalls every few days and throughput drops to zero.",
            ),
            variants={
                "strict_fifo": (
                    _DEAD_ON_REJECT,
                    _BOUNDED_RETRY,
                    TextEdit(
                        OUTBOX,
                        _CLAIM_RESPECTS_LEASE,
                        """            "SELECT * FROM outbox WHERE status = 'pending' "
            "ORDER BY event_id LIMIT 1",
        ).fetchone()
        if row is None or row["lease_until"] > now:
            return None
""",
                    ),
                ),
                "no_backoff": (
                    _DEAD_ON_REJECT,
                    TextEdit(
                        OUTBOX,
                        "delay = config.RETRY_DELAY_SECONDS * 2**claim.attempts",
                        "delay = 0.0",
                    ),
                    _BOUNDED_RETRY,
                ),
            },
        ),
        _family(
            "schema_skew",
            incident="rolling_upgrade",
            delayed_effect=False,
            direct=(
                "During the rolling upgrade, events written by the previous release are not "
                "handled correctly. Both event versions must work.",
                "Orders written by the old release misbehave after the upgrade. Keep "
                "compatibility with the previous event format.",
            ),
            vague=(
                "Some orders have gone wrong since we started the upgrade.",
                "A portion of orders has looked odd ever since the new release began rolling out.",
            ),
            variants={
                "v1_dropped": (TextEdit(EVENTS, _V1_BRANCH, ""),),
                "v1_wrong_units": (
                    TextEdit(
                        EVENTS,
                        'amount_cents=payload["amount_cents"], currency="USD"',
                        'amount_cents=payload["amount_cents"] * 100, currency="USD"',
                    ),
                ),
            },
        ),
        _family(
            "timeout_race",
            incident="lease_race",
            delayed_effect=True,
            direct=(
                "When the provider is slow and two workers are running, an order can be processed "
                "by both, and parcels are shipped twice.",
                "Slow provider responses let a second worker take an event that is still being "
                "processed. Each order must ship once.",
            ),
            vague=(
                "The warehouse occasionally gets two shipping requests for the same order.",
                "Duplicate parcels spike whenever the provider has a slow day.",
            ),
            variants={
                "short_lease": (TextEdit(CONFIG, "LEASE_SECONDS = 3.0", "LEASE_SECONDS = 0.5"),),
                "lease_ignored": (
                    TextEdit(
                        OUTBOX,
                        _CLAIM_RESPECTS_LEASE,
                        """            "SELECT * FROM outbox WHERE status = 'pending' "
            "ORDER BY event_id LIMIT 1",
        ).fetchone()
        if row is None:
            return None
""",
                    ),
                ),
            },
        ),
    )
}

VARIANTS: dict[str, FaultVariant] = {
    variant.variant_id: variant for family in FAMILIES.values() for variant in family.variants
}

# Fault pairs whose symptoms overlap, so that fixing one exposes or masks the other.
INTERACTING_PAIRS: tuple[tuple[str, str], ...] = (
    ("poison_retry", "schema_skew"),
    ("duplicate_charge", "timeout_race"),
    ("lost_outbox_event", "duplicate_charge"),
    ("poison_retry", "timeout_race"),
)
