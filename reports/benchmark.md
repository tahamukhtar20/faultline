# Benchmark results

75 episodes. Regenerate with `python -m faultline.benchmark`.

## Pass rate by level

| Policy | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
| --- | --- | --- | --- | --- | --- |
| claude-code-claude-haiku-5-5 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| claude-code-claude-opus-5-5 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| gold | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| random | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| scripted | 0/3 | 1/3 | 1/3 | 0/3 | 0/3 |

## Reward components, trajectory length and time

| Policy | Level | Reward | Steps | Seconds | Public | Hidden | Resilience | Perf. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claude-code-claude-haiku-5-5 | 1 | +0.514 | 32.0 | 515 | 1.00 | 1.00 | 1.00 | 1.00 |
| claude-code-claude-haiku-5-5 | 2 | +0.445 | 30.3 | 425 | 1.00 | 1.00 | 1.00 | 1.00 |
| claude-code-claude-haiku-5-5 | 3 | +0.561 | 35.0 | 643 | 1.00 | 1.00 | 1.00 | 1.00 |
| claude-code-claude-haiku-5-5 | 4 | +0.464 | 36.0 | 738 | 1.00 | 1.00 | 1.00 | 1.00 |
| claude-code-claude-haiku-5-5 | 5 | +0.548 | 31.0 | 542 | 1.00 | 1.00 | 1.00 | 1.00 |
| claude-code-claude-opus-5-5 | 1 | +0.497 | 49.0 | 937 | 1.00 | 1.00 | 1.00 | 1.00 |
| claude-code-claude-opus-5-5 | 2 | +0.430 | 45.0 | 887 | 1.00 | 1.00 | 1.00 | 1.00 |
| claude-code-claude-opus-5-5 | 3 | +0.551 | 44.3 | 890 | 1.00 | 1.00 | 1.00 | 1.00 |
| claude-code-claude-opus-5-5 | 4 | +0.462 | 38.3 | 765 | 1.00 | 1.00 | 1.00 | 1.00 |
| claude-code-claude-opus-5-5 | 5 | +0.544 | 34.7 | 680 | 1.00 | 1.00 | 1.00 | 1.00 |
| gold | 1 | +0.543 | 2.7 | 66 | 1.00 | 1.00 | 1.00 | 1.00 |
| gold | 2 | +0.473 | 2.0 | 31 | 1.00 | 1.00 | 1.00 | 1.00 |
| gold | 3 | +0.591 | 4.3 | 133 | 1.00 | 1.00 | 1.00 | 1.00 |
| gold | 4 | +0.496 | 4.3 | 142 | 1.00 | 1.00 | 1.00 | 1.00 |
| gold | 5 | +0.574 | 5.3 | 208 | 1.00 | 1.00 | 1.00 | 1.00 |
| random | 1 | -0.681 | 14.3 | 82 | 0.00 | 0.25 | 0.33 | 0.33 |
| random | 2 | -0.677 | 10.0 | 52 | 0.33 | 0.25 | 0.33 | 0.33 |
| random | 3 | -0.681 | 14.3 | 89 | 0.00 | 0.33 | 0.17 | 0.33 |
| random | 4 | -0.677 | 10.0 | 56 | 0.33 | 0.17 | 0.33 | 0.33 |
| random | 5 | -0.677 | 10.0 | 59 | 0.33 | 0.17 | 0.17 | 0.33 |
| scripted | 1 | -0.012 | 12.0 | 8 | 0.00 | 0.92 | 0.83 | 1.00 |
| scripted | 2 | +0.154 | 12.3 | 6 | 1.00 | 0.83 | 1.00 | 1.00 |
| scripted | 3 | +0.257 | 13.7 | 45 | 0.67 | 1.00 | 0.83 | 1.00 |
| scripted | 4 | +0.021 | 12.7 | 26 | 1.00 | 0.83 | 0.83 | 1.00 |
| scripted | 5 | +0.074 | 13.3 | 59 | 1.00 | 0.83 | 0.67 | 1.00 |

## Failure categories

| Policy | Level | Failures |
| --- | --- | --- |
| claude-code-claude-haiku-5-5 | 1 | none |
| claude-code-claude-haiku-5-5 | 2 | none |
| claude-code-claude-haiku-5-5 | 3 | none |
| claude-code-claude-haiku-5-5 | 4 | none |
| claude-code-claude-haiku-5-5 | 5 | none |
| claude-code-claude-opus-5-5 | 1 | none |
| claude-code-claude-opus-5-5 | 2 | none |
| claude-code-claude-opus-5-5 | 3 | none |
| claude-code-claude-opus-5-5 | 4 | none |
| claude-code-claude-opus-5-5 | 5 | none |
| gold | 1 | none |
| gold | 2 | none |
| gold | 3 | none |
| gold | 4 | none |
| gold | 5 | none |
| random | 1 | public_tests (3), integrity_violation (2), legacy_event_failed (1) |
| random | 2 | integrity_violation (2), public_tests (2), legacy_event_failed (1) |
| random | 3 | public_tests (3), integrity_violation (2), blocked_queue (1), double_fulfilment (1) |
| random | 4 | integrity_violation (2), public_tests (2), duplicate_charge (1), phantom_charge (1) |
| random | 5 | integrity_violation (2), public_tests (2), duplicate_charge (1), phantom_charge (1) |
| scripted | 1 | public_tests (3), blocked_queue (1), double_fulfilment (1), legacy_event_failed (1) |
| scripted | 2 | phantom_charge (1), legacy_event_failed (1) |
| scripted | 3 | blocked_queue (1), public_tests (1), poison_not_failed (1) |
| scripted | 4 | blocked_queue (2), order_not_completed (1), phantom_charge (1) |
| scripted | 5 | blocked_queue (2), double_fulfilment (2), lost_event (1), phantom_charge (1) |
