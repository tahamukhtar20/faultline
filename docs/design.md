# Design

## The question Faultline answers

A coding agent can often make tests pass without fixing anything. Faultline measures the difference. It asks whether the service works when a worker dies mid-payment, when an old event format arrives, or when the provider is slow, and it takes the answer from a record the agent's code cannot touch.

## Architecture

```text
                     host (trusted)
  +--------------------------------------------------+
  |  FaultlineEnv  --  RewardEngine  --  Verifier    |
  +-----+-----------------------------------+--------+
        | docker exec              docker run | fresh containers
        v                                     v
  +-----------+  +--------------------+   +--------------------+
  | solver    |  | live service       |   | scenario stacks    |
  | no network|  | api, worker,       |   | api, workers,      |
  | runs tests|  | provider           |   | provider, per test |
  +-----+-----+  +---------+----------+   +---------+----------+
        |                  |                        |
        +-- workspace volume (read-only to services)
                                        sanitized copy of the workspace
```

An episode has two copies of the service. The live copy gives the agent real logs and a real database to query, but it never affects the score. Every score comes from fresh containers, built from a sanitized copy of the workspace, so nothing the agent's code left running can influence the result.

## Trust boundaries

| Component | Runs agent code | Network | Trusted for results |
| --- | --- | --- | --- |
| Solver container | Yes, through tests | None | No |
| API and worker containers | Yes | Sandbox network only | No |
| Provider sidecar | No | Sandbox network only | Yes: its ledger is the source of truth |
| Toolbox | No, and started with `python -I` so workspace files cannot shadow it | Inherits solver | Only for tool output |
| Host | No | Normal | Yes |

Every container is non-root, with a read-only root filesystem, no capabilities, `no-new-privileges`, and memory, CPU and process limits. Writable storage is a size-limited tmpfs volume, so agent code cannot fill the host's disk.

The sandbox network is internal and its bridge has no IP address on the host. Docker's `--internal` flag alone is not enough: containers can still reach services on the host through the gateway. Removing the bridge address closes that path, but it also means the host cannot reach the containers over the network. So the host talks to the provider sidecar through the sidecar's stdin and stdout, and the sidecar forwards client requests to the API. That channel carries all ledger reads and fault injection, and because it is not on the network, the agent's code has no way to reach it.

Anything read back from a sandbox is treated as hostile. Archives containing links, device files or paths that leave the root are rejected, and a planted symlink ends the episode as an integrity violation.

Docker containers share the host's kernel. This is a strong research sandbox, not a boundary for hosting untrusted code from strangers, which would call for microVMs.

## The world

The service is an order pipeline built the way real ones often are: a FastAPI API writes orders and events in one transaction, and workers claim events from the outbox under a lease. A worker charges through the provider with an idempotency key and then ships the order. Shipping has no idempotency key, so a duplicate call sends a second parcel.

## Faults

Each fault is an exact text edit of the clean world, so it can be injected and reversed precisely.

| Family | Variants | Symptom the verifier measures |
| --- | --- | --- |
| `duplicate_charge` | attempt, worker or clock in the payment key | Two captures after a crash and redelivery |
| `lost_outbox_event` | order committed first, or event committed first | An order nothing will process, or a charge for an order that does not exist |
| `poison_retry` | strict FIFO, or no backoff, with unbounded retries | Healthy orders stuck behind a rejected one |
| `schema_skew` | previous format dropped, or its amount misread | Old orders fail, or are charged the wrong amount |
| `timeout_race` | lease too short, or ignored | Two workers process an order and ship it twice |

Some pairs interact. A dropped event format turns old orders into poison events, for example, so repairing one fault can expose the other.

## Generation

A level and a seed fully determine a task. The seed chooses the faults and variants, renames identifiers across the code, words the issue, and places misleading comments, so memorized patches do not transfer between instances.

Difficulty is computed from the task, not assigned by hand:

```text
score = faults + interactions + log noise + public/private test gap + delayed effects + budget pressure
```

| Level | Faults | Issue text | Public tests reveal the faults | Log noise | Steps |
| --- | ---: | --- | --- | ---: | ---: |
| 1 | 1 | Direct | Yes | 0 | 60 |
| 2 | 1 | Direct | No | 1 | 50 |
| 3 | 2 | Direct | One of two | 1 | 50 |
| 4 | 2, interacting | Vague | No | 2 | 40 |
| 5 | 3, two interacting | Vague, one fault unmentioned | No | 3 | 35 |

A task is valid only if the injected code fails verification and the gold patches pass it. `validate_instance` is this gate, and it runs in the test suite across the curriculum.

## Scoring

Nine private scenarios run in parallel, each on its own stack with random order IDs and amounts.

| Component | Scenarios |
| --- | --- |
| Hidden invariants | `crash_after_charge`, `crash_before_charge`, `crash_during_create`, `rolling_upgrade` |
| Resilience | `transient_payment_failure`, `charge_response_lost`, `poison_queue`, `lease_race` |
| Performance | `throughput` |

Duplicate charges, charges for missing orders, wrong amounts and duplicate shipments are critical: no other score can make up for them. Crashes are real `SIGKILL`s of containers, so agent code cannot intercept them. The order-creation crash test sits outside the agent's code and fires on the database's own commit statements, so it cannot be moved. The public test score needs a complete report of every expected test, not only a clean exit code.

Any time limit a scenario enforces is stated in the issue: orders held by a crashed worker recover within 15 seconds, and an order whose payment keeps failing is marked failed within 30. The crash test also starts counting commits only once an order request arrives, so database work at startup is not mistaken for a half-written order.

Shaped rewards are the change in a potential over these scores. Any path that returns to the same state therefore nets to zero, which is why repairing, restoring and repairing again cannot farm reward.

## Other benchmark formats

Harbor, Inspect AI and Verifiers run an agent in one container and grade in another, with no Docker socket for starting scenario stacks. [`faultline.grader`](../src/faultline/grader) runs the same nine scenarios there using plain processes:

- Scenarios run one at a time on fixed ports, each under its own unprivileged user, so one scenario's leftovers cannot reach the next.
- A root [supervisor](../src/faultline/sandbox/supervisor.py) wraps each service. It is a child subreaper, so it can find and kill everything the service spawned, the way a container's init would. Killing it is the grader's equivalent of `SIGKILL`ing a container.
- `/tests` and the results directory are closed to the scenario users, and process and file-size limits apply.
- The agent's work arrives as a tar of `/app`. Links and anything else that is not a plain file are rejected, and if the grader itself fails, it exits with an error and writes no reward, so a broken run cannot read as a score.

This is weaker isolation than the Docker verifier, and the provider sidecar the agent talks to is a separate development copy. Its ledger never reaches the grader, which starts its own.

These formats score 1.0 or 0.0. Each fault breaks only some scenarios, so partial credit would pay about 0.46 for submitting the broken code unchanged. [`exporters/harbor.py`](../src/faultline/exporters/harbor.py) writes the task directory that the Inspect and Verifiers integrations also build on, and it pins the Python image by digest and service dependencies from the sandbox's Dockerfile. OpenEnv is different: [`integrations/openenv`](../integrations/openenv) wraps the Gymnasium environment directly, so it keeps structured actions, shaped rewards and the Docker verifier.

## Checkpoints

A checkpoint is a versioned JSON manifest plus content-addressed blobs, covering the workspace, the database, the provider's ledger, and the controller state. Its ID is the hash of the manifest, and every blob is verified on load, so tampering or corruption fails closed. Branches share the blobs they have in common.

There are two kinds of restore, for two kinds of caller:

- **The agent's `restore` action** moves files and data only. The step budget keeps counting and shaped reward falls with the state, so rewinding offers no advantage.
- **The experimenter's `env.restore()` and `env.branch()`** rewind the whole episode, including budgets and reward state, so that runs can fork from any point.

## Trade-offs

- **Real containers over simulation.** Failures are realistic, but verification takes seconds, so results are cached by workspace hash and only recomputed when the code changes.
- **Structured tools over a shell.** Actions are auditable and easy to bound, though less like an unrestricted coding agent.
- **Observations are deterministic.** Wall-clock time is left out of them, and is available as `env.remaining_seconds`. Log text can still differ between runs because it holds timestamps and container names.
- **One world, in depth.** A single service with interacting faults supports stronger verifiers than many shallow repositories. More worlds are the natural next step.
