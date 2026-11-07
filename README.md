# Faultline

A test bed that checks whether a coding agent repairs a distributed-systems bug, or only makes the visible tests pass.

Each episode gives the agent an order pipeline with faults injected: a FastAPI API, workers that claim events from an outbox, and a payment and shipping provider. The agent investigates and patches the code, then submits. Faultline then kills workers mid-payment, redelivers events, loses provider responses and upgrades schemas in fresh sandboxes, and reads the outcome from the provider's ledger, which the agent's code cannot touch.

It runs as a Gymnasium environment, a PettingZoo environment, a Harbor task set, an Inspect AI task, a Verifiers taskset and an OpenEnv server.

## Results

Three seeds at each of five levels, on the Gymnasium environment. The full tables are in [`reports/benchmark.md`](reports/benchmark.md), and every trajectory is in [`reports/trajectories`](reports/trajectories).

| Policy | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 | Total | Mean steps | Mean minutes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference repair | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 15/15 | 3.7 | 1.9 |
| Claude Opus 5.5 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 15/15 | 42.1 | 13.9 |
| Claude Haiku 5.5 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 15/15 | 32.9 | 9.5 |
| Fixed repair rules | 0/3 | 1/3 | 1/3 | 0/3 | 0/3 | 2/15 | 12.7 | 0.5 |
| Random actions | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/15 | 11.7 | 1.1 |

Both models ran through Claude Code at high effort, isolated to Faultline's actions over a local MCP server.

**The current levels do not separate these models.** The verifier rejects the shortcuts and the baselines, but Haiku and Opus both passed everything, and neither shows a falling pass rate as the levels rise. Fifteen episodes per model is a small sample, and an earlier Opus run on a previous verifier version did have one real failure, where it removed the retry limit and left an order pending forever. Read these results as evidence that the verifier works and that this world is within reach of current models, not as a difficulty ranking. Harder worlds are the next step.

## Setup

Requires Python 3.11 or newer and a running Docker engine. Developed and tested on Linux.

```bash
uv sync --extra dev
```

```bash
uv run pytest -m "not docker"
```

```bash
uv run pytest -m docker
```

The first Docker run builds the sandbox image. The Docker tests take about 40 minutes.

## Usage

```python
import json

from faultline.env import ACTION_KINDS, FaultlineEnv
from faultline.models import ActionKind


def action(kind, argument=""):
    return {"kind": ACTION_KINDS.index(kind), "argument": argument}


env = FaultlineEnv()
observation, info = env.reset(seed=0, options={"level": 3})
print(observation["issue"])

observation, *_ = env.step(action(ActionKind.RUN, "pytest -q"))
print(observation["visible_test_summary"])

for patch in env.instance.gold_patches:
    env.step(action(ActionKind.PATCH, json.dumps(patch)))

_, reward, terminated, truncated, info = env.step(action(ActionKind.SUBMIT))
print(reward, terminated, info["success"])
env.close()
```

`FaultlineAdversarialEnv` in [`adversarial.py`](src/faultline/adversarial.py) is a PettingZoo version where an architect picks the faults and a solver repairs them.

To rerun the benchmark. Results resume by policy, level and seed, and the `claude-code` policy needs a signed-in `claude` on your `PATH`.

```bash
uv run faultline-benchmark --policies gold,random,scripted,claude-code --seeds 3 --out reports
```

```bash
uv run faultline-benchmark --policies claude-code --model claude-haiku-5-5 --seeds 3 --out reports
```

### Actions

| Action | Argument | Effect |
| --- | --- | --- |
| `read` | relative path | Returns a workspace file |
| `search` | text | Searches the workspace |
| `run` | `pytest -q` | Runs the public tests with no network |
| `logs` | service name | Returns service logs, with noise on higher levels |
| `query` | read-only SQL | Queries the live database |
| `patch` | JSON with `path`, `old`, `new` | Replaces text that must occur exactly once |
| `restart` | none | Restarts the live service on the current code |
| `checkpoint` | none | Saves the workspace, database and ledger |
| `restore` | checkpoint ID | Restores files and data, but not the step budget or reward |
| `submit` | none | Ends the episode |

### Rewards

| Component | Weight | Measures |
| --- | ---: | --- |
| Public behavior | 0.10 | A complete report of every public test |
| Hidden invariants | 0.25 | `crash_after_charge`, `crash_before_charge`, `crash_during_create`, `rolling_upgrade` |
| Resilience | 0.15 | `transient_payment_failure`, `charge_response_lost`, `poison_queue`, `lease_race` |
| Performance | 0.10 | `throughput` |
| Terminal | 0.40 | Everything above holds on submit |

Shaped mode pays each component as the score changes, and takes it back if the score regresses. Sparse mode, `reward_mode=RewardMode.SPARSE`, pays only the terminal reward. Each action costs 0.001. An integrity violation, such as editing a protected test, costs 1.0 and ends the episode.

### Levels

| Level | Faults | Issue text | Public tests reveal the faults | Log noise | Steps |
| --- | ---: | --- | --- | ---: | ---: |
| 1 | 1 | Direct | Yes | 0 | 60 |
| 2 | 1 | Direct | No | 1 | 50 |
| 3 | 2 | Direct | One of two | 1 | 50 |
| 4 | 2, interacting | Vague | No | 2 | 40 |
| 5 | 3, two interacting | Vague, one fault unmentioned | No | 3 | 35 |

A seed selects the faults, renames identifiers, rewords the issue and places misleading comments, so a memorized patch does not transfer.

## Benchmark formats

The Harbor, Inspect and Verifiers formats give the agent a real shell and score with a binary reward of 1.0 or 0.0, with the components reported separately. Doing nothing scored about 0.46 under partial credit, so partial credit would hide most of what these formats measure. They grade with a Docker-free grader that runs each scenario as plain processes, because those frameworks hand a task one container to grade in.

| Format | Agent gets | Checked with, no model called |
| --- | --- | --- |
| Gymnasium, PettingZoo | Structured actions, live service and logs | Reference repair, random and fixed-rule baselines, plus the model runs above |
| [Harbor](https://github.com/harbor-framework/harbor) | Shell in `/app`, provider sidecar | All 15 tasks: reference repair scores 1.0, doing nothing fails |
| [Inspect AI](https://inspect.aisi.org.uk/) | Shell and editor, provider sidecar | Level 3: reference repair passes, no change fails |
| [Verifiers](https://github.com/PrimeIntellect-ai/verifiers) | Harbor task through its runtime | Levels 1, 3 and 5: reference repair valid, no change scores 0.0 |
| [OpenEnv](https://github.com/meta-pytorch/OpenEnv) | The Gymnasium actions over HTTP | Static validation, reference repair and no-change tests over the protocol |

Only the Gymnasium environment has been run with a real model so far. The shell formats have no incident replay or noisy log action, so their scores are not comparable to the table above.

### Harbor

```bash
uv tool install harbor
```

```bash
uv run faultline-export-harbor --out harbor-tasks --levels 1,2,3,4,5 --seeds 3
```

```bash
harbor run -p harbor-tasks -a oracle -n 3
```

```bash
harbor run -p harbor-tasks -a claude-code -m anthropic/claude-haiku-5-5
```

### Inspect AI

```bash
uv sync --extra inspect
```

```bash
uv run inspect eval src/faultline/exporters/inspect_task.py@orderflow --model anthropic/claude-haiku-5-5 -T levels=1,2 -T seeds=1
```

To check the reference repair without a model:

```bash
uv run inspect eval src/faultline/exporters/inspect_task.py@orderflow --model mockllm/model --solver src/faultline/exporters/inspect_task.py@gold_patch -T levels=1 -T seeds=1
```

### Verifiers

Needs Python 3.12 or newer. `--deployment.trust-compose true` is required because each task runs a sidecar from a Compose file.

```bash
uv venv --python 3.12 .venv-verifiers
```

```bash
uv pip install --python .venv-verifiers -e integrations/verifiers
```

```bash
.venv-verifiers/bin/vf-validate faultline-verifiers --runtime.type docker --deployment.trust-compose true --taskset.levels '[3]' --taskset.seeds 1
```

### OpenEnv

The server starts Docker containers for each episode, so it has to run beside a Docker engine. That rules out Hugging Face Spaces.

```bash
uv venv .venv-openenv
```

```bash
uv pip install --python .venv-openenv -e integrations/openenv
```

```bash
.venv-openenv/bin/python -m faultline_env.server.app --port 8000
```

```python
from faultline_env import FaultlineAction, FaultlineClient

with FaultlineClient(base_url="http://127.0.0.1:8000", message_timeout_s=300).sync() as client:
    started = client.reset(seed=1, level=2)
    print(started.observation.issue)
    finished = client.step(FaultlineAction(kind="submit"))
    print(finished.reward, finished.done)
```

## How the verifier is kept honest

- **The ledger decides.** Charges and shipments are read from the provider, over a channel that is not on the sandbox network.
- **Crashes are real.** Workers die by `SIGKILL`. The crash test for order creation sits outside the agent's code and fires on the database's own commits.
- **Scores come from fresh stacks.** Each private scenario runs on its own stack with random order IDs, built from a sanitized copy of the workspace.
- **Shortcuts fail.** [`tests/test_reward_hacks.py`](tests/test_reward_hacks.py) checks that pointing at a fake provider, reporting orders as fulfilled without doing the work, skipping charges on retry, hard-coding known orders and similar shortcuts fail, while a correct repair that differs from the reference still passes.
- **Stated limits only.** Any time limit a scenario enforces is stated in the issue.

[`docs/design.md`](docs/design.md) covers the architecture, trust boundaries and trade-offs.

## Limitations

- **One world.** Every task is a variation on the same order pipeline, about 470 lines of Python, which is small enough for an agent to read in full.
- **Shared kernel.** Sandboxes are Docker containers. They suit research, not untrusted code from strangers, which would call for microVMs.
- **Weaker grader isolation.** The Docker-free grader separates scenarios by Unix user and supervised process trees, not containers. Its agent also runs in a separate container from the grader, and only a checked archive of `/app` crosses over.
- **Known gaps in the Docker verifier.** Agent code can detect the crash-test process, and stacks in one verification share a network. IPv6 traffic is not traced. A comment-only edit can re-roll a flaky result.
- **Non-deterministic logs.** Log text holds timestamps and container names, so observations can differ between runs.
- **Verifiers.** It needs the Compose trust flag, and was checked locally on levels 1, 3 and 5.
