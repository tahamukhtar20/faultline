"""Runs baselines over seeded tasks and writes a report with measured results."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from faultline.baselines import POLICIES, EpisodeResult, Policy, run_episode
from faultline.env import FaultlineEnv
from faultline.taskgen import LEVELS
from faultline.verifiers import clear_cache

COMPONENTS = ("public_behavior", "hidden_invariants", "resilience", "performance")


def _run_one(
    policy_name: str, policy: Policy, level: int, seed: int, trajectory_dir: Path | None
) -> EpisodeResult:
    env = FaultlineEnv(level=level, trajectory_dir=trajectory_dir)
    try:
        return run_episode(env, policy_name, policy, level=level, seed=seed)
    finally:
        env.close()


def run_benchmark(
    policies: dict[str, Policy],
    *,
    levels: list[int],
    seeds: list[int],
    workers: int,
    results_path: Path,
    trajectory_root: Path | None = None,
) -> list[EpisodeResult]:
    """Run every policy on every task. Results are appended as they finish, so a run can resume."""
    results: list[EpisodeResult] = []
    finished: set[tuple[str, int, int]] = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            record = json.loads(line)
            results.append(EpisodeResult(**{k: v for k, v in record.items() if k != "seed"}))
            finished.add((record["policy"], record["level"], record["seed"]))
    results_path.parent.mkdir(parents=True, exist_ok=True)
    jobs = [
        (name, policy, level, seed)
        for name, policy in policies.items()
        for level in levels
        for seed in seeds
        if (name, level, seed) not in finished
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool, results_path.open("a") as output:
        futures = {
            pool.submit(
                _run_one,
                name,
                policy,
                level,
                seed,
                trajectory_root / name if trajectory_root else None,
            ): (name, level, seed)
            for name, policy, level, seed in jobs
        }
        for future in as_completed(futures):
            name, level, seed = futures[future]
            try:
                result = future.result()
            except Exception as error:
                print(f"{name} level {level} seed {seed} crashed: {error!r}", flush=True)
                continue
            results.append(result)
            output.write(json.dumps({**asdict(result), "seed": seed}) + "\n")
            output.flush()
            print(
                f"{name:10s} L{level} seed {seed}: {'PASS' if result.success else 'fail'} "
                f"reward {result.total_reward:+.3f} steps {result.steps} "
                f"{result.failure_categories}",
                flush=True,
            )
    return results


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(results: list[EpisodeResult]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[EpisodeResult]] = defaultdict(list)
    for result in results:
        grouped[(result.policy, result.level)].append(result)
    summary = []
    for (policy, level), group in sorted(grouped.items()):
        summary.append(
            {
                "policy": policy,
                "level": level,
                "episodes": len(group),
                "passes": sum(item.success for item in group),
                "mean_reward": _mean([item.total_reward for item in group]),
                "mean_steps": _mean([float(item.steps) for item in group]),
                "mean_seconds": _mean([item.seconds for item in group]),
                "mean_difficulty": _mean([float(item.difficulty_score) for item in group]),
                "components": {
                    name: _mean([item.components.get(name, 0.0) for item in group])
                    for name in COMPONENTS
                },
                "failures": dict(
                    Counter(c for item in group for c in item.failure_categories).most_common()
                ),
                "ended_by": dict(Counter(item.ended_by for item in group)),
            }
        )
    return {"episodes": len(results), "by_policy_and_level": summary}


def write_report(results: list[EpisodeResult], directory: Path) -> None:
    summary = summarize(results)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "benchmark.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    rows = summary["by_policy_and_level"]
    policies = sorted({row["policy"] for row in rows})
    lines = [
        "# Benchmark results",
        "",
        f"{summary['episodes']} episodes. Regenerate with `python -m faultline.benchmark`.",
        "",
        "## Pass rate by level",
        "",
    ]
    levels = sorted({row["level"] for row in rows})
    by_key = {(row["policy"], row["level"]): row for row in rows}
    lines.extend(
        _table(
            ["Policy", *[f"Level {level}" for level in levels]],
            [
                [
                    policy,
                    *[
                        f"{by_key[(policy, level)]['passes']}/{by_key[(policy, level)]['episodes']}"
                        if (policy, level) in by_key
                        else "-"
                        for level in levels
                    ],
                ]
                for policy in policies
            ],
        )
    )
    lines.extend(["", "## Reward components, trajectory length and time", ""])
    lines.extend(
        _table(
            [
                "Policy",
                "Level",
                "Reward",
                "Steps",
                "Seconds",
                "Public",
                "Hidden",
                "Resilience",
                "Perf.",
            ],
            [
                [
                    row["policy"],
                    str(row["level"]),
                    f"{row['mean_reward']:+.3f}",
                    f"{row['mean_steps']:.1f}",
                    f"{row['mean_seconds']:.0f}",
                    *[f"{row['components'][name]:.2f}" for name in COMPONENTS],
                ]
                for row in rows
            ],
        )
    )
    lines.extend(["", "## Failure categories", ""])
    lines.extend(
        _table(
            ["Policy", "Level", "Failures"],
            [
                [
                    row["policy"],
                    str(row["level"]),
                    ", ".join(f"{name} ({count})" for name, count in row["failures"].items())
                    or "none",
                ]
                for row in rows
            ],
        )
    )
    (directory / "benchmark.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", default="gold,random,scripted")
    parser.add_argument("--levels", default=",".join(str(level) for level in LEVELS))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--model", default=None, help="Claude model for the 'claude' policy")
    parser.add_argument("--effort", default="high", help="Effort for the 'claude' policy")
    args = parser.parse_args()

    policies: dict[str, Policy] = {}
    for name in args.policies.split(","):
        if name == "claude":
            from faultline.claude_agent import DEFAULT_MODEL, make_claude_policy

            model = args.model or DEFAULT_MODEL
            policies[f"claude:{model}"] = make_claude_policy(model=model, effort=args.effort)
        else:
            policies[name] = POLICIES[name]

    clear_cache()
    results = run_benchmark(
        policies,
        levels=[int(level) for level in args.levels.split(",")],
        seeds=list(range(args.seeds)),
        workers=args.workers,
        results_path=args.out / "results.jsonl",
        trajectory_root=args.out / "trajectories",
    )
    write_report(results, args.out)
    print(f"wrote {args.out / 'benchmark.md'}")


if __name__ == "__main__":
    main()
