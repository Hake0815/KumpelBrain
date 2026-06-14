"""Run complete rollout policy-iteration cycles within a runtime budget."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def runtime_budget_exhausted(
    started_at: float,
    max_runtime_hours: float | None,
    *,
    now: float | None = None,
) -> bool:
    if max_runtime_hours is None:
        return False
    current = time.monotonic() if now is None else now
    return current - started_at >= max_runtime_hours * 3600.0


def build_iteration_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("train_self_play.py")),
        "--rollout-dir",
        str(args.rollout_dir),
        "--model-dir",
        str(args.model_dir),
        "--seed-games",
        str(args.seed_games),
        "--root-states",
        str(args.root_states),
        "--train-steps",
        str(args.train_steps),
        "--batch-size",
        str(args.batch_size),
        "--opponent-history",
        str(args.opponent_history),
        "--replay-shards",
        str(args.replay_shards),
        "--collection-mode",
        args.collection_mode,
        "--concurrent-games",
        str(args.concurrent_games),
        "--inference-batch-size",
        str(args.inference_batch_size),
        "--batch-linger-ms",
        str(args.batch_linger_ms),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate complete rollout policy-iteration cycles"
    )
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-runtime-hours", type=float)
    parser.add_argument("--seed-games", type=int, default=100)
    parser.add_argument("--root-states", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--opponent-history", type=int, default=4)
    parser.add_argument("--replay-shards", type=int, default=5)
    parser.add_argument(
        "--collection-mode",
        choices=("auto", "serial", "coordinator"),
        default="auto",
    )
    parser.add_argument("--concurrent-games", type=int, default=32)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--batch-linger-ms", type=float, default=1.0)
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        default=Path("training_data/rollouts"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("training_data/models"),
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("KUMPEL_DEVICE", "cuda"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_iterations < 1:
        parser.error("--max-iterations must be at least 1")
    if args.max_runtime_hours is not None and args.max_runtime_hours <= 0:
        parser.error("--max-runtime-hours must be greater than 0")
    return args


def main() -> int:
    args = parse_args()
    if not args.rollout_dir.is_absolute():
        args.rollout_dir = REPO_ROOT / args.rollout_dir
    if not args.model_dir.is_absolute():
        args.model_dir = REPO_ROOT / args.model_dir
    command = build_iteration_command(args)
    if args.dry_run:
        print(" ".join(command))
        return 0

    started_at = time.monotonic()
    for iteration in range(1, args.max_iterations + 1):
        if runtime_budget_exhausted(started_at, args.max_runtime_hours):
            elapsed = (time.monotonic() - started_at) / 3600.0
            print(
                f"Runtime budget reached after {elapsed:.2f} hours; "
                "not starting another policy iteration."
            )
            return 0
        print(f"\n=== Rollout policy iteration {iteration}/{args.max_iterations} ===")
        env = os.environ.copy()
        env["KUMPEL_DEVICE"] = args.device
        try:
            result = subprocess.run(command, cwd=REPO_ROOT, env=env)
        except KeyboardInterrupt:
            print("\nTraining automation interrupted.", file=sys.stderr)
            return 130
        if result.returncode != 0:
            print(
                f"Policy iteration {iteration} failed with "
                f"exit code {result.returncode}.",
                file=sys.stderr,
            )
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
