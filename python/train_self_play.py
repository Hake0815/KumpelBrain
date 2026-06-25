"""Run one rollout-based approximate policy-iteration cycle."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
for path in (
    repo_root / "python",
    repo_root / "python" / "network",
    repo_root / "python" / "training",
    repo_root / "python" / "game_logic_wrappers",
    repo_root / "cpp" / "build",
):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

import csharp_runtime  # noqa: F401
import torch
from tqdm import tqdm

from inference_models import create_self_play_models, resolve_compute_device
from inference_service import (
    BatchedInferenceClient,
    BatchedInferenceService,
    DirectInferenceClient,
)
from main import create_deck_list
from training.gameplay import play_candidate_match, run_seed_game
from training.policy_iteration import (
    OpponentPool,
    RolloutEvaluator,
    run_champion_gate,
    sample_phase_balanced_roots,
)
from training.rollout_data import RolloutShard, RolloutStore
from training.rollout_runner import ContinuationRolloutRunner
from training.trainer import (
    CheckpointMetadata,
    ModelStore,
    RolloutTrainer,
    split_roots,
)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    return float(value) if value else default


def resolve_collection_mode(
    requested: str,
    device: torch.device,
    games: int,
) -> str:
    if requested == "auto":
        return "coordinator" if device.type == "cuda" and games > 1 else "serial"
    return requested


def _initial_manifest() -> dict:
    return {"format": "rollout-policy-v1", "iteration": 0, "iterations": []}


def _load_manifest(store: ModelStore) -> dict:
    if not store.manifest_path.exists():
        return _initial_manifest()
    return json.loads(store.manifest_path.read_text(encoding="utf-8"))


def ensure_champion(store: ModelStore, device: torch.device) -> CheckpointMetadata:
    if store.champion_path.exists():
        network, selector, _ = create_self_play_models(
            device,
            embed_on_compute_device=device.type == "cuda",
        )
        return store.load_checkpoint(store.champion_path, network, selector, device)

    network, selector, _ = create_self_play_models(
        device,
        embed_on_compute_device=device.type == "cuda",
    )
    metadata = CheckpointMetadata(
        iteration=0,
        champion_id=uuid.uuid4().hex[:12],
        role="champion",
    )
    store.save_checkpoint(
        store.champion_path,
        network,
        selector,
        metadata=metadata,
        metrics={"initialized_randomly": True},
    )
    store.manifest_path.write_text(
        json.dumps(_initial_manifest(), indent=2),
        encoding="utf-8",
    )
    return metadata


def load_client(
    path: Path,
    device: torch.device,
) -> tuple[DirectInferenceClient, CheckpointMetadata]:
    network, selector, _ = create_self_play_models(
        device,
        embed_on_compute_device=device.type == "cuda",
    )
    metadata = ModelStore.load_checkpoint(path, network, selector, device)
    network.eval()
    selector.eval()
    return DirectInferenceClient(network, selector, device), metadata


def collect_seed_roots(
    *,
    games: int,
    champion: DirectInferenceClient,
    opponent_pool: OpponentPool,
    device: torch.device,
    collection_mode: str,
    concurrent_games: int,
    inference_batch_size: int,
    batch_linger_ms: float,
    seed: int,
) -> list:
    mode = resolve_collection_mode(collection_mode, device, games)
    service: BatchedInferenceService | None = None
    champion_client = champion
    if mode == "coordinator":
        service = BatchedInferenceService(
            champion.network,
            champion.selector,
            device,
            max_batch=inference_batch_size,
            linger_ms=batch_linger_ms,
        )
        champion_client = BatchedInferenceClient(service)

    rng = random.Random(seed)
    seeds = [rng.randrange(2**31) for _ in range(games)]

    def play_one(game_seed: int):
        if service is not None:
            service.register_game()
        try:
            return run_seed_game(
                deck_list1=create_deck_list(),
                deck_list2=create_deck_list(),
                player1_name="player1",
                player2_name="player2",
                champion=champion_client,
                opponent=opponent_pool.sample(champion_client),
                compute_device=device,
                seed=game_seed,
            )
        except BaseException:
            if service is not None:
                service.game_finished()
            raise

    roots = []
    try:
        with tqdm(total=games, desc="Seed games", unit="game", smoothing=0) as bar:
            if mode == "serial":
                for game_seed in seeds:
                    roots.extend(play_one(game_seed))
                    bar.update(1)
            else:
                with ThreadPoolExecutor(
                    max_workers=min(concurrent_games, games),
                    thread_name_prefix="seed-game",
                ) as executor:
                    futures = [executor.submit(play_one, game_seed) for game_seed in seeds]
                    for future in as_completed(futures):
                        roots.extend(future.result())
                        bar.update(1)
    finally:
        if service is not None:
            service.shutdown()
    return roots


def train_candidate(
    *,
    model_store: ModelStore,
    rollout_store: RolloutStore,
    device: torch.device,
    train_steps: int,
    batch_size: int,
    replay_shards: int,
    iteration: int,
    candidate_id: str,
) -> tuple[DirectInferenceClient, dict]:
    network, selector, _ = create_self_play_models(
        device,
        embed_on_compute_device=device.type == "cuda",
    )
    ModelStore.load_checkpoint(
        model_store.champion_path,
        network,
        selector,
        device,
    )
    trainer = RolloutTrainer(network, selector, device, seed=iteration)
    training_roots, validation_roots = split_roots(
        rollout_store.recent(replay_shards)
    )
    if not training_roots:
        raise RuntimeError("No rollout training roots were generated")

    last_result = None
    with tqdm(range(train_steps), desc="Train steps", unit="step", smoothing=0) as bar:
        for _ in bar:
            batch = trainer.sample_batch(training_roots, batch_size)
            last_result = trainer.train_step(batch)
            bar.set_postfix(
                loss=f"{last_result.loss:.4f}",
                value=f"{last_result.interaction_value:.4f}",
                policy=f"{last_result.interaction_policy:.4f}",
                refresh=False,
            )

    validation_loss = (
        trainer.validation_loss(validation_roots, batch_size)
        if validation_roots
        else None
    )
    calibration = (
        trainer.calibration_metrics(validation_roots, batch_size)
        if validation_roots
        else {"interaction_brier": None, "target_brier": None}
    )
    metrics = {
        "train_steps": trainer.train_steps,
        "training_roots": len(training_roots),
        "validation_roots": len(validation_roots),
        "validation_loss": validation_loss,
        "calibration": calibration,
        "final_train_loss": None if last_result is None else last_result.loss,
    }
    ModelStore.save_checkpoint(
        model_store.candidate_path,
        network,
        selector,
        metadata=CheckpointMetadata(iteration, candidate_id, "candidate"),
        metrics=metrics,
    )
    network.eval()
    selector.eval()
    return DirectInferenceClient(network, selector, device), metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one rollout policy-iteration cycle"
    )
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
    parser.add_argument(
        "--concurrent-games",
        type=int,
        default=_env_int("KUMPEL_CONCURRENT_GAMES", 32),
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=_env_int("KUMPEL_BATCH_MAX", 16),
    )
    parser.add_argument(
        "--batch-linger-ms",
        type=float,
        default=_env_float("KUMPEL_BATCH_LINGER_MS", 1.0),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gate-min-games", type=int, default=200)
    parser.add_argument("--gate-max-games", type=int, default=800)
    parser.add_argument("--gate-block-size", type=int, default=100)
    parser.add_argument("--root-action-rollouts", type=int, default=10)
    parser.add_argument("--root-action-temperature", type=float, default=1.0)
    parser.add_argument("--target-contexts-per-action", type=int, default=2)
    parser.add_argument("--target-choices-per-context", type=int, default=3)
    parser.add_argument("--target-rollouts-per-choice", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_compute_device()
    rollout_store = RolloutStore(args.rollout_dir)
    model_store = ModelStore(args.model_dir)
    champion_metadata = ensure_champion(model_store, device)
    champion, champion_metadata = load_client(model_store.champion_path, device)
    iteration = max(
        champion_metadata.iteration + 1,
        len(list(args.rollout_dir.glob("rollout_*.pt"))) + 1,
    )
    opponent_pool = OpponentPool(
        model_store,
        device,
        history_limit=args.opponent_history,
        seed=args.seed + iteration,
    )

    roots = collect_seed_roots(
        games=args.seed_games,
        champion=champion,
        opponent_pool=opponent_pool,
        device=device,
        collection_mode=args.collection_mode,
        concurrent_games=args.concurrent_games,
        inference_batch_size=args.inference_batch_size,
        batch_linger_ms=args.batch_linger_ms,
        seed=args.seed + iteration,
    )
    sampled_roots = sample_phase_balanced_roots(
        roots,
        args.root_states,
        seed=args.seed + iteration,
    )
    if not sampled_roots:
        raise RuntimeError("Seed games produced no recreatable trainable roots")

    runner = ContinuationRolloutRunner(
        champion,
        opponent_pool.sample,
        device,
        seed=args.seed + iteration,
    )
    evaluator = RolloutEvaluator(
        champion,
        runner,
        root_action_rollouts=args.root_action_rollouts,
        root_action_temperature=args.root_action_temperature,
        target_contexts_per_action=args.target_contexts_per_action,
        target_choices_per_context=args.target_choices_per_context,
        target_rollouts_per_choice=args.target_rollouts_per_choice,
        seed=args.seed + iteration,
    )
    evaluated_roots = []
    with tqdm(desc="Rollout branches", unit="rollout", smoothing=0) as progress:
        for root in sampled_roots:
            evaluated = evaluator.evaluate_root(root, progress=progress.update)
            if evaluated.action_estimates:
                evaluated_roots.append(evaluated)

    if not evaluated_roots:
        representative_mismatches = evaluator.stats.mismatches[:5]
        details = (
            "; ".join(representative_mismatches)
            if representative_mismatches
            else "no replay mismatch details were recorded"
        )
        raise RuntimeError(
            "All forced continuations were incompatible with their recreated "
            f"determinizations; no rollout shard was written. Mismatches: {details}"
        )

    shard = RolloutShard(
        iteration=iteration,
        champion_id=champion_metadata.champion_id,
        roots=evaluated_roots,
        metadata={
            "attempted_rollouts": evaluator.stats.attempted,
            "successful_rollouts": evaluator.stats.succeeded,
            "mismatched_rollouts": evaluator.stats.mismatched,
            "representative_mismatches": evaluator.stats.mismatches[:10],
            "seed_games": args.seed_games,
            "root_action_rollouts": args.root_action_rollouts,
            "root_action_temperature": args.root_action_temperature,
            "target_contexts_per_action": args.target_contexts_per_action,
            "target_choices_per_context": args.target_choices_per_context,
            "target_rollouts_per_choice": args.target_rollouts_per_choice,
        },
    )
    rollout_store.save(shard)

    candidate_id = uuid.uuid4().hex[:12]
    candidate, training_metrics = train_candidate(
        model_store=model_store,
        rollout_store=rollout_store,
        device=device,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        replay_shards=args.replay_shards,
        iteration=iteration,
        candidate_id=candidate_id,
    )

    gate_progress = tqdm(
        total=args.gate_max_games,
        desc="Gating matches",
        unit="game",
        smoothing=0,
    )

    def play_block(count: int, candidate_is_player1: bool) -> list[float]:
        scores = []
        for _ in range(count):
            scores.append(
                play_candidate_match(
                    deck_list1=create_deck_list(),
                    deck_list2=create_deck_list(),
                    candidate=candidate,
                    champion=champion,
                    candidate_is_player1=candidate_is_player1,
                    compute_device=device,
                )
            )
            gate_progress.update(1)
        return scores

    try:
        gate = run_champion_gate(
            play_block,
            block_size=args.gate_block_size,
            minimum_games=args.gate_min_games,
            maximum_games=args.gate_max_games,
        )
    finally:
        gate_progress.close()

    manifest = _load_manifest(model_store)
    iteration_record = {
        "iteration": iteration,
        "champion_before": champion_metadata.champion_id,
        "candidate": candidate_id,
        "rollout": shard.metadata,
        "training": training_metrics,
        "gate": {
            "promoted": gate.promoted,
            "games": gate.games,
            "wins": gate.wins,
            "draws": gate.draws,
            "losses": gate.losses,
            "lower": gate.interval.lower,
            "upper": gate.interval.upper,
            "reason": gate.reason,
        },
    }
    manifest["iteration"] = iteration
    manifest.setdefault("iterations", []).append(iteration_record)
    if gate.promoted:
        model_store.promote(
            model_store.candidate_path,
            previous_champion_id=champion_metadata.champion_id,
            iteration=iteration,
            history_limit=args.opponent_history,
            manifest=manifest,
        )
        print(f"Promoted candidate {candidate_id} to champion.")
    else:
        model_store.manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        print(f"Candidate rejected: {gate.reason}.")


if __name__ == "__main__":
    main()
