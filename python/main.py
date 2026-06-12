import functools
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from threading import Event
import time
import uuid
from pathlib import Path

from tqdm import tqdm

repo_root = Path(__file__).resolve().parents[1]
file_dir = repo_root / "python"
wrapper_dir = repo_root / "python" / "game_logic_wrappers"
network_dir = repo_root / "python" / "network"
cpp_build_dir = repo_root / "cpp" / "build"
for p in (file_dir, wrapper_dir, network_dir, cpp_build_dir):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

# Pin BLAS/OpenMP in worker children before torch initializes thread pools.
if multiprocessing.parent_process() is not None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

from game_player import GamePlayer
from inference_models import create_self_play_models
from network.kumpel_network import KumpelNetwork
from network.selector import Selector
from profiling import profiling_enabled


def create_deck_list():
    return {
        "professorsResearch": 8,
        "TWM128": 8,
        "TWM129": 8,
        "ultraBall": 12,
        "nightStretcher": 10,
        "FireNRG": 7,
        "PsychicNRG": 7,
    }


def callback_on_game_end(message: str, event: Event, uuid: uuid.UUID):
    event.set()


# Shared across games in the main process (serial path).
_MAIN_NETWORK: KumpelNetwork | None = None
_MAIN_SELECTOR: Selector | None = None
_MAIN_DEVICE: torch.device | None = None


def _ensure_main_models() -> tuple[KumpelNetwork, Selector, torch.device]:
    global _MAIN_NETWORK, _MAIN_SELECTOR, _MAIN_DEVICE
    if _MAIN_NETWORK is None:
        _MAIN_NETWORK, _MAIN_SELECTOR, _MAIN_DEVICE = create_self_play_models()
    assert _MAIN_SELECTOR is not None and _MAIN_DEVICE is not None
    return _MAIN_NETWORK, _MAIN_SELECTOR, _MAIN_DEVICE


# Per-worker models for process pool (each process has its own copy).
_WORKER_NETWORK: KumpelNetwork | None = None
_WORKER_SELECTOR: Selector | None = None
_WORKER_DEVICE: torch.device | None = None


def _init_worker_process() -> None:
    global _WORKER_NETWORK, _WORKER_SELECTOR, _WORKER_DEVICE
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _WORKER_NETWORK, _WORKER_SELECTOR, _WORKER_DEVICE = create_self_play_models()


def run_single_game(
    game_num: int,
    network=None,
    selector=None,
    compute_device=None,
) -> int:
    """Run a single game and return the game number."""
    if network is None or selector is None or compute_device is None:
        network, selector, compute_device = _ensure_main_models()

    game_uuid = uuid.uuid4()
    event = Event()

    game_player = GamePlayer(
        deck_list1=create_deck_list(),
        deck_list2=create_deck_list(),
        player1_name="player1",
        player2_name="player2",
        game_uuid=game_uuid,
        callback_on_game_end=functools.partial(
            callback_on_game_end, event=event, uuid=game_uuid
        ),
        network=network,
        selector=selector,
        compute_device=compute_device,
        enable_file_logging=False,
    )

    game_player.play_game()
    event.wait()
    return game_num


def _run_single_game_worker(game_num: int) -> int:
    assert (
        _WORKER_NETWORK is not None
        and _WORKER_SELECTOR is not None
        and _WORKER_DEVICE is not None
    )
    return run_single_game(
        game_num,
        network=_WORKER_NETWORK,
        selector=_WORKER_SELECTOR,
        compute_device=_WORKER_DEVICE,
    )


def _resolve_num_games() -> int:
    num_games_env = os.environ.get("KUMPEL_NUM_GAMES", "").strip()
    if num_games_env:
        return max(1, int(num_games_env))
    num_game_batches = int(os.environ.get("KUMPEL_NUM_BATCHES", "1"))
    num_games_per_batch = int(os.environ.get("KUMPEL_GAMES_PER_BATCH", "10"))
    return max(1, num_game_batches * num_games_per_batch)


def _resolve_workers(num_games: int) -> int:
    cpu_count = os.cpu_count() or 1
    default_workers = min(cpu_count, num_games)
    workers_env = os.environ.get("KUMPEL_WORKERS", "").strip()
    workers = int(workers_env) if workers_env else default_workers
    return max(1, min(workers, num_games))


def _run_games_in_process_pool(num_games: int, workers: int) -> None:
    ctx = get_context("spawn")
    with tqdm(total=num_games, unit="game", desc="Games") as bar:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_worker_process,
        ) as executor:
            futures = [
                executor.submit(_run_single_game_worker, i) for i in range(num_games)
            ]
            for future in as_completed(futures):
                future.result()
                bar.update(1)


def _run_games_serial(num_games: int) -> None:
    _ensure_main_models()
    with tqdm(total=num_games, unit="game", desc="Games") as bar:
        for i in range(num_games):
            run_single_game(i)
            bar.update(1)


def main():
    num_games = _resolve_num_games()
    workers = _resolve_workers(num_games)

    use_processes = os.environ.get("KUMPEL_PARALLEL", "process").lower() in (
        "process",
        "processes",
        "1",
        "true",
    )
    use_threads = os.environ.get("KUMPEL_PARALLEL", "").lower() in (
        "thread",
        "threads",
    )

    start_time = time.time()

    if use_processes and not use_threads and workers > 1:
        _run_games_in_process_pool(num_games, workers)
    else:
        _run_games_serial(num_games)

    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.2f} seconds")
    print(f"Games per second: {num_games / elapsed:.2f}")
    if profiling_enabled():
        print(
            "(Per-game profiling printed at game end; set KUMPEL_PROFILE=0 to disable)"
        )


if __name__ == "__main__":
    main()
