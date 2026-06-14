import functools
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from queue import Empty, Queue
from threading import Event, Thread
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

# CoreCLR must initialize before torch, or pythonnet falls back to Mono and crashes
# when loading .NET 10 game assemblies.
import csharp_runtime  # noqa: F401

import torch

from game_player import GamePlayer
from inference_models import create_self_play_models
from inference_service import (
    BatchedInferenceClient,
    BatchedInferenceService,
    DirectInferenceClient,
)
from network.kumpel_network import KumpelNetwork
from network.selector import Selector
from profiling import profiling_enabled

_GAME_PROGRESS_KWARGS = {
    "unit": "game",
    "desc": "Games",
    # Average rate since start (not EMA of the last update burst).
    "smoothing": 0,
}


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
_MAIN_INFERENCE: DirectInferenceClient | None = None


def _ensure_main_models() -> tuple[DirectInferenceClient, torch.device]:
    global _MAIN_NETWORK, _MAIN_SELECTOR, _MAIN_DEVICE, _MAIN_INFERENCE
    if _MAIN_INFERENCE is None:
        _MAIN_NETWORK, _MAIN_SELECTOR, _MAIN_DEVICE = create_self_play_models()
        assert _MAIN_SELECTOR is not None and _MAIN_DEVICE is not None
        _MAIN_INFERENCE = DirectInferenceClient(
            _MAIN_NETWORK, _MAIN_SELECTOR, _MAIN_DEVICE
        )
    assert _MAIN_DEVICE is not None
    return _MAIN_INFERENCE, _MAIN_DEVICE


# Per-worker models for process pool (each process has its own copy).
_WORKER_INFERENCE: DirectInferenceClient | None = None
_WORKER_DEVICE: torch.device | None = None


def _init_worker_process() -> None:
    global _WORKER_INFERENCE, _WORKER_DEVICE
    import csharp_runtime  # noqa: F401

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    network, selector, device = create_self_play_models()
    _WORKER_INFERENCE = DirectInferenceClient(network, selector, device)
    _WORKER_DEVICE = device


def run_single_game(
    game_num: int,
    inference=None,
    compute_device=None,
) -> int:
    """Run a single game and return the game number."""
    if inference is None or compute_device is None:
        inference, compute_device = _ensure_main_models()

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
        inference=inference,
        compute_device=compute_device,
        enable_file_logging=False,
    )

    game_player.play_game()
    event.wait()
    return game_num


def _run_single_game_worker(game_num: int) -> int:
    assert _WORKER_INFERENCE is not None and _WORKER_DEVICE is not None
    return run_single_game(
        game_num,
        inference=_WORKER_INFERENCE,
        compute_device=_WORKER_DEVICE,
    )


def _resolve_num_games() -> int:
    num_games_env = os.environ.get("KUMPEL_NUM_GAMES", "").strip()
    if num_games_env:
        return max(1, int(num_games_env))
    num_game_batches = int(os.environ.get("KUMPEL_NUM_BATCHES", "1"))
    num_games_per_batch = int(os.environ.get("KUMPEL_GAMES_PER_BATCH", "10"))
    return max(1, num_game_batches * num_games_per_batch)


def _physical_core_count() -> int:
    """Return physical CPU cores when sysfs topology is available."""
    topology_root = Path("/sys/devices/system/cpu")
    if not topology_root.is_dir():
        return os.cpu_count() or 1

    core_ids: set[tuple[int, int]] = set()
    for cpu_dir in topology_root.glob("cpu[0-9]*"):
        topology_dir = cpu_dir / "topology"
        core_id_path = topology_dir / "core_id"
        package_id_path = topology_dir / "physical_package_id"
        if not core_id_path.is_file() or not package_id_path.is_file():
            continue
        try:
            core_id = int(core_id_path.read_text().strip())
            package_id = int(package_id_path.read_text().strip())
        except ValueError:
            continue
        core_ids.add((package_id, core_id))

    if core_ids:
        return len(core_ids)
    return os.cpu_count() or 1


def _resolve_workers(num_games: int) -> int:
    physical_cores = _physical_core_count()
    default_workers = min(physical_cores, num_games)
    workers_env = os.environ.get("KUMPEL_WORKERS", "").strip()
    workers = int(workers_env) if workers_env else default_workers
    return max(1, min(workers, num_games))


def _resolve_coordinator_settings(num_games: int) -> tuple[int, int, float]:
    max_batch = int(os.environ.get("KUMPEL_BATCH_MAX", "8"))
    linger_ms = float(os.environ.get("KUMPEL_BATCH_LINGER_MS", "2"))
    concurrent_env = os.environ.get("KUMPEL_CONCURRENT_GAMES", "").strip()
    if concurrent_env:
        concurrent_games = int(concurrent_env)
    else:
        concurrent_games = max(max_batch, min(num_games, 16))
    concurrent_games = max(1, min(concurrent_games, num_games))
    return max_batch, concurrent_games, linger_ms


def _run_games_in_process_pool(num_games: int, workers: int) -> None:
    ctx = get_context("spawn")
    with tqdm(total=num_games, **_GAME_PROGRESS_KWARGS) as bar:
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
    with tqdm(total=num_games, **_GAME_PROGRESS_KWARGS) as bar:
        for i in range(num_games):
            run_single_game(i)
            bar.update(1)


def _run_games_coordinator(num_games: int) -> None:
    max_batch, concurrent_games, linger_ms = _resolve_coordinator_settings(num_games)
    network, selector, device = create_self_play_models(embed_on_compute_device=True)
    service = BatchedInferenceService(
        network,
        selector,
        device,
        max_batch=max_batch,
        linger_ms=linger_ms,
    )
    inference = BatchedInferenceClient(service)
    completion_queue: Queue[None] = Queue()

    def _run_one_game() -> None:
        game_uuid = uuid.uuid4()
        game_over_event = Event()
        service.register_game()
        try:
            game_player = GamePlayer(
                deck_list1=create_deck_list(),
                deck_list2=create_deck_list(),
                player1_name="player1",
                player2_name="player2",
                game_uuid=game_uuid,
                callback_on_game_end=functools.partial(
                    callback_on_game_end, event=game_over_event, uuid=game_uuid
                ),
                inference=inference,
                compute_device=device,
                enable_file_logging=False,
            )
            game_player.play_game()
            game_over_event.wait()
        finally:
            completion_queue.put(None)

    with tqdm(total=num_games, **_GAME_PROGRESS_KWARGS) as bar:
        active_threads: list[Thread] = []
        next_game = 0
        finished = 0

        def _launch_game() -> None:
            nonlocal next_game
            thread = Thread(
                target=_run_one_game,
                name=f"game-{next_game}",
                daemon=False,
            )
            thread.start()
            active_threads.append(thread)
            next_game += 1

        while next_game < num_games and len(active_threads) < concurrent_games:
            _launch_game()

        while finished < num_games:
            completion_queue.get()
            finished += 1
            while True:
                try:
                    completion_queue.get_nowait()
                    finished += 1
                except Empty:
                    break

            bar.update(finished - bar.n)
            active_threads = [thread for thread in active_threads if thread.is_alive()]
            while next_game < num_games and len(active_threads) < concurrent_games:
                _launch_game()

        for thread in active_threads:
            thread.join()

    service.shutdown()


def main():
    num_games = _resolve_num_games()
    workers = _resolve_workers(num_games)
    parallel_mode = os.environ.get("KUMPEL_PARALLEL", "process").lower()

    start_time = time.time()

    if parallel_mode in ("coordinator", "coord"):
        _run_games_coordinator(num_games)
    elif parallel_mode in ("thread", "threads"):
        _run_games_serial(num_games)
    elif parallel_mode in ("process", "processes", "1", "true") and workers > 1:
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
