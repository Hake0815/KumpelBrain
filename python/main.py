import functools
import os, sys
from threading import Event
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

file_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if file_folder not in sys.path:
    sys.path.insert(0, file_folder)

wrapper_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "game_logic_wrappers",
)
if wrapper_folder not in sys.path:
    sys.path.insert(0, wrapper_folder)

from game_player import GamePlayer


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


def callback_on_game_end(message: str, event: Event, uuid: str):
    event.set()


def run_single_game(game_num: int):
    """Run a single game and return the result"""
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
        enable_file_logging=True,  # Disable file logging for performance
    )

    game_player.play_game()
    event.wait()
    return game_num


def run_game_batch(batch_size, first_game_num: int):
    """Run a single game and return the result"""
    return [run_single_game(i + first_game_num) for i in range(batch_size)]


start_time = time.time()
num_game_batches = 1
num_games_per_batch = 1
num_games = num_game_batches * num_games_per_batch
max_workers = min(8, num_game_batches)  # More workers are slower

# Use ThreadPoolExecutor for parallel execution
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [
        executor.submit(run_game_batch, num_games_per_batch, i * num_games_per_batch)
        for i in range(num_game_batches)
    ]
    completed = 0
    for future in tqdm(as_completed(futures), total=num_game_batches):
        completed += 1
        if completed % 8 == 0:
            print(f"Completed {completed}/{num_games} games")

end_time = time.time()
elapsed = end_time - start_time
print(f"Time taken: {elapsed:.2f} seconds")
print(f"Games per second: {num_games / elapsed:.2f}")
