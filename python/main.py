import asyncio
import functools
import os, sys
from threading import Event
import threading
import time

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


def callback_on_game_end(message: str, event: Event):
    print(message)
    event.set()


event = Event()

start_time = time.time()
gamePlayer = GamePlayer(
    log_file_path="log.txt",
    deck_list1=create_deck_list(),
    deck_list2=create_deck_list(),
    player1_name="player1",
    player2_name="player2",
    callback_on_game_end=functools.partial(callback_on_game_end, event=event),
)
t = threading.Thread(target=gamePlayer.play_game)

t.start()

event.wait()
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
