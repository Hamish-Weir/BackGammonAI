
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
import multiprocessing as mp
import os
from random import randint
import sys
import time
import traceback

import torch

from agents.MCTSAgent import MCTSAgent
from agents.T1RandomAgent import RandomAgent as T1RandomAgent
from agents.T2RandomAgent import RandomAgent as T2RandomAgent
from src.Game import Game
from src.Player import Player
from src.Colour import Colour
from src.Board import Board

def worker_init():
    # Called once per worker process
    # Limit threads used by numpy/pytorch/etc.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

def play_game():
    print(f"{os.getpid()}: Started")
    try:
        player1=Player(
            name="T1RandomAgent",
            agent=T1RandomAgent(Colour.RED),
        )

        player2=Player(
            name="T1RandomAgent",
            agent=T2RandomAgent(Colour.BLUE),
        )

        turn = 0
        current_player = Colour.RED
        board = Board()

        players = {
            Colour.RED: player1,
            Colour.BLUE: player2,
        }

        opponentMove = None

        while True:
            turn += 1

            currentPlayer: Player = players[current_player]
            playerAgent = currentPlayer.agent
            
            playerBoard = deepcopy(board)

            start = time.time_ns()
            ms = playerAgent.make_move(playerBoard, opponentMove)
            end = time.time_ns()

            currentPlayer.move_time += end - start

            board.make_move_sequence(ms,current_player)

            if board.has_ended():
                break

            current_player = Colour.opposite(current_player)
        
        print(f"{os.getpid()}: Finished")
        return board.get_winner(), turn
    
    except Exception as e:
        # write full traceback to a per-process file so you can inspect it
        pid = os.getpid()
        with open(f"worker_error_{pid}.log", "w") as f:
            f.write("Exception in worker:\n")
            traceback.print_exc(file=f)
        raise


if __name__ == "__main__":
    print(f"Start Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    mp.set_start_method('spawn')
    Red_Wins = 0
    Blue_Wins = 0
    game_lengths = []
    start = time.time_ns()
    with ProcessPoolExecutor(max_workers=16,initializer=worker_init) as ex:
            futures = [ex.submit(play_game) for i in range(100)]
            for future in as_completed(futures):
                # If a worker raised, .result() will re-raise that exception here.
                win,game_length = future.result()
                if win == Colour.RED:
                    Red_Wins+=1
                elif win == Colour.BLUE:
                    Blue_Wins+=1
                game_lengths.append(game_length)
    # play_game()
    end = time.time_ns()   
    print(f"Red Wins: {Red_Wins}")
    print(f"Blue Wins: {Blue_Wins}")
    print(f"Longest Game: {max(game_lengths)}")
    print(f"Average Game: {sum(game_lengths)/len(game_lengths)}")
    print(f"Shortest Game: {min(game_lengths)}")
    print(f"Time: {(end-start)/10**9}")
    print(f"End Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    
