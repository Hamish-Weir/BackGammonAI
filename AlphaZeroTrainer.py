# Attributes:
# - model: AlphaZeroNet
# - optimizer: torch.optim.SGD
# - replay_buffer: deque
# - batch_size: 32
# - device: torch.device
# - temperature: float (for exploration)
# - simulations: 600 train, 1000 evaluate
# - num_selfplay_games: 64?
# - update_steps: all game states in replay buffer
# - best_model_path: "models/best_model.pth"

# Methods:
# 1. self_play():
#    for each game:
#        initialize board
#        play full game with AlphaZeroAgent
#        for each move in game:
#            store (encoded_board, policy_target, game_result)
#    append to replay buffer

# 2. sample_batch():
#    randomly sample from replay_buffer

# 3. train_step():
#    forward pass -> compute loss -> backward -> optimizer.step()

# 4. evaluate_model():
#    play test matches vs best_model
#    if win rate > threshold:
#        replace best_model.pth

# 5. train():
#    repeat self-play + training steps + evaluation

from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import multiprocessing as mp
from datetime import datetime
import os
import pickle
import random
import time
import traceback

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import SGD

from BackgammonUtils import BackgammonUtils
from src.Board import Board
from src.Player import Player
from src.Colour import Colour

from agents.AlphaZeroAgent import AlphaZeroAgent
from networks.AlphaZeroNetwork import AlphaZeroNet

TRAIN_SIMS = 600
EVAL_SIMS = 1000

GAMES_PER_TRAIN = 400
GAMES_PER_EVAL = 100


class AlphaZeroTrainer():

    def __init__(
        self,
        learning_rate       = 0.02,
        batch_size          = 32,
        device              = "cpu",
        best_model_path     = "models/best_model.pth",
        new_model_path      = "models/new_model.pth",
        dataset_path        = "dataset.pkl"
    ):
        

        self.learning_rate          = learning_rate
        self.batch_size             = batch_size

        self.dataset_path           = dataset_path
        self.deque_size             = GAMES_PER_TRAIN * 3
        self.dataset                = self.load_deque() # List of List of gamestates (boardstates from past 10 training iterations)

        self.epochs                 = 2
        if device:
            self.device             = torch.device(device)
        else:
            self.device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_model_path        = best_model_path
        self.new_model_path         = new_model_path
        self.best_model             = AlphaZeroNet().to(device)
        self.new_model              = AlphaZeroNet().to(device)

        # Load best model if exists
        try:
            self.best_model.load_state_dict(torch.load(self.best_model_path,weights_only=True))
            self.new_model.load_state_dict(torch.load(self.best_model_path,weights_only=True))
            print("Loaded existing best model", flush=True)
        except:
            torch.save(self.best_model.state_dict(), self.best_model_path)
            self.new_model.load_state_dict(self.best_model.state_dict())
            print("Training new model from scratch", flush=True)

    def _play_step(self):
        print(f"Playing Self Play Games...  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}", flush=True)
        self.dataset.extend(generate_dataset())
        self.save_deque() # Persist games over runs

    def _train_step(self):
        print(f"Training Network...         {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}", flush=True)
        flat_dataset = [pos for game in self.dataset for pos in game]

        optimizer = SGD(
            self.new_model.parameters(),
            lr=self.learning_rate,  # initial learning rate
            momentum=0.9,           # momentum term
            weight_decay=1e-4       # for regularization
        )

        self.new_model.train()
        
        for _ in range(self.epochs): # *Not technicaly epochs


            states = torch.stack([s for (s, _, _) in flat_dataset]).to(self.device)
            policies = torch.tensor(np.array([p for (_, p, _) in flat_dataset]), dtype=torch.float).to(self.device)
            values = torch.tensor(np.array([v for (_, _, v) in flat_dataset]), dtype=torch.float).to(self.device)

            for start in range(0, len(states), self.batch_size):
                s_batch = states[start:start + self.batch_size]
                p_batch = policies[start:start + self.batch_size]
                v_batch = values[start:start + self.batch_size]

                v_pred, policy  = self.new_model(s_batch)

                # Loss = policy loss + value loss + L2 regularization
                policy_loss = -(p_batch * policy).sum(dim=1).mean()
                value_loss = F.mse_loss(v_pred.view(-1), v_batch)

                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.new_model.eval()

    def _evaluate_step(self):
        print(f"Playing Evaluation Games... {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}", flush=True)
        torch.save(self.new_model.state_dict(), self.new_model_path)
        wins = evaluate_model()

        print(f"New model wins {wins}/{GAMES_PER_EVAL}", flush=True)

        if wins/GAMES_PER_EVAL >  0.55: # Update Best model when new is better
            print("New model accepted, Saved!", flush=True)

            #New model better than previous Best Model
            torch.save(self.new_model.state_dict(), self.best_model_path)  # Overwrite Best Model Save
            self.best_model.load_state_dict(torch.load(self.best_model_path))   # Read Best Model Save

        else:
            print("New model rejected. Continuing.", flush=True)
            # self.new_model.load_state_dict(torch.load(self.best_model_path)) # Read Best Model Save


    def train(self, iterations=20):
        print("Starting Training...", flush=True)
        self.best_model.load_state_dict(torch.load(self.best_model_path,weights_only=True))
        self.new_model.load_state_dict(torch.load(self.best_model_path,weights_only=True))

        for it in range(0, iterations):
            try:
                print(f"\n### TRAINING ITERATION {it+1} ###", flush=True)
                start = time.perf_counter()

                # SELF-PLAY
                self._play_step()

                # TRAIN NETWORK
                self._train_step()

                # EVALUATE AGAINST BEST MODEL
                self._evaluate_step()


                end = time.perf_counter()
                print(f"Iteration Time: {end-start}", flush=True)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error Occured: {e}", flush=True)

    def load_deque(self):
        """Safely load a deque"""
        
        if not os.path.exists(self.dataset_path):
            print(f"{self.dataset_path} not found. Creating a new empty deque.")
            return deque(maxlen=self.deque_size)

        try:
            with open(self.dataset_path, "rb") as f:
                obj = pickle.load(f)

            if not isinstance(obj, deque):
                print(f"Warning: {self.dataset_path} did not contain a deque. Creating a new one.")
                return deque(maxlen=self.deque_size)
            
            print("Loaded Existing Game Data")

            return obj

        except Exception as e:
            print(f"Error loading {self.dataset_path}: {e}")
            print("Creating a new empty deque.")
            return deque(maxlen=self.deque_size)
      
    def save_deque(self):
        """Safely save a deque to disk."""
        with open(self.dataset_path, "wb") as f:
            pickle.dump(self.dataset, f)

# ---------- Trainer usage ----------

def worker_init():
    # Called once per worker process
    # Limit threads used by numpy/pytorch/etc.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

def self_play_game():
    # print(f"{os.getpid()}: Started")
    try:
        agent1=AlphaZeroAgent(Colour.RED,TRAIN_SIMS,training = True)
        agent2=AlphaZeroAgent(Colour.BLUE,TRAIN_SIMS,training = True)

        players = {
            Colour.RED: agent1,
            Colour.BLUE: agent2,
        }

        current_player = Colour.RED
        board = Board()

        opponentMove = None

        game_history = []

        while True:
            
            dice = [random.randint(1,6),random.randint(1,6)]

            playerAgent:AlphaZeroAgent = players[current_player]
            
            playerBoard = deepcopy(board)
            playerDice = deepcopy(dice)

            ms = playerAgent.make_move(playerBoard, playerDice, opponentMove)

            game_history.extend(playerAgent.get_state())
            
            board.make_move_sequence(ms,current_player)

            if board.has_ended():
                break
            
            current_player = Colour.opposite(current_player)
        
        winner = board.get_winner()

        for i, (s, p, pl) in enumerate(game_history):
            value = 1 if pl == winner else -1
            game_history[i] = (s, p, value)

        # print(f"{os.getpid()}: Finished")
        return game_history
    
    except Exception as e:
        # write full traceback to a per-process file so you can inspect it
        pid = os.getpid()
        with open(f"worker_error_{pid}.log", "w") as f:
            f.write("Exception in worker:\n")
            traceback.print_exc(file=f)
        raise

def play_best_game(parity):
    # print(f"{os.getpid()}: Started")
    try:
        if parity:
            
            bestAgent=AlphaZeroAgent(Colour.RED,EVAL_SIMS,training = False, model_path="models/best_model.pth")

            newAgent=AlphaZeroAgent(Colour.BLUE,EVAL_SIMS,training = False, model_path="models/new_model.pth")
            
            players = {
                Colour.RED: bestAgent,
                Colour.BLUE: newAgent,
            }
        else:

            bestAgent=AlphaZeroAgent(Colour.BLUE,EVAL_SIMS,training = False, model_path="models/best_model.pth")

            newAgent=AlphaZeroAgent(Colour.RED,EVAL_SIMS,training = False, model_path="models/new_model.pth")
            
            players = {
                Colour.RED: newAgent,
                Colour.BLUE: bestAgent,
            }
            

        current_player = Colour.RED
        board = Board()
        opponentMove = None

        while True:
            dice = [random.randint(1,6),random.randint(1,6)]

            currentAgent:AlphaZeroAgent = players[current_player]
            
            playerBoard = deepcopy(board)
            playerDice = deepcopy(dice)

            ms = currentAgent.make_move(playerBoard, playerDice, opponentMove)

            board.make_move_sequence(ms,current_player)

            if board.has_ended():
                break
            
            current_player = Colour.opposite(current_player)
        
        winner = board.get_winner()
        # print(f"{os.getpid()}: Finished")
        if winner == newAgent.colour:
            return 1
        else:
            return 0
    
    except Exception as e:
        # write full traceback to a per-process file so you can inspect it
        pid = os.getpid()
        with open(f"worker_error_{pid}.log", "w") as f:
            f.write("Exception in worker:\n")
            traceback.print_exc(file=f)
        raise

def generate_dataset():

    # explicit multiprocessing context 
    ctx = mp.get_context("spawn")

    dataset = []
    with ProcessPoolExecutor(max_workers=mp.cpu_count(), mp_context=ctx, initializer=worker_init) as ex:
        futures = [ex.submit(self_play_game) for _ in range(GAMES_PER_TRAIN)]
        for future in as_completed(futures):
            # If a worker raised, .result() will re-raise that exception here.
            game = future.result()
            dataset.append(game)

    return dataset

def evaluate_model():

    # Use a multiprocessing context explicitly
    ctx = mp.get_context("spawn")

    new_wins = 0
    with ProcessPoolExecutor(max_workers=mp.cpu_count(), mp_context=ctx, initializer=worker_init) as ex:
        futures = [ex.submit(play_best_game,i % 2) for i in range(GAMES_PER_EVAL)]
        for future in as_completed(futures):
            # If a worker raised, .result() will re-raise that exception here.
            win = future.result()
            new_wins += win

    return new_wins

if __name__ == '__main__':
    print(f"Start Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    print("Initializing...", flush=True)
    
    trainer = AlphaZeroTrainer()

    trainer.train(iterations=100)


# Itterations   TrainingGames   EvaluationGames
# 3             400             100
