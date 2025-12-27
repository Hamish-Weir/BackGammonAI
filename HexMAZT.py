from collections import deque
import os
import pickle
import random
import sys
import time
import traceback, sys, os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from datetime import datetime
import torch
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np

from agents.Group18.UnionMCTS import UnionMCTS
from agents.Group18.MyAlphaZeroNetwork import AlphaZeroNet
from agents.Group18.MyAlphaZeroAgent import AlphaZeroAgent
from agents.Group18.MyRandomAgent import RandomAgent
from agents.Group18.MyUtils import Utils as U

# parent_parent = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
# sys.path.append(parent_parent)

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

# TRAINING_SIMS = 800
# EVALUATION_SIMS = 1600
# DIRICHLET_ALPHA = 0.03
# DIRICHLET_EPSILON = 0.25
# LEARNING_RATE = 0.2
# BATCH_SIZE = 32
# NUM_BATCHES = 64
# EPOCHS = 6

# GAMES_PER_TRAIN = 200
# GAMES_PER_EVAL = 50



class AlphaZeroTrainer:

    def __init__(
        self,
        board_size          = 11,
        train_sims          = 800,
        eval_sims           = 1600,
        c_puct              = 1.4, 
        dirichlet_alpha     = 0.03, 
        dirichlet_epsilon   = 0.25, 

        learning_rate       = 0.2,
        batch_size          = 32,
        num_batches         = 64,
        epochs              = 6,
        games_per_train     = 400,
        games_per_eval      = 100,
        device              = "cpu",
        best_model_path     = "agents/Group18/models/BestAlphaZeroModel.pth",
        new_model_path      = "agents/Group18/models/NewAlphaZeroModel.pth",
        dataset_path        = "dataset.pkl"
    ):
        
        self.train_sims             = train_sims
        self.eval_sims              = eval_sims
        self.c_puct                 = c_puct
        self.dirichlet_alpha        = dirichlet_alpha
        self.dirichlet_epsilon      = dirichlet_epsilon

        self.learning_rate          = learning_rate
        self.batch_size             = batch_size
        self.num_batches            = num_batches
        self.epochs                 = epochs
        self.games_per_train        = games_per_train
        self.games_per_eval         = games_per_eval

        self.board_size             = board_size
        self.output_size            = (board_size * board_size) + 1

        self.dataset_path           = dataset_path
        self.deque_size             = games_per_train * 3
        self.dataset                = self.load_deque() # List of List of gamestates (boardstates from past 10 training iterations)

        self.device                 = device
        self.best_model_path        = best_model_path
        self.new_model_path         = new_model_path
        self.best_model             = AlphaZeroNet(board_size).to(device)
        self.new_model              = AlphaZeroNet(board_size).to(device)

        # Load best model if exists
        try:
            self.best_model.load_state_dict(torch.load(self.best_model_path,weights_only=True))
            self.new_model.load_state_dict(torch.load(self.best_model_path,weights_only=True))
            print("Loaded existing best model", flush=True)
        except:
            torch.save(self.best_model.state_dict(), self.best_model_path)
            self.new_model.load_state_dict(self.best_model.state_dict())
            print("Training new model from scratch", flush=True)

    def _train_step(self):
        flat_dataset = [pos for game in self.dataset for pos in game]

        optimizer = torch.optim.SGD(
            self.new_model.parameters(),
            lr=self.learning_rate,  # initial learning rate
            momentum=0.9,           # momentum term
            weight_decay=1e-4       # for regularization
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[100, 200, 300],
            gamma=0.1
        )

        # scheduler and optimizer most likely wont work as intended since i have serialized the training step

        self.new_model.train()
        
        for _ in range(self.epochs): # *Not technicaly epochs

            all_batches = random.choices(population = flat_dataset, k = self.batch_size*self.num_batches) # get 2048 random game state samples 
            states = torch.stack([s for (s, _, _) in all_batches]).to(self.device)
            policies = torch.tensor(np.array([p for (_, p, _) in all_batches]), dtype=torch.float).to(self.device)
            values = torch.tensor(np.array([v for (_, _, v) in all_batches]), dtype=torch.float).to(self.device)

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
            
            scheduler.step()

        self.new_model.eval()
    
    def train(self, iterations=20):
        print("Starting Training...", flush=True)
        self.best_model.load_state_dict(torch.load(self.best_model_path,weights_only=True))
        self.new_model.load_state_dict(torch.load(self.best_model_path,weights_only=True))

        for it in range(23, iterations):
            try:
                start = time.perf_counter()

                print(f"\n### TRAINING ITERATION {it+1} ###", flush=True)

                # SELF-PLAY
                print(f"Playing Self Play Games...  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}", flush=True)
                self.dataset.extend(generate_dataset(self))
                self.save_deque() # Persist games over runs

                # TRAIN NETWORK
                print(f"Training Network...         {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}", flush=True)
                self._train_step()

                # EVALUATE AGAINST BEST MODEL
                print(f"Playing Evaluation Games... {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}", flush=True)
                wins_best, wins_random, wins_mcts = evaluate_model(self)
                print(f"New model wins {wins_best}/{self.games_per_eval}", flush=True)

                if wins_best/self.games_per_eval >  0.55: # Update Best model when new is better
                    print("New best model saved!", flush=True)

                    #New model better than previous Best Model
                    torch.save(self.new_model.state_dict(), self.best_model_path)  # Overwrite Best Model Save
                    self.best_model.load_state_dict(torch.load(self.best_model_path))   # Read Best Model Save

                    # Save Evaluations against Random and MCTS
                    print(f"Win rate vs random: {wins_random}/{self.games_per_eval}", flush=True)
                    print(f"Win rate vs mcts: {wins_mcts}/{self.games_per_eval}", flush=True)
                else:
                    print("New model rejected. Continuing.", flush=True)
                    # self.new_model.load_state_dict(torch.load(self.best_model_path)) # Read Best Model Save

                with open("eval_data.txt", "a") as f:
                        f.write(f"{it},{wins_best},{wins_random},{wins_mcts},{self.games_per_eval}\n")
                f.close()

                end = time.perf_counter()
                print(f"Iteration Time: {end-start}", flush=True)

            except KeyboardInterrupt:
                break
            except:
                print("Error Occured", flush=True)

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

# ---------- Worker helper (top-level) ----------
def worker_init():
    # Called once per worker process
    # Limit threads used by numpy/pytorch/etc.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

def self_play_game(board_size, train_sims, dirichlet_alpha, dirichlet_epsilon, output_size, model_path):
    # Keep everything top-level and re-create heavy objects inside worker.
    # Wrap in try/except to capture worker errors:
   
    try:

        agent = AlphaZeroAgent(
            Colour.RED,
            sims=train_sims,
            c_puct=1.4,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            training=True,
            temp_turn=20,
            model_path=model_path
        )

        board = Board(board_size)
        turn = 1
        opp_move = None
        game_history = []

        while not board.has_ended(Colour.RED) and not board.has_ended(Colour.BLUE):
            move = agent.make_move(turn, board, opp_move)
            opp_move = move
            x, y = move.x, move.y

            state_tensor = U.encode_board(board, agent.colour)
            N_values = agent.get_N_Values()

            policy = np.zeros(output_size)
            for (mx, my), n in N_values.items():
                idx = mx * board_size + my if mx != -1 else output_size - 1
                policy[idx] = n
            policy = policy / (np.sum(policy) + 1e-12)

            game_history.append((state_tensor, policy, agent.colour))

            if x == -1 and y == -1:
                U.invert_board(board)
            else:
                board.set_tile_colour(x, y, agent.colour)
            agent.colour = Colour.opposite(agent.colour)
            turn += 1

        # convert outcome to values
        winner = board.get_winner()
        for i, (s, p, pl) in enumerate(game_history):
            value = 1 if pl == winner else -1
            game_history[i] = (s, p, value)

        return game_history

    except Exception as e:
        # write full traceback to a per-process file so you can inspect it
        pid = os.getpid()
        with open(f"worker_error_{pid}.log", "w") as f:
            f.write("Exception in worker:\n")
            traceback.print_exc(file=f)
        raise

def play_best_game(parity, board_size, eval_sims, dirichlet_alpha, dirichlet_epsilon, best_model_path, new_model_path):
    # Keep everything top-level and re-create heavy objects inside worker.
    # Wrap in try/except to capture worker errors:
   
    try:
        if parity == 1:
            new_agent = AlphaZeroAgent(
                Colour.RED,
                sims=eval_sims,
                c_puct=1.4,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
                training=True,
                model_path=new_model_path
            )

            best_agent = AlphaZeroAgent(
                Colour.BLUE,
                sims=eval_sims,
                c_puct=1.4,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
                training=True,
                model_path=best_model_path
            )
        else:
            new_agent = AlphaZeroAgent(
                Colour.BLUE,
                sims=eval_sims,
                c_puct=1.4,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
                training=True,
                model_path=new_model_path
            )

            best_agent = AlphaZeroAgent(
                Colour.RED,
                sims=eval_sims,
                c_puct=1.4,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
                training=True,
                model_path=best_model_path
            )

        to_play = bool(parity)
        players = [best_agent,new_agent]

        board = Board(board_size)
        turn = 1
        opp_move = None  

        while not board.has_ended(Colour.RED) and not board.has_ended(Colour.BLUE):
            agent = players[int(to_play)]
            move = agent.make_move(turn, board, opp_move)
            opp_move = move
            x, y = move.x, move.y

            if x == -1 and y == -1:
                U.invert_board(board)
            else:
                board.set_tile_colour(x, y, agent.colour)

            to_play = not to_play
            turn += 1

        # convert outcome to values
        winner = board.get_winner()
        if new_agent.colour == winner:
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

def play_random_game(parity, board_size, eval_sims, dirichlet_alpha, dirichlet_epsilon, new_model_path):
    # Keep everything top-level and re-create heavy objects inside worker.
    # Wrap in try/except to capture worker errors:
   
    try:
        if parity == 1:
            new_agent = AlphaZeroAgent(
                Colour.RED,
                sims=eval_sims,
                c_puct=1.4,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
                training=True,
                model_path=new_model_path
            )

            random_agent = RandomAgent(Colour.BLUE)
        else:
            new_agent = AlphaZeroAgent(
                Colour.BLUE,
                sims=eval_sims,
                c_puct=1.4,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
                training=True,
                model_path=new_model_path
            )

            random_agent = RandomAgent(Colour.RED)

        to_play = bool(parity)
        players = [random_agent,new_agent]

        board = Board(board_size)
        turn = 1
        opp_move = None
            
        while not board.has_ended(Colour.RED) and not board.has_ended(Colour.BLUE):
            agent:AlphaZeroAgent|RandomAgent = players[int(to_play)]
            move = agent.make_move(turn, board, opp_move)
            opp_move = move
            x, y = move.x, move.y

            if x == -1 and y == -1:
                U.invert_board(board)
            else:
                board.set_tile_colour(x, y, agent.colour)

            to_play = not to_play
            turn += 1

        # convert outcome to values
        winner = board.get_winner()
        if new_agent.colour == winner:
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

def play_mcts_game(parity, board_size, eval_sims, dirichlet_alpha, dirichlet_epsilon, new_model_path):
    # Keep everything top-level and re-create heavy objects inside worker.
    # Wrap in try/except to capture worker errors:
   
    try:
        if parity == 1:
            new_agent = AlphaZeroAgent(
                Colour.RED,
                sims=eval_sims,
                c_puct=1.4,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
                training=True,
                model_path=new_model_path
            )

            mcts_agent = UnionMCTS(Colour.BLUE)
        else:
            new_agent = AlphaZeroAgent(
                Colour.BLUE,
                sims=eval_sims,
                c_puct=1.4,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
                training=True,
                model_path=new_model_path
            )

            mcts_agent = UnionMCTS(Colour.RED)

        to_play = bool(parity)
        players = [mcts_agent,new_agent]

        board = Board(board_size)
        turn = 1
        opp_move = None
            


        while not board.has_ended(Colour.RED) and not board.has_ended(Colour.BLUE):
            agent:AlphaZeroAgent|UnionMCTS = players[int(to_play)]
            move = agent.make_move(turn, board, opp_move)
            opp_move = move
            x, y = move.x, move.y

            if x == -1 and y == -1:
                U.invert_board(board)
            else:
                board.set_tile_colour(x, y, agent.colour)

            to_play = not to_play
            turn += 1

        # convert outcome to values
        winner = board.get_winner()
        if new_agent.colour == winner:
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

# ---------- Trainer usage ----------
def generate_dataset(trainer: AlphaZeroTrainer):

    torch.save(trainer.new_model.state_dict(), trainer.new_model_path)

    args = (trainer.board_size, trainer.train_sims, trainer.dirichlet_alpha,
            trainer.dirichlet_epsilon, trainer.output_size, trainer.new_model_path)

    # explicit multiprocessing context 
    ctx = mp.get_context("spawn")

    dataset = []
    with ProcessPoolExecutor(max_workers=mp.cpu_count(), mp_context=ctx, initializer=worker_init) as ex:
        futures = [ex.submit(self_play_game, *args) for _ in range(trainer.games_per_train)]
        for future in as_completed(futures):
            # If a worker raised, .result() will re-raise that exception here.
            game = future.result()
            dataset.append(game)

    return dataset

def evaluate_model(trainer: AlphaZeroTrainer):

    torch.save(trainer.new_model.state_dict(), trainer.new_model_path)

    args_base = (trainer.board_size, trainer.eval_sims, trainer.dirichlet_alpha, trainer.dirichlet_epsilon, trainer.new_model_path)

    # Use a multiprocessing context explicitly
    ctx = mp.get_context("spawn")

    wins_best = 0
    with ProcessPoolExecutor(max_workers=mp.cpu_count(), mp_context=ctx, initializer=worker_init) as ex:
        futures = [ex.submit(play_best_game,i % 2, *args_base, trainer.best_model_path) for i in range(trainer.games_per_eval)]
        for future in as_completed(futures):
            # If a worker raised, .result() will re-raise that exception here.
            win = future.result()
            wins_best += win


    wins_random = 0
    wins_mcts = 0

    if wins_best*2 > trainer.games_per_eval: # Won more than half eval games

        with ProcessPoolExecutor(max_workers=mp.cpu_count(), mp_context=ctx, initializer=worker_init) as ex:
            futures = [ex.submit(play_random_game,i % 2, *args_base) for i in range(trainer.games_per_eval)]
            for future in as_completed(futures):
                # If a worker raised, .result() will re-raise that exception here.
                win = future.result()
                wins_random += win
        
        with ProcessPoolExecutor(max_workers=mp.cpu_count(), mp_context=ctx, initializer=worker_init) as ex:
            futures = [ex.submit(play_mcts_game,i % 2, *args_base) for i in range(trainer.games_per_eval)]
            for future in as_completed(futures):
                # If a worker raised, .result() will re-raise that exception here.
                win = future.result()
                wins_mcts += win

    return wins_best, wins_random, wins_mcts

if __name__ == '__main__':
    print(f"Start Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    print("Initializing...", flush=True)
    
    trainer = AlphaZeroTrainer(
        train_sims=100,
        eval_sims=200,
        games_per_train=32,
        games_per_eval=16,
        epochs=6,
    )

    torch.set_num_threads(1)
   
    trainer.train(iterations=10000)