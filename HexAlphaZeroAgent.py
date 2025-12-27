from __future__ import annotations

from copy import deepcopy
from random import choice
import sys
import threading
import time

import numpy as np

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from networks.AlphaZeroNetwork import AlphaZeroNet
from BackgammonUtils import BackgammonUtils as U

sec5 = 4 * 10**9

#  AlphaZero MCTS Node
class MCTSNode:

    def __init__(self, model, board:np.ndarray, turn:int, player:int, opp_move:tuple[int,int]|None = None, parent:MCTSNode|None = None):
        
        self.opp_move   = opp_move                      # Move taken to reach this node
        self.parent     = parent
        if self.parent:
            self.parent.children[self.opp_move] = self  # Add self to tree

        self.board      = board
        self.turn       = turn
        self.player     = player

        self.winner     = U.get_winner(board)           # Winning Player at this node

        if not self.winner:
            self.legal      = U.get_legal_moves(board,turn) # Legal Moves from this node
        else:
            self.legal = []

        self.children   = {}                            # Move: MCTSNode
        self.v, self.P  = self._get_val_pol(model)      # Board Value, Move: policy (from neural network)
        self.N          = defaultdict(int)              # Move: child visit count
        self.W          = defaultdict(float)            # Move: child total value
        self.Q          = defaultdict(float)            # Move: child mean value
        
    def _get_val_pol(self, model):
        encoded = U.encode_board(self.board, self.player).to(torch.device("cpu")).unsqueeze(0)

        # predict
        with torch.no_grad():
            value, policy = model(encoded)
            value = float(value.item())
            policy = policy[0].cpu().numpy()

        # convert flat policy into dict
        pol = {}
        for x, y in self.legal:
            if x==-1 and y==-1:
                idx = 121
            else:
                if self.player == 1:
                    idx = x * 11 + y
                else:
                    idx = y * 11 + x
                
            pol[(x, y)] = float(policy[idx])

        # normalize
        s = sum(pol.values()) + 1e-12
        for k in pol:
            pol[k] /= s

        return value, pol

    def _best_PUCT_move(self, c_puct:float):
        best_score, best_move = -1e9, None
        for (x, y) in self.legal:
            move = (x, y)
            P = self.P.get(move, 0.0)
            N_total = sum(self.N.values()) + 1
            score = self.Q[move] + c_puct * P * math.sqrt(N_total) / (1 + self.N[move])
            if score > best_score:
                best_score, best_move = score, move
        return best_move
        
    def select(self, c_puct=1.4) -> tuple[tuple[int,int], MCTSNode]|tuple[tuple[int,int], None]:
        if not self.winner:
            # Get Best Move
            best_move = self._best_PUCT_move(c_puct)

            # Get Next Node
            node = self.children.get(best_move,None) # Get next Child or Unexpanded Node

            return best_move, node # Move that lead to this node, Node
        
        else: 
            raise Exception("Cant Select on Terminal Node")

    def backup(self,value: int, move: tuple[int,int]):
        self.N[move] += 1
        self.W[move] += value
        self.Q[move] = self.W[move] / self.N[move]
        
        if self.parent:
            self.parent.backup(-value, self.opp_move)

    def size(self):
        return 1 + sum([child.size() for child in self.children.values()])
    
    def clean_tree_root(self, move):
        if self.parent:
            stack = [
                value
                for key, value in self.parent.children.items()
                if key is None or key != move
            ]
        
            self.parent.children.clear()
            self.parent.parent = None
            self.parent=None
        
            while stack:
                node = stack.pop()
                stack.extend(node.children.values())
                node.children.clear()
                node.parent = None
  
    def clean_tree(self):
        stack = [self]
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            node.children.clear()
            node.parent = None

# Alpha Zero Agent
class AlphaZeroAgent(AgentBase):

    _board_size: int = 11

    def __init__(
            self, 
            colour: Colour, 
            sims = 300, 
            c_puct=1.4, dirichlet_alpha=0.03, dirichlet_epsilon=0.25, temp_turn = 0,  training = True, model_path="agents/Group18/models/BestAlphaZeroModel.pth"):
        super().__init__(colour)

        self.training = training # turns temp selection on or off for the first 20 moves
        self.sims = sims
        self.c_puct = c_puct
        self.dirichlet_alpha=dirichlet_alpha
        self.dirichlet_epsilon=dirichlet_epsilon
        self.temp_turn = temp_turn

        self.root = None

        self.device = "cpu"
        self.model = AlphaZeroNet(self._board_size)
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)

    def make_move(self, turn:int, board:Board, opp_move:Move|None) -> Move:
        
        def cleanup():
            self.root.clean_tree_root((x, y))

        # Reset Tree (Or Navigate to next board and prune rest of the tree)

        internal_board = np.array([[1 if tile.colour == Colour.RED else -1 if tile.colour == Colour.BLUE else 0  for tile in row] for row in board.tiles]) # 1d array of board
        internal_colour = 1 if self.colour == Colour.RED else -1

        if not self.root or not opp_move: # if first turn or opponent swap move
            self.root = MCTSNode(self.model,internal_board,turn,internal_colour,None,None)
        elif opp_move.is_swap(): # internal representation is incorrect after swap, so wipe it
            self.root = MCTSNode(self.model,internal_board,turn,internal_colour,None,None)
        else:
            x, y = opp_move.x, opp_move.y
            self.root = self.root.children.get((x,y),None)
            if self.root:
                threading.Thread(target=cleanup, daemon=True).start()
            else: # Unreached State (opponent has made what model thinks is a "bad move")
                self.root = MCTSNode(self.model,internal_board,turn,internal_colour,None,None)


        # self.root = MCTSNode(self.model,internal_board,turn,internal_colour,None,None)     

        # Run Tree Search
        self.run()

        x, y = self._select_move_from_root(1 if turn<=self.temp_turn else 0)

        # self.root = self.root.clean_tree_root((x,y)) # Should never be none, but doesnt hurt to check

        self.root = self.root.children.get((x,y),None)
        if self.root:
            threading.Thread(target=cleanup, daemon=True).start() # Clean Tree asynchronously


        return Move(x, y)
    
    def run(self):
        """
        Run self.sims simulations from the given root (board, turn, player).
        """
        if not self.training:
            start = time.time_ns()
        
        self._add_dirichlet_noise_to_root()

        #Run rest of Simulations
        for i in range(self.sims):
            if not self.training: # Move Time Limit
                end = time.time_ns()
                if end-start>sec5:
                    break

            node = self.root

            #Select
            while node and not node.winner: # While Node is Expanded and not Terminal
                parent = node
                best_move, node = node.select(self.c_puct)

            #Expand
            try:
                value, node = self.expand(best_move, node, parent)
            except:
                print(best_move, flush=True)
                raise Exception()
            #Backup
            parent.backup(-value,best_move)

            
    def expand(self, best_move: tuple[int,int], node: MCTSNode|None, parent:MCTSNode|None):

        if node: # Previously Reached Node
            if node.winner: # Previously Reached Terminal State
                if node.winner == node.player: # Always False, Later Fix #TODO
                    return 1.0, node
                else:
                    return -1.0, node
            else:
                raise Exception("Previously reached nodes should always be Terminal")
            
        else:
            next_board   = U.next_board(parent.board, best_move, parent.player)
            next_player  = -parent.player
            next_turn    = parent.turn+1

            node = MCTSNode(self.model,next_board,next_turn,next_player,best_move,parent)

            if node.winner: # New Terminal State
                if node.winner == node.player: # Always False, Later Fix #TODO
                    return 1.0, node
                else:
                    return -1.0, node
                
            else:   # New Non-Terminal State
                return node.v, node

    def get_N_Values(self):
        return self.root.N

    def _select_move_from_root(self, tau:float = 0.0):
        """
        Select a move from node using visit counts and temperature.
        If temperature == 0: pick the move with highest visit count (deterministic).
        If temperature > 0: sample with probabilities proportional to N^(1/temperature).
        """
        moves = self.root.legal
        visit_counts = np.array([self.root.N.get(m) for m in moves], dtype=np.float64)
        policy_counts = np.array([self.root.P.get(m) for m in moves], dtype=np.float64)

        if not moves:
            raise Exception("Root should always have Legal Moves") #Should never happen

        if tau <= 0 or not self.training:   # Deterministic: Pick most visited Move
            # argmax by visit count
            max_visits = np.max(visit_counts)
            candidates = np.where(visit_counts == max_visits)[0]

            if len(candidates) == 1:    # if only one candidate, use it
                best_index = candidates[0]
            else:                       # else revert to policy_counts
                best_index = candidates[np.argmax(policy_counts[candidates])]

            return moves[best_index]
        else:                               # Scholastic: Pick using Temperature based on visit count
            adjusted = visit_counts ** (1.0 / tau)
            probs = adjusted / (adjusted.sum() + 1e-12)
            choice_idx = np.random.choice(len(moves), p=probs)
            return moves[choice_idx]

    def _add_dirichlet_noise_to_root(self):
        """
        Add Dirichlet noise to the root node's policy. This mixes the original policy
        with a Dirichlet sample over legal moves.
        """

        # list legal moves in deterministic order to align with node.P keys

        # extract original probs in same order as legal
        L = len(self.root.legal)

        orig_probs = np.array([self.root.P.get(m, 0.0) for m in self.root.legal], dtype=np.float64)
        
        # normalize just in case
        s = orig_probs.sum()
        if s <= 0:
            # fallback uniform
            orig_probs = np.ones_like(orig_probs) / L
        else:
            orig_probs = orig_probs / s

        # sample dirichlet noise
        if self.dirichlet_alpha > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * L)
            mixed = (1 - self.dirichlet_epsilon) * orig_probs + self.dirichlet_epsilon * noise
        else:
            # self.dirichlet_alpha = 0 -> no noise
            mixed = orig_probs

        # write back into node.P for the legal moves
        for m, p in zip(self.root.legal, mixed):
            self.root.P[m] = float(p)

        # ensure normalization
        s2 = sum(self.root.P.values()) + 1e-12
        for k in self.root.P:
            self.root.P[k] = self.root.P[k] / s2
