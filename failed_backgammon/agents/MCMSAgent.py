"""
Monte Carlo *-Minimax Search (MCMS) agent for Backgammon.

Based on:
Lanctot et al. (2013) Monte Carlo *-minimax search, IJCAI'13.

State representation
--------------------
- Board is a numpy ndarray of shape (28,)
- Indices 0..23  : points on the board
- Board.P1BAR = 24
- Board.P2BAR = 25
- Board.P1OFF = 26
- Board.P2OFF = 27

Board values:
- +n : n checkers belonging to Player 1
- -n : n checkers belonging to Player 2

State also includes:
- player_to_move : +1 for Player 1, -1 for Player 2

Move representation
-------------------
A *move* is a list of *individual moves*.
Each individual move is a tuple:
    (from_point, to_point)

Example:
    [(24, 21), (21, 18)]  # enter from bar, then move

This matches the MCMS requirement that actions are sequences
of atomic decisions.
"""

from __future__ import annotations
import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

from BackgammonUtils import BackgammonUtils

from src.MoveSequence import MoveSequence
from src.Colour import Colour
from src.Move import Move
from src.Board import Board
from src.AgentBase import AgentBase

InternalMove = Tuple[int, int, int]
InternalMoveSequence = List[InternalMove]


@dataclass(frozen=True)
class State:
    board: np.ndarray  # shape (28,)
    player: int        # +1 (P1) or -1 (P2)

    def clone(self) -> "State":
        return State(self.board.copy(), self.player)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def bar_index(player: int) -> int:
    return Board.P1BAR if player == 1 else Board.P2BAR


def off_index(player: int) -> int:
    return Board.P1OFF if player == 1 else Board.P2OFF

# ---------------------------------------------------------------------------
# Dice and chance model
# ---------------------------------------------------------------------------

def all_dice_rolls() -> List[Tuple[int, int, float]]:
    """Return all dice outcomes with probabilities."""
    prob = 1.0 / 36.0
    outcomes = [(d1, d2, prob) for d1 in range(1, 7) for d2 in range(1, 7)]
    return outcomes

# ---------------------------------------------------------------------------
# State transition
# ---------------------------------------------------------------------------

def apply_move(state: State, move_sequence: InternalMoveSequence) -> State:
    board = state.board.copy()
    player = state.player

    BackgammonUtils.do_next_board_total(board,move_sequence,player)

    return State(board, -player)


# ---------------------------------------------------------------------------
# Evaluation and rollout policy
# ---------------------------------------------------------------------------

def terminal(state: State) -> bool:
    return state.board[Board.P1OFF] >= 15 or state.board[Board.P2OFF] <= -15


def evaluate(state: State) -> float:
    """Heuristic evaluation from Player 1's perspective."""
    if state.board[Board.P1OFF] >= 15:
        return 1.0
    if state.board[Board.P2OFF] <= -15:
        return -1.0

    # Simple pip count heuristic
    return BackgammonUtils.heuristic_evaluation(state.board, state.player)


def rollout(state: State, max_depth: int = 50) -> float:
    current = state.clone()
    for _ in range(max_depth):
        if terminal(current):
            break
        dice = [random.randint(1, 6), random.randint(1, 6)]
        move_sequences = BackgammonUtils.get_legal_move_sequences(current.board, dice, current.player)
 
        if len(move_sequences) == 1: # Skip or Single move
            move_sequence = move_sequences[0]
        else:
            move_sequence = random.choice(move_sequences)

            
        current = apply_move(current, move_sequence)
    return evaluate(current)


# ---------------------------------------------------------------------------
# MCMS Node
# ---------------------------------------------------------------------------

class MCMSNode:
    def __init__(self, state: State, depth: int):
        self.state = state
        self.depth = depth
        self.children: Dict[Tuple[int, int], Dict[Tuple[InternalMove, ...], MCMSNode]] = {}
        self.value = 0.0
        self.visits = 0


# ---------------------------------------------------------------------------
# MCMS algorithm
# ---------------------------------------------------------------------------

class MCMSAgent(AgentBase):

   

    def __init__(
            self, 
            colour, 
            simulations: int = 100, 
            max_depth: int = 3
        ):
        super().__init__(colour)

        self.simulations = simulations
        self.max_depth = max_depth

    def make_move(self, board:Board, dice:tuple[int,int], opp_move:Move | None = None) -> InternalMoveSequence:
        player = 1 if self.colour == Colour.RED else -1
        internal_board = BackgammonUtils.get_internal_board(board)
        internal_dice = tuple(dice)

        root = MCMSNode(State(internal_board,player), 0)

        for _ in range(self.simulations):
            self._simulate_root(root,internal_dice)

        # Choose best expected move
        best_move = None
        best_value = -float('inf') if player == 1 else float('inf')

        for c_dice, move_nodes in root.children.items():
            for move_key, node in move_nodes.items():
                if node.visits == 0:
                    continue
                v = node.value / node.visits
                if player == 1 and v > best_value:
                    best_value = v
                    best_move = list(move_key)
                if player == -1 and v < best_value:
                    best_value = v
                    best_move = list(move_key)

        return BackgammonUtils.get_external_movesequence(list(best_move))

    def _simulate(self, node: MCMSNode) -> float:
        global c
        if terminal(node.state) or node.depth >= self.max_depth:
            value = rollout(node.state)
            node.visits += 1
            node.value += value
            return value

        # Chance node: dice roll
        dice = (random.randint(1, 6), random.randint(1, 6))

        if dice not in node.children:
            node.children[dice] = {}
            moves = BackgammonUtils.get_legal_move_sequences(node.state.board, list(dice), node.state.player)
            for m in moves:
                next_state = apply_move(node.state, m)
                node.children[dice][tuple(m)] = MCMSNode(next_state, node.depth + 1)

        # Select move (uniform for simplicity)
        move_key, child = random.choice(list(node.children[dice].items()))
        value = self._simulate(child)

        node.visits += 1
        node.value += value
        return value

    def _simulate_root(self, node: MCMSNode, dice: tuple[int,int]) -> float:

        # Backup
        if terminal(node.state) or node.depth >= self.max_depth:
            value = rollout(node.state)
            node.visits += 1
            node.value += value
            return value

        # Expand
        if dice not in node.children:
            node.children[dice] = {}
            moves = BackgammonUtils.get_legal_move_sequences(node.state.board, list(dice),node.state.player)
            for m in moves:
                next_state = apply_move(node.state, m)
                node.children[dice][tuple(m)] = MCMSNode(next_state, node.depth + 1)


        # Select
        move_key, child = random.choice(list(node.children[dice].items()))
        value = self._simulate(child)

        node.visits += 1
        node.value += value
        return value
# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple starting position (not full standard setup)
    board = np.zeros(28, dtype=int)
    board[0] = 2
    board[11] = 5
    board[16] = 3
    board[18] = 5

    board[23] = -2
    board[12] = -5
    board[7] = -3
    board[5] = -5

    state = State(board, player=1)
    agent = MCMSAgent(simulations=200)
    move = agent.make_move(state)
    print("Chosen move:", move)
