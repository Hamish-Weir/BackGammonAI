from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.MoveSequence import MoveSequence
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from BackgammonUtils import BackgammonUtils

# node_type = 0: Action node
# node_type = 1: Chance node

@dataclass
class MCTSNode:
    board: np.ndarray
    player: int
    parent: Optional[MCTSNode] = None
    node_move: Optional[list] = None
    children: Dict[int, MCTSNode] = field(default_factory=dict)
    visits: int = 0
    value: float = 0.0
    untried_moves: Optional[List[list]] = None

    def ucb_score(self, exploration: float = 0.8) -> float:
        if self.visits == 0:
            return float("inf")
        return (
            self.value / self.visits
            + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        )

    def __str__(self):
        return f"MCTSNode(Type: {self.node_type}, children: {len(self.children.values())}, Visits: {self.visits}, Value: {self.value}, Action {self.node_action})\n"

    def __repr__(self) -> str:
        return str(self)

class MCTSAgent(AgentBase):
    def __init__(
        self,
        colour: Colour,
        simulations: int = 5000,
        # num_rollouts: int = 0,
        # rollout_depth: int = 0,
        random_seed = None,
    ):
        super().__init__(colour)
        self.simulations = simulations
        # self.num_rollouts = num_rollouts
        # self.rollout_depth = rollout_depth
        self.random = random.Random(random_seed)

    def make_move(
        self,
        board: Board,
        opp_move: MoveSequence | None,
    ) -> MoveSequence:
        root_board = BackgammonUtils.get_internal_board(board)
        player = self._player_id()

        self.root = MCTSNode(board=root_board, player=player)

        self.root.untried_moves = BackgammonUtils.get_legal_move_sequences(root_board, player)

        for i in range(self.simulations):
            node = self._select(self.root)
            value = self._simulate(node)
            self._backpropagate(node, value)

        if self.root.children:
            best_child = max(self.root.children.values(), key=lambda c: (c.visits, c.value))
            best_move = best_child.node_action
        else: 
            raise Exception("Too Few Simulations Run")

        # for v in self.root.children.values():
        #     print(f"{v}, ",end="")
        # print("")
        return BackgammonUtils.get_external_movesequence(best_move)

    # ---------------- MCTS phases ---------------- #

    def _select(self, node: MCTSNode) -> MCTSNode:
        while not BackgammonUtils.game_over(node.board):
            if node.untried_moves:
                return self._expand(node)
            node = max(node.children.values(), key=lambda c: c.ucb_score())
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        # node_type = 0: Action node
        # node_type = 1: Chance node

        move_made = node.untried_moves.pop()

        next_board = node.board.copy()

        BackgammonUtils.do_next_board_total(
        next_board , move_made, node.player
        )

        child = MCTSNode(
            board           = next_board,
            player          = node.player,
            parent          = node,
            node_move       = move_made,
        )

        child.untried_moves = BackgammonUtils.get_legal_move_sequences(
            node.board, child.player
        )   

        node.children[id(child)] = child
        return child

    def _simulate(self, node: MCTSNode) -> float:

        if BackgammonUtils.game_over(node.board):
            winner = BackgammonUtils.has_won(node.board)
            return 1.0 if winner == node.player else -1.0
        return  BackgammonUtils.heuristic_evaluation(node.board, node.player)
    
    def _backpropagate(self, node: MCTSNode, value: float):
        while node:
            node.visits += 1
            node.value += value
            value = -value
            node = node.parent


    # ---------------- Helpers ---------------- #

    def _player_id(self) -> int:
        return 1 if self.colour == Colour.RED else -1
