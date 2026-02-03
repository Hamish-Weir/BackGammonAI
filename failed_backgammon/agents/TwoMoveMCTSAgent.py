from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import multiprocessing as mp

import numpy as np

from src.MoveSequence import MoveSequence
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from BackgammonUtils import BackgammonUtils

# node_type = 0: Action node, no double
# node_type = 1: Action node, double, first move
# node_type = 2: Action node, double, second move
# node_type = 3: Chance node

@dataclass
class MCTSNode:
    board: np.ndarray
    player: int
    node_type: int
    parent: Optional[MCTSNode] = None
    node_action: Optional[list] = None
    children: Dict[int, MCTSNode] = field(default_factory=dict)
    visits: int = 0     # N
    value: float = 0.0  # W
    untried_moves: Optional[List[list]] = None

    def ucb_score(self, exploration: float = 1.4) -> float:
        if self.visits == 0:
            return float("inf")
        return (
            self.value / self.visits
            + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        )
    
    def chance_score(self) -> float:
        if self.visits == 0:
            return float("inf")
        return (
            1/(2*self.visits) if self.node_action in [[1,1],[2,2],[3,3],[4,4],[5,5],[6,6]] else 1/self.visits # visit non-doubles twice as much
        )
    
    def node_score(self):
        if self.node_type == 0 or self.node_type == 2:
            return self.chance_score()
        else:
            return self.ucb_score()

    def __str__(self):
        return f"MCTSNode(Type: {self.node_type}, children: {len(self.children.values())}, Visits: {self.visits}, Value: {self.value}, Action {self.node_action})\n"

    def __repr__(self) -> str:
        return str(self)

class TwoMoveMCTSAgent(AgentBase):
    def __init__(
        self,
        colour: Colour,
        simulations: int = 5000,
        num_rollouts: int = 0,
        rollout_depth: int = 0,
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
        dice: tuple[int, int],
        opp_move: MoveSequence | None,
    ) -> MoveSequence:
        root_board = BackgammonUtils.get_internal_board(board)
        player = self._player_id()

        if dice[0] != dice[1]:
            self.root = MCTSNode(board=root_board, player=player, node_type=0, node_action=dice)
        else:
            self.root = MCTSNode(board=root_board, player=player, node_type=1, node_action=dice)
        self.root.untried_moves = BackgammonUtils.TwoMove_get_legal_move_sequences(root_board, list(dice), player)

        for i in range(self.simulations):
            node = self._select(self.root)
            value = self._simulate(node)
            self._backpropagate(node, value)

        if self.root.children:
            if self.root.node_type == 0:
                best_child = max(self.root.children.values(), key=lambda c: (c.visits, c.value))
                best_move = best_child.node_action
            else:
                first_best_child = max(self.root.children.values(), key=lambda c: (c.visits, c.value))
                if first_best_child.children:
                    second_best_child = max(first_best_child.children.values(), key=lambda c: (c.visits, c.value))
                    best_move = first_best_child.node_action + second_best_child.node_action
                else:
                    if first_best_child.untried_moves:
                        best_move = first_best_child.node_action + self.random.choice(first_best_child.untried_moves)
                    else:
                        best_move = first_best_child.node_action
        else: 
            raise Exception("Too Few Simulations Run")

        return BackgammonUtils.get_external_movesequence(best_move)

    # ---------------- MCTS phases ---------------- #

    def _select(self, node: MCTSNode) -> MCTSNode:
        while not BackgammonUtils.game_over(node.board):
            if node.untried_moves:
                return self._expand(node)
            node = max(node.children.values(), key=lambda c: c.node_score())
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        # node_type = 0: Action node, no double
        # node_type = 1: Action node, double, First
        # node_type = 2: Action node, double, Second
        # node_type = 3: Chance node

        action_made = node.untried_moves.pop()

        if node.node_type == 0:     
            # Action node, not Double -> next 3
            # self  node_action => Dice
            # Child node_action => Move Sequence
            #       action_made => Move Sequence

            next_board = node.board.copy()

            BackgammonUtils.do_next_board_total(
            next_board , action_made, node.player
            )

            child = MCTSNode(
                board           = next_board,
                player          = node.player,
                parent          = node,
                node_action     = action_made,
                node_type       = 3
            )

            child.untried_moves = [[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[2,2],[2,3],[2,4],[2,5],[2,6],[3,3],[3,4],[3,5],[3,6],[4,4],[4,5],[4,6],[5,5],[5,6],[6,6]]
            
        elif node.node_type == 1:   
            # Action node, double, first move -> next 2
            # self  node_action => Dice
            # Child node_action => Move Sequence
            #       action_made => Move Sequence

            next_board = node.board.copy()

            BackgammonUtils.do_next_board_total(
            next_board , action_made, node.player
            )

            child = MCTSNode(
                board           = next_board,
                player          = node.player,
                parent          = node,
                node_action     = action_made,
                node_type       = 2
            )

            child.untried_moves = BackgammonUtils.TwoMove_get_legal_move_sequences(
                    next_board, node.node_action, child.player
                )
            
        elif node.node_type == 2:   
            # Action node, double, second move -> next 3
            # self  node_action => Move Sequence
            # Child node_action => Move Sequence
            #       action_made => Move Sequence
            next_board = node.board.copy()

            BackgammonUtils.do_next_board_total(
            next_board , action_made, node.player
            )

            child = MCTSNode(
                board           = next_board,
                player          = node.player,
                parent          = node,
                node_action     = action_made,
                node_type       = 3
            )

            child.untried_moves = [[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[2,2],[2,3],[2,4],[2,5],[2,6],[3,3],[3,4],[3,5],[3,6],[4,4],[4,5],[4,6],[5,5],[5,6],[6,6]]
            
        elif node.node_type == 3:   
            # Chance node  -> next 0 or 1
            # self  node_action => Move Sequence
            # Child node_action => Dice
            #       action_made => Dice
            
            if action_made[0] == action_made[1]: # chance node, therefor node_action is dice
                nt = 1
            else:
                nt = 0
            
            child = MCTSNode(
                board           = node.board, # no move made, so children can share board object
                player          =-node.player, # swap player on exiting chance node
                parent          = node,
                node_action     = action_made,
                node_type       = nt
            )

            child.untried_moves = BackgammonUtils.TwoMove_get_legal_move_sequences(
                    node.board, action_made, child.player
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
            if node.node_type == 0: # Action node, not Double -> parent 3
                node.visits += 1
                node.value += value
                value = -value
                node = node.parent

            elif node.node_type == 1: # Action node, Double, First -> parent 3
                node.visits += 1
                node.value += value
                value = -value
                node = node.parent

            elif node.node_type == 2: # Action node, Double, Second -> parent 2
                node.visits += 1
                node.value += value
                value = value
                node = node.parent

            elif node.node_type == 3: # Chance node -> parent 0 or 2
                node.visits += 1
                node.value += value
                value = value
                node = node.parent


    # ---------------- Helpers ---------------- #

    def _player_id(self) -> int:
        return 1 if self.colour == Colour.RED else -1

    def _random_dice(self) -> list[int]:
        d1 = self.random.randint(1, 6)
        d2 = self.random.randint(1, 6)
        return [d1, d2]