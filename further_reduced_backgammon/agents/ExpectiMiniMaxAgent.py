from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from BackgammonUtils import BackgammonUtils


class ExpectiMiniMaxAgent(AgentBase):
    """
    Monte Carlo *-Minimax Search agent (Lanctot et al., IJCAI-13).

    - Max/Min nodes: players
    - Chance nodes: dice
    - Evaluation: Monte Carlo rollouts + heuristic
    """

    def __init__(
        self,
        colour: Colour,
        search_depth: int = 2,
        # 1 -> 0.0014 sec / turn
        # 2 -> 0.9 sec / turn
        # 3 -> 460 sec / turn
    ):
        super().__init__(colour)
        self.search_depth = search_depth

    def make_move(
        self,
        board: Board,
        dice: Tuple[int, int],
        opp_move: Move | None,
    ) -> Move:
        internal = BackgammonUtils.get_internal_board(board)
        player = self._player_id()

        legal_sequences = BackgammonUtils.get_legal_move_sequences(
            internal, list(dice), player
        )

        best_value = -math.inf
        best_sequence = legal_sequences[0]

        for seq in legal_sequences:
            next_board=internal.copy()
            BackgammonUtils.do_next_board_total(
                next_board, seq, player
            )

            value = self._chance_node(
                next_board,
                self._opp_player_id(),
                depth=self.search_depth-1,
            )

            if value > best_value:
                best_value = value
                best_sequence = seq

        return BackgammonUtils.get_external_movesequence(best_sequence)

    # ExpectiMiniMax Logic

    def _chance_node(
        self,
        board: np.ndarray,
        player: int,
        depth: int,
    ) -> float:
        """Chance node over dice outcomes"""

        if self._terminal(board, depth):
            return self._evaluate(board)

        value = 0.0
        dice_outcomes = [((1,1),1/36),((1,2),2/36),((1,3),2/36),((1,4),2/36),((1,5),2/36),((1,6),2/36),((2,2),1/36),((2,3),2/36),((2,4),2/36),((2,5),2/36),((2,6),2/36),((3,3),1/36),((3,4),2/36),((3,5),2/36),((3,6),2/36),((4,4),1/36),((4,5),2/36),((4,6),2/36),((5,5),1/36),((5,6),2/36),((6,6),1/36)]

        prob = 1.0 / len(dice_outcomes)

        for dice, prob in dice_outcomes:
            value += prob * self._player_node(
                board, player, dice, depth
            )

        return value

    def _player_node(
        self,
        board: np.ndarray,
        player: int,
        dice: Tuple[int, int],
        depth: int,
    ) -> float:
        legal_sequences = BackgammonUtils.get_legal_move_sequences(
            board, list(dice), player
        )

        if not legal_sequences:
            return self._chance_node(
                board, self._opp(player), depth - 1
            )

        if player == self._player_id(): # Max Node
            best = -math.inf
            for seq in legal_sequences:
                next_board=board.copy()
                BackgammonUtils.do_next_board_total(
                    next_board, seq, player
                )
                best = max(
                    best,
                    self._chance_node(
                        next_board, self._opp(player), depth - 1
                    ),
                )
            return best

        else: # Min Node
            worst = math.inf
            for seq in legal_sequences:
                next_board=board.copy()
                BackgammonUtils.do_next_board_total(
                    next_board, seq, player
                )
                worst = min(
                    worst,
                    self._chance_node(
                        next_board, self._opp(player), depth - 1
                    ),
                )
            return worst

    # Evaluation
    
    def _evaluate(self, board: np.ndarray) -> float:
        """*-Minimax leaf evaluation via Monte-Carlo rollouts"""

        if BackgammonUtils.game_over(board):
            winner = BackgammonUtils.has_won(board)
            return 1.0 if winner == self._player_id() else -1.0

        return BackgammonUtils.heuristic_evaluation(board,self._player_id())


    # Helpers

    def _terminal(self, board: np.ndarray, depth: int) -> bool:
        return (
            depth == 0             
            or BackgammonUtils.game_over(board)
        )



    def _player_id(self) -> int:
        return 1 if self.colour == Colour.RED else -1

    def _opp_player_id(self) -> int:
        return -self._player_id()

    def _opp(self, player: int) -> int:
        return -player
