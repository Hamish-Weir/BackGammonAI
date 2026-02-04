from random import choice
from BackgammonUtils import BackgammonUtils
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.MoveSequence import MoveSequence


class RandomAgent(AgentBase):

    def __init__(self, colour):
        super().__init__(colour)

    def make_move(self, board: Board, opp_move: Move | None) -> Move:
        """Makes a move based on the current board state."""

        internal_board = BackgammonUtils.get_internal_board(board)
        legal = BackgammonUtils.get_legal_move_sequences(internal_board,1 if self.colour == Colour.RED else -1)

        movesequence = choice(legal)
        
        return BackgammonUtils.get_external_movesequence(list(movesequence))
