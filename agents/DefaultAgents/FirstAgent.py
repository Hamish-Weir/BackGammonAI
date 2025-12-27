from random import choice
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class FirstAgent(AgentBase):

    def __init__(self, colour):
        super().__init__(colour)

    def make_move(self, board: Board, dice:tuple[int,int], opp_move: Move | None) -> Move:
        """Makes a move based on the current board state."""

        legal_moves = board.get_legal_move_sequences(dice,self.colour)

        return legal_moves[0]

