from random import choice, randint
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

        player = 1 if self.colour == Colour.RED else -1

        movesequence = []
        for i in range(2):
            legal_moves = BackgammonUtils.get_legal_moves(internal_board,randint(1,6),player)
            if legal_moves:
                move = choice(legal_moves)
            else:
                break

            movesequence.append(move)
            BackgammonUtils.do_next_board_partial(internal_board,move,player)

        return BackgammonUtils.get_external_movesequence(list(movesequence))
