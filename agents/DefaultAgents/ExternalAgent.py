from subprocess import PIPE, Popen

from BackgammonUtils import BackgammonUtils
from src.MoveSequence import MoveSequence
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from copy import deepcopy


class ExternalJavaAgent(AgentBase):
    """
    The class inherits from AgentBase, which is an abstract class.
    The AgentBase contains the colour property which you can use to get the agent's colour.
    You must implement the make_move method to make the agent functional.
    You CANNOT modify the AgentBase class, otherwise your agent might not function.
    """

    def __deepcopy__(self, memo):
        """
        Excludes the subprocess from deepcopy
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__dict__ = {
            k: deepcopy(v, memo)
            for k, v in self.__dict__.items()
            if k != "agent_process"
        }
        result.agent_process = None
        return result

    def __init__(self, colour: Colour):
        super().__init__(colour)
        # spawn a process that calls a compiled java NaiveAgent.class file and passes two arguments:
        # - "R" or "B" to tell the agent which colour it is
        # - 11, which is the size of the board
        self.agent_process = Popen(
            [
                "./agents/DefaultAgents/CPPAgent.exe",
                colour.get_char(),
                "11",
            ],
            stdout=PIPE,
            stdin=PIPE,
            text=True,
        )

        # self.agent_process = Popen(
        #     [
        #         "java",
        #         "-cp",
        #         "agents/DefaultAgents",
        #         "NaiveAgent",
        #         colour.get_char(),
        #         "11",
        #     ],
        #     stdout=PIPE,
        #     stdin=PIPE,
        #     text=True,
        # )


    def make_move(self, board: Board, dice:tuple[int,int], opp_move: MoveSequence | None) -> Move:
        """The game engine will call this method to request a move from the agent.
        If the agent is to make the first move, opp_move will be None.
        If the opponent has made a move, opp_move will contain the opponent's move.
        If the opponent has made a swap move, opp_move will contain a Move object with x=-1 and y=-1,
        the game engine will also change your colour to the opponent colour.

        Args:
            turn (int): The current turn
            board (Board): The current board state
            opp_move (Move | None): The opponent's last move

        Returns:
            Move: The agent's move
        """
        # translate the python objects into string representations

        internal_board = BackgammonUtils.get_internal_board(board)
        internal_opp_move = BackgammonUtils.get_internal_move(opp_move)


        board_str = ",".join([str(n) for n in internal_board])

        dice_str = f"{dice[0],dice[1]}"

        if internal_opp_move:
            moveseq_str = ','.join(['.'.join(map(str, opp_move[i])) if i < len(opp_move) else '' for i in range(4)])
        else:
            moveseq_str = ",,,"

        
        if opp_move is None:
            # should be turn 1
            command = f"START;{board_str};{dice};{moveseq_str}"
        else:
            command = f"CHANGE;{board_str};{dice};{moveseq_str}"

        # send the command to the agent process and get response
        self.agent_process.stdin.write(command + "\n")
        self.agent_process.stdin.flush()

        response = self.agent_process.stdout.readline().rstrip()

        # assuming the response takes the form "x,y" with -1,-1 if the agent wants to make a swap move
        ms_list = [(s,e,d) for move in response for s,e,d in move.split(".") ]     #ms = m;m;m;m  m = s.e.d   

        return BackgammonUtils.get_external_movesequence(ms_list)
