from copy import deepcopy
import logging
import os
from random import randint
import sys
from time import perf_counter_ns as time
from typing import TextIO

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Dice import Dice
from src.EndState import EndState
from src.Move import Move
from src.MoveSequence import MoveSequence
from src.Player import Player

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(levelname)s]-%(asctime)s - %(message)s", level=logging.INFO
)

def format_result(
    *,
    player1_name,
    player2_name,
    winner,
    win_method,
    player_1_move_time,
    player_2_move_time,
    total_turns,
    total_time,
) -> dict[str, str]:
    return {
        "player1": player1_name,
        "player2": player2_name,
        "winner": winner,
        "win_method": win_method,
        "player1_move_time": player_1_move_time,
        "player2_move_time": player_2_move_time,
        "total_turns": total_turns,
        "total_game_time": total_time,
    }

class Game:

    MAXIMUM_TIME = 5 * 60 * 10**9

    players: dict[Colour, Player]
    _turn: int
    logDest: str | TextIO

    def __init__(
        self,        
        player1: Player,
        player2: Player,
        logDest: str | TextIO = sys.stderr,
        verbose: bool = False,
        silent: bool = False,
        timelimitOff = False
    ):
        self._turn = 0
        self._start_time = time()

        self._board = Board()

        self.current_player = Colour.RED
        self.player1 = player1
        self.player2 = player2

        self.players = {
            Colour.RED: self.player1,
            Colour.BLUE: self.player2,
        }

        if verbose:
            logger.setLevel(logging.DEBUG)

        if silent:
            logger.setLevel(logging.CRITICAL)
            logDest = os.devnull

        if logDest != sys.stderr:
            self.logDest = open(logDest, "w")
        else:
            self.logDest = logDest

        self.timelimitOff = timelimitOff

    @property
    def turn(self):
        return self._turn

    @property
    def board(self):
        return self._board
    
    def run(self):
            """Runs the match."""
        # try:
            assert issubclass(type(self.players[Colour.RED].agent), AgentBase)
            assert issubclass(type(self.players[Colour.BLUE].agent), AgentBase)
            logger.info("Game started")
            self._play()
        
        # except Exception as e:
        #     self._end_game(None)
        #     raise e
            
        # finally:
            if self.logDest != sys.stderr:
                self.logDest.close()

            print(self.board)
            return self.board.get_winner()

    def _play(self) -> dict[str, str]:
        """Main method for a match.

        The engine will keep sending status messages to agents and
        prompting them for moves until one of the following is met:

        * Win - one of the agents connects their sides of the board.
        * Illegal move - one of the agents sends an illegal message.
        * Timeout - one of the agents fails to send a message before
        the time elapses. This can also be prompted if the agent
        fails to connect.
        """
        endState = EndState.WIN
        opponentMove = None

        while True:
            self._turn += 1

            currentPlayer: Player = self.players[self.current_player]
            playerAgent = currentPlayer.agent
            logger.info(f"Turn {self.turn}: player {currentPlayer.name}")
            logger.info(f"Starting Board:\n{str(self.board)}")

            playerBoard = deepcopy(self.board)

            start = time()
            ms = playerAgent.make_move(playerBoard, opponentMove)
            end = time()


            currentPlayer.move_time += end - start

            logger.debug(
                f"Player {currentPlayer.name}; Move time: {Game.ns_to_s(currentPlayer.move_time)}s"
            )
            logger.info(f"Player {currentPlayer.name}; Move Sequence: {self.current_player}\n{ms}")
            if not self.timelimitOff:
                if currentPlayer.move_time > Game.MAXIMUM_TIME:
                    logger.info(f"Player {currentPlayer.name} timed out")
                    endState = EndState.TIMEOUT
                    break

            


            if self.is_valid_move_sequence( ms, self.board, self.current_player):
                logger.debug("Move is valid")
                self._make_move(ms)
                opponentMove = ms
            else:
                logger.info(f"Player {currentPlayer.name} made an illegal move")
                endState = EndState.BAD_MOVE
                break
            if self.board.has_ended():
                break

            logger.info(f"Turn Ending Board:\n{str(self.board)}")

            self.current_player = Colour.opposite(self.current_player)
            
        return self._end_game(endState)

    def _make_move(self, ms: Move):
        """Performs a valid move on the board, then prints its results."""

        logger.debug(f"Move made: {ms}")
        current_player = self.players[self.current_player]

        self.board.make_move_sequence(ms,self.current_player)

        print(
            f"{self.turn},{current_player.name},{self.current_player.name}{ms},{current_player.move_time}",
            file=self.logDest,
        )

    def _end_game(self, status: EndState) -> dict[str, str]:
        """Wraps up the game and prints results to shell, log and
        agents.
        """
        # calculate total time elapsed
        total_time = time() - self._start_time

        logger.info("Game over")
        logger.info(f"Final Board:\n{str(self.board)}")
        logger.info(f"Total time: {Game.ns_to_s(total_time)}s")
        winner = None

        match status:
            case EndState.WIN:
                # last move overcounts
                logger.info(f"Player {self.players[self.current_player].name} has won")
                winner = self.players[self.current_player].name
            case EndState.BAD_MOVE:
                # the player printed is the winner
                logger.info(
                    f"Player {self.players[self.current_player].name} made an illegal move"
                )
                logger.info(
                    f"Player {self.players[self.current_player.opposite()].name} has won"
                )
                winner = self.players[self.current_player.opposite()].name

            case EndState.TIMEOUT:
                # the player printed is the winner
                # last move overcounts
                logger.info(
                    f"Player {self.players[self.current_player].name} has timed out"
                )
                logger.info(
                    f"Player {self.players[self.current_player.opposite()].name} has won"
                )
                winner = self.players[self.current_player.opposite()].name
            case _:
                logger.error("Game ended abnormally")
                raise Exception("Game ended abnormally")

        for p in self.players.values():
            print(f"{p.name},{p.move_time}", file=self.logDest)
        print(f"winner,{winner},{status.name}", file=self.logDest)
        logger.info(f"Total Game Time: {Game.ns_to_s(total_time)}s")

        return format_result(
            player1_name=self.player1.name,
            player2_name=self.player2.name,
            winner=winner,
            win_method=status.name,
            player_1_move_time=Game.ns_to_s(self.player1.move_time),
            player_2_move_time=Game.ns_to_s(self.player2.move_time),
            total_turns=self._turn,
            total_time=Game.ns_to_s(total_time),
        )

    @staticmethod
    def is_valid_move_sequence(movesequence: MoveSequence, board: Board, player:Colour) -> bool:
        """Checks if the move can be made by the given player at the given
        position.
        """
        if not isinstance(movesequence, MoveSequence):
            return False

        if type(movesequence) is not type(MoveSequence()):
            return False

        

        temp_board = deepcopy(board)

        for move in movesequence.get_moves():
            src, dst, die = move.start, move.end, move.die
            
            if temp_board._tiles[src].colour != player:
                return False

            if (board._tiles[dst].colour == Colour.opposite(player) and board._tiles[dst].p_count > 1): # Cant land on Opponent stack > 1
                return False
            
            if player == Colour.RED:

                # BAR Related

                if temp_board._tiles[Board.P1BAR].p_count > 0 and src != Board.P1BAR: # Must Bar Move if Pip on Bar
                    return False
                
                if src == Board.P1BAR: # Bar Move must land Here
                    if dst != die-1:
                        return False

                # OFF Related
                
                if dst == Board.P1OFF:
                    for tile in temp_board._tiles[0:18]: # Check Bearing off Legal
                        if tile.colour == player and tile.p_count > 0:
                            return False

                    if src != 24-die: # If not exact move off

                        if src < 24 - die: # Cant move that Far
                            return False

                        for tile in temp_board._tiles[src-1:17:-1]: # Must move Furthest from OFF, therefore all further are clear
                            if tile.colour == player and tile.p_count > 0:
                                return False
                
                # Other Related
                
                if src != Board.P1BAR and dst != Board.P1OFF and dst - src != die:
                    return False
                
            else:
                if temp_board._tiles[Board.P2BAR].p_count > 0 and src != Board.P2BAR: # Must Bar Move if Pip on Bar
                    return False
                
                if src == Board.P2BAR: # Bar Move must land Here
                    if dst != 24-die:
                        return False

                # OFF Related

                if dst == Board.P2OFF:
                    for tile in temp_board._tiles[6:24]: # Check Bearing off Legal
                        if tile.colour == player and tile.p_count < 0:
                            raise("Bare Off not legal")
                            return False

                    if src != die-1: # If not exact move off

                        if  src > die: # Cant move that Far
                            return False

                        for tile in temp_board._tiles[src+1:7]: # Must move Furthest from OFF, therefore all further are clear
                            if tile.colour == player and tile.p_count > 0:
                                raise("Not Furthest")
                                return False
            
                # Other Related
                
                if src != Board.P2BAR and dst != Board.P2OFF and src - dst != die:
                    return False
                
            temp_board.make_move(move,player)
            
        return True

    @staticmethod
    def ns_to_s(t):
        """Method for standardised nanosecond to second conversion."""
        return int(t / 10**6) / 10**3