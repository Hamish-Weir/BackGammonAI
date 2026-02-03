
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.Colour import Colour
from src.Tile import Tile
from src.Move import Move
from src.MoveSequence import MoveSequence



class Board():

    P1BAR = 24
    P2BAR = 25

    P1OFF = 26
    P2OFF = 27

    _tiles: tuple[Tile] # array of
    _winner: Colour|None

    def __init__(self):
        self._tiles = (
            # Blue Home
            Tile(0,     2,  Colour.RED  ),
            Tile(1,     0,  None        ),
            Tile(2,     0,  None        ),
            Tile(3,     0,  None        ),
            Tile(4,     0,  None        ),
            Tile(5,     5,  Colour.BLUE ),
            # Blue Out
            Tile(6,     0,  None        ),
            Tile(7,     3,  Colour.BLUE ),
            Tile(8,     0,  None        ),
            Tile(9,     0,  None        ),
            Tile(10,    0,  None        ),
            Tile(11,    5,  Colour.RED  ),
            # Red Out
            Tile(12,    5,  Colour.BLUE ),
            Tile(13,    0,  None        ),
            Tile(14,    0,  None        ),
            Tile(15,    0,  None        ),
            Tile(16,    3,  Colour.RED  ),
            Tile(17,    0,  None        ),
            # Red Home
            Tile(18,    5,  Colour.RED  ),
            Tile(19,    0,  None        ),
            Tile(20,    0,  None        ),
            Tile(21,    0,  None        ),
            Tile(22,    0,  None        ),
            Tile(23,    2,  Colour.BLUE ),

            Tile(24,    0,  Colour.RED  ), # self.P1BAR = 24
            Tile(25,    0,  Colour.BLUE ), # self.P2BAR = 25
            Tile(26,    0,  Colour.RED  ), # self.P1OFF = 26
            Tile(27,    0,  Colour.BLUE ), # self.P2OFF = 27
            )

        self._winner = None

    def __str__(self) -> str:
        RED = "\033[91m"
        BLUE = "\033[0;34m"
        YELLOW = "\033[1;33m"
        END = "\033[0m"

        BOARD_COLOUR = YELLOW
        
        def get_spaced_str(list: list[int]):
            return " ".join([f"{n:4d}" for n in list])

        def get_spaced_str_board(list: list[Tile]):
            list2 = []
            for n in list:
                if n.colour == None:
                    list2.append(f"    ")
                elif n.colour == Colour.BLUE:
                    list2.append(f"{BLUE}{n.p_count:4d}{END}")
                else:
                    list2.append(f"{RED}{n.p_count:4d}{END}")
            return " ".join(list2)

        board_str = f"""                {BLUE}BLUE Home                          {BLUE}Out
        {BOARD_COLOUR}OFF     {get_spaced_str([1, 2, 3, 4, 5, 6])}      {get_spaced_str([7, 8, 9, 10, 11, 12])}     BAR
                {BOARD_COLOUR}+----+----+----+----+----+----+    +----+----+----+----+----+----+
        {BLUE}{self._tiles[self.P2OFF].p_count:4d}    {get_spaced_str_board(self._tiles[0:6])}      {get_spaced_str_board(self._tiles[6:12])}   {BLUE}{self._tiles[self.P2BAR].p_count:4d}
                {BOARD_COLOUR}+----+----+----+----+----+----+    +----+----+----+----+----+----+
        {RED}{self._tiles[self.P1OFF].p_count:4d}    {get_spaced_str_board(self._tiles[23:17:-1])}      {get_spaced_str_board(self._tiles[17:11:-1])}   {RED}{self._tiles[self.P1BAR].p_count:4d}
                {BOARD_COLOUR}+----+----+----+----+----+----+    +----+----+----+----+----+----+
                {get_spaced_str([24, 23, 22, 21, 20, 19])}      {get_spaced_str([18, 17, 16, 15, 14, 13])}
                {RED}Red Home                           {RED}Out{END}
        """
        
        return board_str

    def __eq__(self, value: object) -> bool:
            if not isinstance(value, Board):
                return False

            for i in range(28):
                if self.tiles[i].colour != value.tiles[i].colour:
                    return False
                if self.tiles[i].p_count != value.tiles[i].p_count:
                    return False

            return True

    def has_ended(self) -> bool:
        if self._tiles[self.P1OFF].p_count == 15:
            self._winner = Colour.RED
            return True
        elif self._tiles[self.P2OFF].p_count == 15:
            self._winner = Colour.BLUE
            return True
        return False

    def get_winner(self) -> Colour:
        return self._winner   
    
    @property
    def tiles(self) -> NDArray[np.object_]:
        return self._tiles

    def _move_piece(self, src, dst, player) -> None:

        if self._tiles[src].colour == player:
            if (self._tiles[dst].colour == Colour.opposite(player) and self._tiles[dst].p_count == 1): # Land on Opponent

                self._tiles[src].p_count -= 1
                if player == Colour.RED:
                    self._tiles[self.P2BAR].p_count += 1
                elif player == Colour.BLUE:
                    self._tiles[self.P1BAR].p_count += 1

                self._tiles[dst].p_count = 1
                self._tiles[dst].colour = player

            elif self._tiles[dst].colour == None:       # Land on Empty

                self._tiles[src].p_count -= 1

                self._tiles[dst].p_count = 1
                self._tiles[dst].colour = player

            elif self._tiles[dst].colour == player:     # Land on Self

                self._tiles[src].p_count -= 1

                self._tiles[dst].p_count += 1

            else:
                raise Exception("Bad Destination")


            if self._tiles[src].p_count == 0 and (src != self.P1BAR and src != self.P2BAR):
                self._tiles[src].colour = None
        else:
            raise Exception("Bad Source")

    def make_move(self, move:Move, player:Colour) -> None:
        if move.is_valid(player):
            self._move_piece(move.start,move.end,player)
        else:

            raise Exception(f"Invalid Move: {move}")
    
    def make_move_sequence(self, move_sequence:MoveSequence, player:Colour) -> None:
        if move_sequence.first:
            self.make_move(move_sequence.first,     player)
            if move_sequence.second:
                self.make_move(move_sequence.second,    player)
    
    def get_legal_moves(self, die: int, player: Colour):
        board = np.array([tile.p_count if tile.colour == Colour.RED else -tile.p_count for tile in self._tiles],dtype=np.int8)

        moveSet = set()

        if bool(board[self.P1OFF]==15 or board[self.P2OFF]==-15):
            return moveSet

        if player == Colour.BLUE:
            if board[self.P2BAR] < 0: # Piece on Bar
                end_pip = 24-die
                if board[end_pip] < 2:
                    moveSet.add(Move(int(self.P2BAR),int(end_pip),int(die))) 
            else: # No Piece on Bar & game not over
                if sum(board[6:24] < 0) == 0: # Bearing off is legal
                    start_pip = die - 1
                    if (board[start_pip] < 0): # Exact Throw; move off
                        moveSet.add(Move(int(start_pip),int(self.P2OFF),int(die)))

                    else: # No Exact Throw, Bear off furthest if possible
                        start_pip = np.max(np.where(board[0:7]<0)[0]) # Get Furthest Piece
                        if start_pip < die: # If Can Move Off; move off
                            moveSet.add(Move(int(start_pip),int(self.P2OFF),int(die)))
                
                # all other legal options
                possible_start_pips = np.where(board[0:24]<0)[0]
                
                start_pips = possible_start_pips
                end_pips = start_pips - die

                mask = (end_pips >= 0) & (end_pips < 24) & ((board[end_pips.clip(0,23)]) < 2)

                masked_start_pips = start_pips[mask]
                masked_end_pips = end_pips[mask]

                new_moves = [Move(int(start_pip),int(end_pip),int(die)) for start_pip, end_pip in zip(masked_start_pips,masked_end_pips)]
                moveSet.update(new_moves)

        elif player == Colour.RED:
            if board[self.P1BAR] > 0: # Piece on Bar
                end_pip = die-1
                if board[end_pip] > -2:
                    moveSet.add(Move(int(self.P1BAR),int(end_pip),int(die))) 
            else: # No Piece on Bar & game not over
                if sum(board[0:18] > 0) == 0: # Bearing off is legal
                    start_pip = 24 - die
                    if (board[start_pip] > 0): # Exact Throw; move off
                        moveSet.add(Move(int(start_pip),int(self.P1OFF),int(die)))

                    else: # No Exact Throw, Bear off furthest if possible
                        start_pip = 23 - np.max(np.where(board[23:17:-1]>0)[0]) # Get Furthest Piece idx
                        if start_pip > 24-die: # If Can Move Off; move off
                            moveSet.add(Move(int(start_pip),int(self.P1OFF),int(die)))
                
                # all other legal options
                possible_start_pips = np.where(board[0:24]>0)[0]
                
                start_pips = possible_start_pips
                end_pips = start_pips + die

                mask = (end_pips >= 0) & (end_pips < 24) & ((board[end_pips.clip(0,23)]) > -2)

                masked_start_pips = start_pips[mask]
                masked_end_pips = end_pips[mask]

                new_moves = [Move(int(start_pip),int(end_pip),int(die)) for start_pip, end_pip in zip(masked_start_pips,masked_end_pips)]
                moveSet.update(new_moves)
                
        return list(moveSet)

    def get_legal_move_sequences(self,dice: tuple[int,int], player:Colour):

        def next_board_and_moves(board:Board,move:Move,die:int,player:Colour):
            temp_board = deepcopy(board)
            temp_board.make_move(move,player)
            possible_moves = temp_board.get_legal_moves(die, player)
            return temp_board, possible_moves


        initial_temp_board = deepcopy(self)
        moveSequencesSet = set()

        if dice[0] == dice[1]: # For Doubles
            # try using the first dice, then the second dice, then third, then fourth (i hate doing this but oh well)
            possible_first_moves = initial_temp_board.get_legal_moves(dice[0], player)
            if possible_first_moves:
                for m1 in possible_first_moves:
                    temp_board1, possible_second_moves = next_board_and_moves(initial_temp_board,m1,dice[0],player)
                    if possible_second_moves:
                        for m2 in possible_second_moves:
                            temp_board2,possible_third_moves = next_board_and_moves(temp_board1,m2,dice[0],player)
                            if possible_third_moves:
                                for m3 in possible_third_moves:
                                    _,possible_fourth_moves = next_board_and_moves(temp_board2,m3,dice[0],player)
                                    if possible_fourth_moves:
                                        for m4 in possible_fourth_moves:
                                            moveSequencesSet.add(MoveSequence(m1,m2,m3,m4))
                                    else:
                                        moveSequencesSet.add(MoveSequence(m1,m2,m3))
                            else:
                                moveSequencesSet.add(MoveSequence(m1,m2))
                    else:
                        moveSequencesSet.add(MoveSequence(m1))
            else:
                moveSequencesSet.add(MoveSequence())

        else: # For Non-Doubles
            # try using the first die, then the second die
            possible_first_moves = initial_temp_board.get_legal_moves(dice[0], player)
            for m1 in possible_first_moves:
                _, possible_second_moves = next_board_and_moves(initial_temp_board, m1, dice[1], player)
                if possible_second_moves:
                    for m2 in possible_second_moves:
                        moveSequencesSet.add(MoveSequence(m1,m2))
                else:
                    moveSequencesSet.add(MoveSequence(m1))
                
            # try using the second die, then the first die
            possible_first_moves = initial_temp_board.get_legal_moves(dice[1], player)
            for m1 in possible_first_moves:
                _, possible_second_moves = next_board_and_moves(initial_temp_board, m1, dice[0], player)
                if possible_second_moves:
                    for m2 in possible_second_moves:
                        moveSequencesSet.add(MoveSequence(m1,m2))
                else:
                    moveSequencesSet.add(MoveSequence(m1))
                    
            # if there's no moves available:
            if len(moveSequencesSet)==0: 
                moveSequencesSet.add(MoveSequence())
            
        return sorted(list(moveSequencesSet))