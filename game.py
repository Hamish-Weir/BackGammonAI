from itertools import permutations
import numpy as np
import regex as re
from collections import Counter


import errors as e
 

class BackGammonGame:
    """
    Game of BackGammon in which we have:
        Doubling Dice
        home_board_length: 6
        out_board_length: 6

        Starting Configuration:
        Red Home Board  | Outer Board
        2w 0 0 0 0 6r | 0 3r 0 0 0 6w
        2r 0 0 0 0 6w | 0 3w 0 0 0 6r
        White Home Board| Outer Board
    """

    def __init__(self):
        self.doubling_dice = 0
        #                                   <--White Going | Red Going-->
        #                  White Home        Out            out            Red Home
        self.board_state = [-2,0,0,0,0,6,    0,3,0,0,0,-6,  6,0,0,0,-3,0,  -6,0,0,0,0,2]
        self.bar = {'white':0, 'red':0}
        self.off = {'white':0, 'red':0}


    def get_valid_move_sequences(self, board: int[24], player: int, dice: int[2], bar: dict[str:int], off: dict[str:int], double_die: int):
        
        #Double dice if double is rolled
        if len(dice) == 1:
            dice_seq = [dice[0]] * 4 #Idk how ill be passing doubles in yet
        elif len(dice) == 2 and dice[0] == dice[1]:
            dice_seq = [dice[0]] * 4 #Idk how ill be passing doubles in yet
        else:
            dice_seq = dice[:]  # two different dice
    
        #Get all possible dice orders (will de-dupe duplicates later)
        all_orders = set(permutations(dice_seq))
    
        direction = -1 if player == 'white' else 1
        opponent = 'black' if player == 'white' else 'white'

        def can_land_on(temp_bd, player, dest):
            """True if player may land on dest (empty, own, or exactly 1 opponent)."""
            val = temp_bd[dest]
            if player == "white": 
                return val >= -1 # 1 or less red
            else:
                return val <= 1 # 1 or less white

        def enter_from_bar_dest_index(die, player):
            """If moving a checker from the bar, destination index for a die."""
            # For white: entry points are adjusted
            # die roll: 1 | 2 | 3 | 4 | 5 | 6
            # entry   : 0 | 1 | 2 | 3 | 4 | 5 or 23 | 22 | 21 | 20 | 19 | 18
            # for       red                  and white                       respectively
            return 24 - die if player == "white" else die - 1

        def all_checkers_in_home(temp_bd, player):
            """Return True if all player's checkers are in the home board (for bearing off)."""
            if player == "white":
                # white home: 18..23
                for i, v in enumerate(temp_bd):
                    if v > 0 and i < 18:
                        return False
                return True
            else:
                # red home: 0..5
                for i, v in enumerate(temp_bd):
                    if v < 0 and i > 5:
                        return False
                return True

        def can_bear_off_with(temp_bd, point, die):
            """Check if a checker at 'point' may bear off using die."""

            dest = point + die * direction

            if player == 'white':
                # dest < 0 = bearing off
                if dest < 0:
                    pass #TODO
                
            else:  # black
                # dest > 23 = bearing off
                if dest > 23:
                    pass #TODO


