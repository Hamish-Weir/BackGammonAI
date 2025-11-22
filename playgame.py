from copy import copy
import os
import time
import cProfile
import pstats

import numpy as np
from agent_mcts import Agent_MCTS
import backgammon_helper as bg
from abstract_agent import Agent
from agent_player import Agent_Player
from agent_random import Agent_Random
from random import randrange
import test


class BackGammon:
    def __init__(self,player1:Agent,player2:Agent):
        self.__players = [player1,player2][::-1]
        self.__current_player = 1
        self.board = bg.init_board()

    def next_player(self):
        self.__current_player = -self.__current_player
        
    def get_current_agent(self):
        p = 1 if self.__current_player == 1 else 0
        return self.__players[p]

    def printturn(self,dice):

        remaining_dice_str = "".join(f"{die:2d}" for die in dice)
        print("---------------------------------------------------------------------------------------------------")
        self.printboard()
        print(f"         Die Rolled:{remaining_dice_str}")

    def printmovesequence(self, move_sequence):
        print(f"Move Sequence Taken: ",end='')
        # This is an goofy generatorc

        print([(("bar", int(dest)+1) if (src == bg.P1BAR or src == bg.P2BAR) else (int(src)+1, int(dest)+1)) for (src, dest) in move_sequence]
)
        
    def printwinner(self):
        RED = "\033[91m"
        WHITE = "\033[0m"
        YELLOW = "\033[1;33m"
        

        # player    winner  winner
        #   1       1       white
        #   1       -1      red
        #   -1      1       red
        #   -1      -1      white

        winner = "White" if (bg.get_winner(self.board) == 1) else "Red"
        winner_c = WHITE if (bg.get_winner(self.board) == 1) else RED
        print()
        print(f"         {YELLOW}GAME OVER{WHITE}")
        print(f"         {YELLOW}Winner: {winner_c}{winner}{WHITE}")
        self.printboard()

    def printboard(self):
        RED = "\033[91m"
        WHITE = "\033[0m"
        YELLOW = "\033[1;33m"

        BOARD_COLOUR = YELLOW
        current_player = self.__current_player
        PLAYER_COLOUR = WHITE if current_player == 1 else RED
        TO_PLAY = "White" if current_player == 1 else "Red"

        P1OFF = bg.P1OFF
        P2OFF = bg.P2OFF
        P1BAR = bg.P1BAR
        P2BAR = bg.P2BAR
        
        def get_spaced_str(list):
            return " ".join([f"{n:4d}" for n in list])

        def get_spaced_str_board(list):
            list2 = []
            for n in list:
                if n == 0:
                    list2.append(f"    ")
                elif n > 0:
                    list2.append(f"{WHITE}{n:4d}{WHITE}")
                else:
                    list2.append(f"{RED}{abs(n):4d}{WHITE}")
            return " ".join(list2)

        board_str = f"""                {WHITE}White Home                         {WHITE}Out
        {BOARD_COLOUR}OFF     {get_spaced_str([1, 2, 3, 4, 5, 6])}      {get_spaced_str([7, 8, 9, 10, 11, 12])}     BAR
                {BOARD_COLOUR}+----+----+----+----+----+----+    +----+----+----+----+----+----+
        {WHITE}{self.board[P1OFF]:4d}    {get_spaced_str_board(self.board[0:6])}      {get_spaced_str_board(self.board[6:12])}   {WHITE}{self.board[P1BAR]:4d}
                {BOARD_COLOUR}+----+----+----+----+----+----+    +----+----+----+----+----+----+
        {RED}{-self.board[P2OFF]:4d}    {get_spaced_str_board(self.board[23:17:-1])}      {get_spaced_str_board(self.board[17:11:-1])}   {RED}{-self.board[P2BAR]:4d}
                {BOARD_COLOUR}+----+----+----+----+----+----+    +----+----+----+----+----+----+
                {get_spaced_str([24, 23, 22, 21, 20, 19])}      {get_spaced_str([18, 17, 16, 15, 14, 13])}
                {RED}Red Home                           {RED}Out

            {BOARD_COLOUR}To Play: {PLAYER_COLOUR}{TO_PLAY}{WHITE}"""
        
        print(board_str)
          
    def game_over(self):
        return not (bg.get_winner(self.board) == 0)

    def play(self, silent=False):
        self.board = bg.init_board() 

        # Highest roll starts first, cant start on double
        d1 = randrange(1,7)
        d2 = randrange(1,7)
        while d1 == d2:
            d1 = randrange(1,7)
            d2 = randrange(1,7)

        if d1 < d2:
            self.next_player()

        current_player = self.get_current_agent()
        dice = (d1,d2)

        if not silent:
                self.printturn(dice)

        valid_move_sequences, _ = bg.get_legal_move_sequences(self.board,dice, self.__current_player)
        move_sequence = current_player.get_next_move(self.board, dice, self.__current_player)
        valid_bytes = {seq.tobytes() for seq in valid_move_sequences}
        if move_sequence.tobytes() in valid_bytes:
            bg.do_next_board_total(self.board,move_sequence,self.__current_player)

            if not silent:
                self.printmovesequence(move_sequence)
        else:
            raise ValueError("Invalid Move: Exiting")
        
        self.next_player()

        while(not self.game_over()):
            current_player = self.get_current_agent()

            dice = (randrange(1,7),randrange(1,7))

            if not silent:
                self.printturn(dice)

            valid_move_sequences, _ = bg.get_legal_move_sequences(self.board,dice, self.__current_player)
            move_sequence = current_player.get_next_move(self.board, dice, self.__current_player)
            valid_bytes = {seq.tobytes() for seq in valid_move_sequences}
            if move_sequence.tobytes() in valid_bytes:
                bg.do_next_board_total(self.board,move_sequence,self.__current_player)

                if not silent:
                    self.printmovesequence(move_sequence)
            else:
                raise ValueError("Invalid Move: Exiting")
            
            self.next_player()

        if not silent:
            self.printwinner()
            
        winner = bg.get_winner(self.board)

        RED = "\033[91m"
        WHITE = "\033[0m"
        YELLOW = "\033[1;33m"
        winner_c = "White" if (bg.get_winner(self.board) == 1) else "Red"
        winner_cl = WHITE if (bg.get_winner(self.board) == 1) else RED
        print()
        print(f"         {YELLOW}GAME OVER{WHITE}")
        print(f"         {YELLOW}Winner: {winner_cl}{winner_c}{WHITE}")

        if winner == 0:
            raise Exception
        return winner

    def error(self):
        b = self.board
        
        if not b[b>0].sum()==15:
            self.printboard()
            raise Exception
        if not b[b<0].sum()==-15:
            self.printboard()
            raise Exception
        
game = BackGammon(Agent_Random(),Agent_MCTS(30,10))
# game = BackGammon(Agent_Random(),Agent_Random())
# game.play()

wins = 0
start = time.perf_counter()
with cProfile.Profile() as pr:
    game.play()
end = time.perf_counter()

print(f"Elapsed time: {end - start:.6f} seconds")
print(f"Win Rate: {wins}")

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats(20)

