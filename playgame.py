import os
from game import BackGammonGameState
from abstract_agent import Agent
from agent_player import Agent_Player
from agent_random import Agent_Random

from random import randrange
from random import seed

class BackGammon:
    def __init__(self,player1:Agent,player2:Agent):
        self.__players = [player1,player2]
        self.__current_player = 0
        self.gamestate = BackGammonGameState.new_default()

    def next_player(self):
        self.__current_player = (self.__current_player+1)%2
        
    def get_current_agent(self):
        return self.__players[self.__current_player]
    
    def get_gamestate(self):
        return self.gamestate

    def printturn(self,dice):

        remaining_dice_str = "".join(f"{die:2d}" for die in dice)
        print("---------------------------------------------------------------------------------------------------")
        print(self.gamestate)
        print(f"         Die Rolled:{remaining_dice_str}")

    def printmovesequence(self, move_sequence):
        print(f"Move Sequence Taken: ",end='')
        # This is an goofy generator
        print([(("bar", die) if src == "bar" else (src+1, die)) for (src, die) in move_sequence]
)

    def printwinner(self):
        # os.system('cls')
        print()
        print("         GAME OVER")
        print(f"         Winner: {self.gamestate.game_over()}")
        print(self.gamestate)
        
    

    def make_move(self,move_sequence):
        self.gamestate.make_move_sequence(move_sequence)

    def game_over(self):
        if self.gamestate.game_over() is None:
            return False
        return True

    def play(self):
        while(self.game_over() is False):
            
            current_player = self.get_current_agent()
            die = (randrange(1,7),randrange(1,7))

            self.printturn(die)
            
            valid_move_sequences = self.gamestate.get_valid_move_sequences(die)
            move_sequence = current_player.get_next_move(self.gamestate,die)

            if move_sequence in valid_move_sequences:
                self.gamestate.make_move_sequence(move_sequence)
                self.printmovesequence(move_sequence)
            else:
                raise ValueError("Invalid Move: Exiting")
            
            self.next_player()

        self.printwinner()

seed(0)
game = BackGammon(Agent_Random(),Agent_Random())
game.play()

