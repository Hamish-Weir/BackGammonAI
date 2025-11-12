from copy import deepcopy
from abstract_agent import Agent
from game import BackGammonGameState
from re import fullmatch

class Agent_Player(Agent):

    def get_next_move(self,gamestate: BackGammonGameState, dice: tuple[int, int]):
        valid_action_sequences = gamestate.get_valid_move_sequences(dice)
        temp_gamestate = deepcopy(gamestate)
        temp_dice = list(dice)
        
        # Used to show player what dice they have left
        if temp_dice[0] == temp_dice[1]:
            temp_dice = 4*[temp_dice[0]]
        
        while True:
            action_sequence = []
            usable_dice = temp_dice.copy()
            while usable_dice:
                while True:
                    print("Enter Move (Format: Position/bar/off Die): ",end='')
                    input_move = input()
                    move = self.__format_move(input_move)

                    if move:
                        if self.__is_legal(temp_gamestate,input_move):
                            break
                        else:
                            print("Invalid Move")
                    else:
                        print("Invalid Format")

            if action_sequence in valid_action_sequences:
                return action_sequence
            else:
                print("Illegal Move Sequence")
            
    def __is_legal():
        pass

    def __format_move(input_move:str, player):
        if not bool(fullmatch("^([0-2]?[0-9]|bar) [1-6]$",input_move.lower().strip())):
            return None
        
        position, die = input_move.split()

        if position < 1 or position > 24:
            return None
        
        


    def __make_partial_move(gamestate,partial_move):
        pass