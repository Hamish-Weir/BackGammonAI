from copy import deepcopy
from abstract_agent import Agent
from game import BackGammonGameState
from re import fullmatch

class Agent_Player(Agent):

    def get_next_singular_move(self):
        while True:
            print("Enter Move (Format: 'Position/bar/off Die' or 'R' to reset Move Sequence): ",end='')
            input_move = input()

            if self.__reset_action_sequence(input_move):
                return "Reset"
            move = self.__format_move(input_move)

            if move:
                return move
            else:
                print("Invalid Format")

    def get_next_move(self,gamestate: BackGammonGameState, dice: tuple[int, int]):
        valid_action_sequences = gamestate.get_valid_move_sequences(dice)
        temp_gamestate = deepcopy(gamestate)
        temp_dice = list(dice)
        
        # Used to show player what dice they have left
        if temp_dice[0] == temp_dice[1]:
            temp_dice = 4*[temp_dice[0]]
        
        action_sequence = []
        while not (action_sequence in valid_action_sequences):

            action_sequence = []
            usable_dice = temp_dice.copy()
            while usable_dice:
                move = self.get_next_singular_move()

                if move == "Reset": # Breaks out of loop and resets action_sequence and usable_dice
                    print("Reset Called")
                    break

                if move:
                    if self.__is_legal(move,action_sequence,usable_dice,valid_action_sequences):
                        
                        action_sequence.append(move) # Add Move to action_sequence
                        usable_dice.remove(move[1]) # Remove Die from usable_dice
                        print(f"My Action Sequence: {action_sequence}")
                    else:
                        continue
                else:
                    print("Invalid Move")

                if action_sequence in valid_action_sequences: # No More Moves Possible with remaining die (or out of die)
                        return action_sequence
        return action_sequence
     
    def __is_legal(self,move,action_sequence,usable_dice,valid_action_sequences):
        _, die = move
        if not (die in usable_dice):
            return False
        
        move_sequence_so_far = action_sequence + [move] # create this full move sequence

        matches = [sub for sub in valid_action_sequences if sub[:len(move_sequence_so_far)] == move_sequence_so_far]

        if matches:
            return True
        return False

    def __reset_action_sequence(self,input_move):
        if input_move.lower().strip() == "r":
            return True
        return False

    def __format_move(self, input_move:str):
        """Str to Move -> (position, die) WHERE POSITION INDEXES FROM 0"""
        if not bool(fullmatch("^([0-2]?[0-9]|bar) [1-6]$",input_move.lower().strip())):
            return None
        
        position, die = [x.strip() for x in input_move.lower().split()]
        
        if not(position == "bar"):
            position, die = int(position), int(die)
            if position < 1 or position > 24:
                return None
        else:
            position, die = "bar", int(die)
        
        return (position-1,die)
        

    def __make_partial_move(gamestate,partial_move):
        pass