from abstract_agent import Agent
from game import BackGammonGameState

class Agent_Player(Agent):

    def get_next_move(self,game: BackGammonGameState, dice: tuple[int, int]):
        valid_action_sequences = game.get_valid_move_sequences(dice)
        while True:
            action_sequence = []

            print("Enter Move (Format: Position/bar/off Die): ",end='')
            input_move = input()

            if action_sequence in valid_action_sequences:
                return action_sequence