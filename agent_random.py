from random import randrange
from abstract_agent import Agent
from game import BackGammonGameState

class Agent_Random(Agent):

    def get_next_move(self,game: BackGammonGameState, dice: tuple[int, int]):
        valid_action_sequences = game.get_valid_move_sequences(dice)
        size = len(valid_action_sequences)
        action = randrange(0,size)
        return valid_action_sequences[action]
        