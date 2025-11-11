from game import BackGammonGame
from random import randrange

class RandomAgent:

    def get_next_move(self,game: BackGammonGame, dice: tuple[int, int]):
        valid_action_sequences = game.get_valid_move_sequences(dice)
        size = len(valid_action_sequences)
        action = randrange(0,size)
        return valid_action_sequences[action]
        