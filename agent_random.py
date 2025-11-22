from random import randrange

from abstract_agent import Agent

import backgammon_helper as bg


class Agent_Random(Agent):
    def get_next_move(self,gamestate, dice, player):

        next_move_sequence, _ = bg.get_legal_move_sequences(gamestate, dice, player)
        if next_move_sequence:
            size = len(next_move_sequence)
            action = randrange(0,size)
            return next_move_sequence[action]
