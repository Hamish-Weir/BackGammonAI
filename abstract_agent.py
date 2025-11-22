from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def get_next_move(self,game, dice: tuple[int, int], player:int):
        """Return next move series based on gamestate and dice roll"""
        pass
    