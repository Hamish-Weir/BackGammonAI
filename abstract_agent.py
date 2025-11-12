from abc import ABC, abstractmethod
from game import BackGammonGameState

class Agent(ABC):
    @abstractmethod
    def get_next_move(self,game: BackGammonGameState, dice: tuple[int, int]):
        """Return next move series based on gamestate and dice roll"""
        pass
    