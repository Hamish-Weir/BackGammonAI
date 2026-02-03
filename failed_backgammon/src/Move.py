from dataclasses import dataclass
from math import inf
from src.Colour import Colour
# from Colour import Colour

P1BAR = 24
P2BAR = 25

P1OFF = 26
P2OFF = 27

@dataclass(frozen=True)
class Move:
    """Represents a single player move in a turn of Backgammon."""

    _start: int = 0
    _end:   int = 0
    _die:   int = 0

    def is_valid(self, player:Colour):
        if player == Colour.RED:
            if self._start == P1BAR: # Move from BAR
                if self._end == self._die-1:
                    return True
                return False
            elif self._end == P1OFF: # Move to OFF
                if self._start + self._die >= 24:
                    return True
                return False
            elif self._end - self.start == self._die: # Normal Move
                return True
            return False
        
        elif player == Colour.BLUE:
            if self._start == P2BAR: # Move from BAR
                if self._end == 24 - self._die:
                    return True
                return False
            elif self._end == P2OFF: # Move to OFF
                if self._start < self._die:
                    return True
                return False
            elif self.start - self._end == self._die: # Normal Move
                return True
            return False

    def is_to_off(self):
        if self._end == P1OFF or self._end == P2OFF:
            return True
        return False
    
    def is_from_bar(self):
        
        if self._start == P1BAR or self._start == P2BAR:
            return True
        return False

    def __str__(self) -> str:
        if self.is_to_off():
            S, E, D = f"{self._start+1:3d}", f"OFF", f"{self._die}"
        elif self.is_from_bar():
            S, E, D = f"BAR", f"{self._end+1:3d}", f"{self._die}"
        else:
            S, E, D = f"{self._start+1:3d}", f"{self._end+1:3d}", f"{self._die}"

        return f"({S}->{E}, {D})"
    def __repr__(self) -> str:
        return str(self)
    
    @staticmethod
    def _normalize(pos: int) -> int:
        # Treat 24 and 27 as less than 0
        if pos in (24, 27):
            return -1
        return pos

    def __lt__(self, other):
        if not isinstance(other, Move):
            return NotImplemented

        return (
            self._normalize(self._start),
            self._normalize(self._end),
            self._die,
        ) < (
            self._normalize(other._start),
            self._normalize(other._end),
            other._die,
        )
    

    @property
    def start(self) -> int:
        return self._start

    @property
    def end(self) -> int:
        return self._end

    @property
    def die(self) -> int:
        return self._die

if __name__ == "__main__":
    a = Move(0,     1,      1)
    b = Move(P1BAR, 3,      3)
    c = Move(22,    P1OFF,  5)

    print(f"{a} is valid for P1 :{a.is_valid(Colour.RED)}")
    print(f"{b} is valid for P1 :{b.is_valid(Colour.RED)}")
    print(f"{c} is valid for P1 :{c.is_valid(Colour.RED)}")

    print(f"{a} is valid for P2 :{a.is_valid(Colour.BLUE)}")
    print(f"{b} is valid for P2 :{b.is_valid(Colour.BLUE)}")
    print(f"{c} is valid for P2 :{c.is_valid(Colour.BLUE)}")