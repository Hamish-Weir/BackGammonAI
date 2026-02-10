from copy import deepcopy
from dataclasses import dataclass

from src.Colour import Colour
from src.Move import Move

# from Colour import Colour
# from Move import Move

P1BAR = 24
P2BAR = 25

P1OFF = 26
P2OFF = 27

@dataclass(frozen=True)
class MoveSequence:
    """Represents a single player move in a turn of Backgammon."""

    _first_move:    Move|None = None
    _second_move:   Move|None = None

    def __init__(self, first:Move|None = None, second:Move|None = None): # Compress all moves so none are last.
        moves = [move for move in (first, second) if move is not None]

        if moves:
            if Move._normalize(moves[0].start) < Move._normalize(moves[0].end):
                moves.sort()
            else:
                moves.sort(reverse=True)
        
        # Use object.__setattr__ to bypass frozen restriction
        object.__setattr__(self, "_first_move", moves[0] if len(moves) > 0 else None)
        object.__setattr__(self, "_second_move", moves[1] if len(moves) > 1 else None)

        

    def is_valid(self, player:Colour):
        moves = [self._first_move,self._second_move]
        

        # Check that moves are sequential format
        seen_none = False
        for move in moves:
            if move is None:
                seen_none = True
            elif seen_none:  # saw None earlier but got Object now → invalid
                return False
            else:
                if not move.is_valid(player): # Check move is valid format
                    return False
                
            
        return True
            
    def __str__(self) -> str:
        if self._first_move:
            move_seq_str = f"Move 1: {self._first_move.__str__():<12}"
            if self._second_move:
                move_seq_str += f"; Move 2: {self._second_move.__str__():<12}"
        else:
            move_seq_str = "Skip"
        
        return f"MoveSequence({move_seq_str})"

    def __repr__(self) -> str:
        return "\n" + str(self)

    def __lt__(self, other):
        if not isinstance(other, MoveSequence):
            return NotImplemented

        return (
            self._first_move,
            self._second_move
        ) < (
            other._first_move,
            other._second_move
        )

    @property
    def first(self) -> Move:
        return self._first_move

    @property
    def second(self) -> int:
        return self._second_move
    
    def get_moves(self) -> list[Move]:
        move_sequence = []
        if self._first_move:
            move_sequence.append(self._first_move)
            if self._second_move:
                move_sequence.append(self._second_move)

        return move_sequence



if __name__ == "__main__":
    ms = MoveSequence(Move(),Move())

    print(ms.is_valid(Colour.RED))