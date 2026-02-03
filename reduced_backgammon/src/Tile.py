from src.Colour import Colour
from dataclasses import dataclass


@dataclass
class Tile:
    _x: int
    _count: int = 0
    _colour: Colour = None

    @property
    def x(self):
        return self._x

    @property
    def p_count(self):
        return self._count

    @property
    def colour(self):
        return self._colour

    @p_count.setter
    def p_count(self, count):
        self._count = count

    @colour.setter
    def colour(self, colour):
        self._colour = colour
