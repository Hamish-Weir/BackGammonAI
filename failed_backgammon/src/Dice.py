from copy import deepcopy
from random import randint

class Dice:
    _die1 = 0
    _die2 = 0

    def roll(self):
        self._die1 = randint(1,6)
        self._die2 = randint(1,6)

        return deepcopy([self._die1, self._die2])

    def get(self):
        return deepcopy([self._die1, self._die2])

    @property
    def die1(self) -> int:
        return self._die1

    @property
    def die2(self) -> int:
        return self._die2
    
if __name__ == "__main__":
    dice = Dice()

    for i in range(100):
        d = dice.roll()
        print(d)