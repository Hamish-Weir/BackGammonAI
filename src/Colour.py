from enum import Enum

class Colour(Enum):
    """This enum describes the sides in a game of Backgammon."""

    # RED is positive heading, BLUE is negative heading
    RED = 0
    BLUE = 1

    def get_char(colour):
        """Returns the name of the colour as an uppercase character."""

        if colour == Colour.RED:
            return "R"
        elif colour == Colour.BLUE:
            return "W"
        else:
            return "0"

    def from_char(c):
        """Returns a colour from its char representations."""

        if c == "R":
            return Colour.RED
        elif c == "W":
            return Colour.BLUE
        else:
            return None

    def opposite(colour):
        """Returns the opposite colour."""

        if colour == Colour.RED:
            return Colour.BLUE
        elif colour == Colour.BLUE:
            return Colour.RED
        else:
            return None


if __name__ == "__main__":
    for colour in Colour:
        print(colour)