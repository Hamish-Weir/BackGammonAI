import numpy as np

BOARD_SIZE = 24

START_OF_BOARD  = 0                 # 0
END_OF_BOARD    = BOARD_SIZE-1      # 23
START_OF_BAR    = END_OF_BOARD+1    # 24
END_OF_BAR      = END_OF_BOARD+2    # 25
START_OF_OFF    = END_OF_BOARD+3    # 26
END_OF_OFF      = END_OF_BOARD+4    # 27

class Board():

    P1BAR = START_OF_BAR    # 24
    P2BAR = END_OF_BAR      # 25

    P1OFF = START_OF_OFF    # 26
    P2OFF = END_OF_OFF      # 27


    # Positive = RED
    # Negative = BLUE
    def __init__(self):
        self._tiles = np.array([
            2,  0,  0,  0,  0, -5,  # Red Home
            0, -3,  0,  0,  0,  5,
           -5,  0,  0,  0,  3,  0,
            5,  0,  0,  0,  0, -2,  # Blue Home
            0,  0,                  # Red/Blue Bar
            0,  0])                 # Red/Blue Off
        
        
    
    def set(self,arr):
        assert len(arr) == 28, "Array must be of length 28"
        assert all(isinstance(x, int) for x in arr), "All elements must be integers"
        self._tiles = arr.copy()
    
    def distance(start, end, player):
        assert player == 1 or player == -1
        assert (start <= END_OF_BOARD and start >= START_OF_BOARD) or (start <= END_OF_BAR and start >= START_OF_BAR), f"Start ({start}) in must be in Valid Range"
        assert (end   <= END_OF_BOARD and end   >= START_OF_BOARD) or (end   <= END_OF_OFF and end   >= START_OF_OFF), f"End ({end}) in must be in Valid Range"
        assert not (start == Board.P1BAR and start == Board.P2OFF), f"Invalid Combination of start ({start}) and end ({end})"
        assert not (start == Board.P2BAR and start == Board.P1OFF), f"Invalid Combination of start ({start}) and end ({end})"

        if player == 1:
            match (start,end):
                case (Board.P1BAR,  Board.P1OFF ):  # Whole Board Jump
                    return BOARD_SIZE+1
                case (Board.P1BAR,  _           ):  # Logical Start = -1 
                    return end+1
                case (_,            Board.P1OFF ):  # Logical End = 24
                    return start-BOARD_SIZE
                case _:                             # Normal Move
                    return end-start
        else:
            match (start,end):
                case (Board.P2BAR,  Board.P2OFF ):  # Whole Board Jump
                    return BOARD_SIZE+1
                case (Board.P2BAR,  _           ):  # Logical Start = -1 
                    return BOARD_SIZE-end
                case (_,            Board.P2OFF ):  # Logical End = 24
                    return start+1
                case _:                             # Normal Move
                    return start-end

    def end_point(start,die):
