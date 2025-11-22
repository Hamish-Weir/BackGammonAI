from copy import copy
import sys
import numpy as np

P1BAR = 24
P2BAR = 25
P1OFF = 26
P2OFF = 27

def init_board():
    """
    Returns: ndarray[28] of int8, the initial game state 
    """
    # initializes the game board

    board = np.zeros(28, dtype=np.int8)
    board[0] = -2
    board[5] = 5

    board[7] = 3
    board[11] = -5

    board[12] = 5
    board[16] = -3

    board[18] = -5
    board[23] = 2

    return board

def get_legal_moves(board, die, player):
    """
    Returns:
        List[np.ndarray[np.int8]],
            List of Move = [(_,_),(_,_)]
            Move = (Start Pip, End Pip)
        
        if no moves:
            Empty List = []
        
    """
    possible_moves = []

    if bool(board[P1OFF]==15 or board[P2OFF]==-15):
        return possible_moves

    if player == 1:
        if board[P1BAR] > 0: # Piece on Bar
            end_pip = 24-die
            if board[end_pip] > -2:
                possible_moves.append(np.array([np.int8(P1BAR),np.int8(end_pip)],dtype=np.int8)) 
        else: # No Piece on Bar & game not over
            if sum(board[6:24]>0) == 0: # Bearing off is legal
                start_pip = die - 1
                if (board[start_pip] > 0): # Exact Throw; move off
                    possible_moves.append(np.array([np.int8(start_pip),np.int8(P1OFF)],dtype=np.int8))

                else: # No Exact Throw, Bear off furthest if possible
                    start_pip = np.max(np.where(board[0:6]>0)[0]) # Get Furthest Piece
                    if start_pip<die: # If Can Move Off; move off
                        possible_moves.append(np.array([np.int8(start_pip),np.int8(P1OFF)],dtype=np.int8))
            
            # all other legal options
            possible_start_pips = np.where(board[0:24]>0)[0]
            
            start_pips = possible_start_pips
            end_pips = start_pips - die

            mask = (end_pips >= 0) & (end_pips < 24) & ((board[end_pips.clip(0,23)]) > -2)

            masked_start_pips = start_pips[mask]
            masked_end_pips = end_pips[mask]

            new_moves = np.column_stack(np.array([np.int8(masked_start_pips), np.int8(masked_end_pips)],dtype=np.int8))
            possible_moves.extend([move for move in new_moves])
    else:
        if board[P2BAR] < 0: # Piece on Bar
            end_pip = die-1
            if board[end_pip] < 2:
                possible_moves.append(np.array([np.int8(P2BAR),np.int8(end_pip)],dtype=np.int8)) 
        else: # No Piece on Bar & game not over
            if sum(board[0:18]<0) == 0: # Bearing off is legal
                start_pip = 24-die
                if (board[start_pip] < 0): # Exact Throw; move off
                    possible_moves.append(np.array([np.int8(start_pip),np.int8(P2OFF)],dtype=np.int8))

                else: # No Exact Throw, Bear off furthest if possible
                    start_pip = 23 - np.max(np.where(board[23:17:-1]<0)[0]) # Get Furthest Piece idx
                    if start_pip> 24-die: # If Can Move Off; move off
                        possible_moves.append(np.array([np.int8(start_pip),np.int8(P2OFF)],dtype=np.int8))
            
            # all other legal options
            possible_start_pips = np.where(board[0:24]<0)[0]
            
            start_pips = possible_start_pips
            end_pips = start_pips + die

            mask = (end_pips >= 0) & (end_pips < 24) & ((board[end_pips.clip(0,23)]) < 2)

            masked_start_pips = start_pips[mask]
            masked_end_pips = end_pips[mask]

            new_moves = np.column_stack(np.array([np.int8(masked_start_pips), np.int8(masked_end_pips)],dtype=np.int8))
            possible_moves.extend([move for move in new_moves])
            
    return possible_moves

def get_legal_move_sequences(board,dice, player):
    """
    Returns:
        List[np.ndarray[np.ndarray[np.int8]]],
            List of Move Sequence = [Move Sequence,Move Sequence,...]
            Move Sequence = [Move,Move,...]
            Move = (Start Pip, End Pip)
        
        if no moves:
            List[np.array[np.array[]]]
                List of Move Sequence = [Move Sequence]
                Move Sequence = [move]
                Move = ()

        List of ALL Legal Move Sequences
    """
    moves = []
    boards = []
   

    if dice[0] == dice[1]: # For Doubles
        # try using the first dice, then the second dice, then third, then fourth (i hate doing this but oh well)
        possible_first_moves = get_legal_moves(board, dice[0],player)
        if possible_first_moves:
            for m1 in possible_first_moves:
                temp_board1 = get_next_board_partial(board,m1,player)
                possible_second_moves = get_legal_moves(temp_board1, dice[0], player)
                if possible_second_moves:
                    for m2 in possible_second_moves:
                        temp_board2 = get_next_board_partial(temp_board1,m2,player)
                        possible_third_moves = get_legal_moves(temp_board2, dice[0], player)
                        if possible_third_moves:
                            for m3 in possible_third_moves:
                                temp_board3 = get_next_board_partial(temp_board2,m3,player) 
                                possible_fourth_moves = get_legal_moves(temp_board3, dice[0], player)
                                if possible_fourth_moves:
                                    for m4 in possible_fourth_moves:
                                        temp_board4 = get_next_board_partial(temp_board3,m4,player)
                                        moves.append(np.array([m1,m2,m3,m4],dtype=np.int8))
                                        boards.append(get_flip_board(temp_board4))
                                else:
                                    moves.append(np.array([m1,m2,m3],dtype=np.int8))
                                    boards.append(get_flip_board(temp_board3))
                        else:
                            moves.append(np.array([m1,m2],dtype=np.int8))
                            boards.append(get_flip_board(temp_board2))
                else:
                    moves.append(np.array([m1],dtype=np.int8))
                    boards.append(get_flip_board(temp_board1))
        else:
            moves.append(np.array([],dtype=np.int8))
            boards.append(copy(board))
    else: # For Non-Doubles
        # try using the first dice, then the second dice
        possible_first_moves = get_legal_moves(board, dice[0], player)
        for m1 in possible_first_moves:
            temp_board1 = get_next_board_partial(board,m1,player)
            possible_second_moves = get_legal_moves(temp_board1,dice[1], player)
            for m2 in possible_second_moves:
                temp_board2 = get_next_board_partial(temp_board1,m2,player)
                moves.append(np.array([m1,m2],dtype=np.int8))
                boards.append(get_flip_board(temp_board2))
            
        # try using the second dice, then the first one
        possible_first_moves = get_legal_moves(board, dice[1], player)
        for m1 in possible_first_moves:
            temp_board1 = get_next_board_partial(board,m1,player)
            possible_second_moves = get_legal_moves(temp_board1,dice[0], player)
            for m2 in possible_second_moves:
                temp_board2 = get_next_board_partial(temp_board1,m2,player)
                moves.append(np.array([m1,m2],dtype=np.int8))
                boards.append(get_flip_board(temp_board2))
                
        # if there's no pair of moves available, allow one move:
        if len(moves)==0: 
            # first dice:
            possible_first_moves = get_legal_moves(board, dice[0], player)
            for m in possible_first_moves:
                temp_board1 = get_next_board_partial(board,m,player)
                moves.append(np.array([m],dtype=np.int8))
                boards.append(get_flip_board(temp_board1))
                
            # second dice:
            possible_first_moves = get_legal_moves(board, dice[1], player)
            for m in possible_first_moves:
                temp_board1 = get_next_board_partial(board,m,player)
                moves.append(np.array([m],dtype=np.int8))
                boards.append(get_flip_board(temp_board1))

            if len(moves) == 0:
                moves.append(np.array([],dtype=np.int8))
                boards.append(get_flip_board(board))

    if len(moves) > 1:
        def safe_get(x, i):
            """Return x[0][i][0] if it exists, otherwise return -inf."""
            if len(x[0]) > i:
                return x[0][i][0]
            else:
                return float("-inf")
            
        paired = list(zip(moves, boards))
        if player == 1:
            paired.sort(key= lambda x: [safe_get(x, 0), safe_get(x, 1), safe_get(x, 2), safe_get(x, 3)],reverse=False)  
        else:
            paired.sort(key= lambda x: [safe_get(x, 0), safe_get(x, 1), safe_get(x, 2), safe_get(x, 3)],reverse=True)  

        moves, boards = zip(*paired)
    return moves, boards

def get_unique_legal_move_sequences(board,dice, player):
    """
    Returns:
        Same as get_legal_move_sequences but only ones ending in unique board states

        List of ALL Legal Move Sequences
    """
    moves = []
    boards = []
    seen_board_states = set()

    if dice[0] == dice[1]: # For Doubles
        # try using the first dice, then the second dice, then third, then fourth (i hate doing this but oh well)
        possible_first_moves = get_legal_moves(board, dice[0], player)
        if possible_first_moves:
            for m1 in possible_first_moves:
                temp_board1 = get_next_board_partial(board,m1,player)
                possible_second_moves = get_legal_moves(temp_board1, dice[0], player)
                if possible_second_moves:
                    for m2 in possible_second_moves:
                        temp_board2 = get_next_board_partial(temp_board1,m2,player)
                        possible_third_moves = get_legal_moves(temp_board2, dice[0], player)
                        if possible_third_moves:
                            for m3 in possible_third_moves:
                                temp_board3 = get_next_board_partial(temp_board2,m3,player) 
                                possible_fourth_moves = get_legal_moves(temp_board3, dice[0], player)
                                if possible_fourth_moves:
                                    for m4 in possible_fourth_moves:
                                        temp_board4 = get_next_board_partial(temp_board3,m4,player)
                                        if not temp_board4.tobytes() in seen_board_states:
                                            moves.append(np.array([m1,m2,m3,m4],dtype=np.int8))
                                            boards.append(get_flip_board(temp_board4))
                                            seen_board_states.add(temp_board4.tobytes())
                                else:
                                    if not temp_board3.tobytes() in seen_board_states:
                                        moves.append(np.array([m1,m2,m3],dtype=np.int8))
                                        boards.append(get_flip_board(temp_board3))
                                        seen_board_states.add(temp_board3.tobytes())
                        else:
                            if not temp_board2.tobytes() in seen_board_states:
                                moves.append(np.array([m1,m2],dtype=np.int8))
                                boards.append(get_flip_board(temp_board2))
                                seen_board_states.add(temp_board2.tobytes())
                else:
                    if not temp_board1.tobytes() in seen_board_states:
                        moves.append(np.array([m1],dtype=np.int8))
                        boards.append(get_flip_board(temp_board1))
                        seen_board_states.add(temp_board1.tobytes())
        else:
            moves.append(np.array([],dtype=np.int8))
            boards.append(copy(board))
    else: # For Non-Doubles
        # try using the first dice, then the second dice
        possible_first_moves = get_legal_moves(board, dice[0], player)
        for m1 in possible_first_moves:
            temp_board1 = get_next_board_partial(board,m1,player)
            possible_second_moves = get_legal_moves(temp_board1,dice[1], player)
            for m2 in possible_second_moves:
                temp_board2 = get_next_board_partial(temp_board1,m2,player)
                if not temp_board2.tobytes() in seen_board_states:
                    moves.append(np.array([m1,m2],dtype=np.int8))
                    boards.append(get_flip_board(temp_board2))
                    seen_board_states.add(temp_board2.tobytes())
            
        # try using the second dice, then the first one
        possible_first_moves = get_legal_moves(board, dice[1], player)
        for m1 in possible_first_moves:
            temp_board1 = get_next_board_partial(board,m1,player)
            possible_second_moves = get_legal_moves(temp_board1,dice[0], player)
            for m2 in possible_second_moves:
                temp_board2 = get_next_board_partial(temp_board1,m2,player)
                if not temp_board2.tobytes() in seen_board_states:
                    moves.append(np.array([m1,m2],dtype=np.int8))
                    boards.append(get_flip_board(temp_board2))
                    seen_board_states.add(temp_board1.tobytes())
                
        # no pair of moves available, allow one move:
        if len(moves)==0: 
            # first dice:
            possible_first_moves = get_legal_moves(board, dice[0], player)
            for m in possible_first_moves:
                temp_board1 = get_next_board_partial(board,m,player)
                moves.append(np.array([m],dtype=np.int8))
                boards.append(get_flip_board(temp_board1))
                
            # second dice:
            possible_first_moves = get_legal_moves(board, dice[1], player)
            for m in possible_first_moves:
                temp_board1 = get_next_board_partial(board,m,player)
                moves.append(np.array([m],dtype=np.int8))
                boards.append(get_flip_board(temp_board1))

        # no moves available
        if len(moves) == 0:
            moves.append(np.array([],dtype=np.int8))
            boards.append(get_flip_board(board))

    return moves, boards

def get_next_board_partial(board,move,player):
    """
        Returns: ndarray[28] of int8, board state after single move
    """
    copy_board = copy(board)
    do_next_board_partial(copy_board,move,player)
    return copy_board

def get_next_board_total(board, move_sequence, player):
    next_board = copy(board)
    do_next_board_total(next_board,move_sequence,player)
    return next_board

def get_flip_board(board):
    """Get what the board looks like to opponent"""
    cannon_board = np.zeros(28,dtype=int)
    cannon_board[0:24] = -board[0:24][::-1]

    cannon_board[P2OFF], cannon_board[P1OFF] = -board[P1OFF], -board[P2OFF]
    cannon_board[P2BAR], cannon_board[P1BAR] = -board[P1BAR], -board[P2BAR]

    return cannon_board

def get_cannon_board(board,player):
    if player == 1:
        return copy(board)
    else:
        return get_flip_board(board)

def get_cannon_move(move, player):
    if player == 1:
        return move
    else:
        start_pip, end_pip = move

        if start_pip == P2BAR:
            start_pip = P1BAR
        else:
            start_pip = 23 - start_pip
            
        if end_pip == P2OFF:
            end_pip = P1OFF
        else:
            end_pip = 23 - end_pip

        return np.array([np.int8(start_pip),np.int8(end_pip)],dtype=np.int8)
    
def get_cannon_move_sequence(move_sequence, player):
    if player == 1:
        return move_sequence
    else:
        moves = [get_cannon_move(move,-1) for move in move_sequence]
        return np.array(moves)

def get_winner(board):
    if board[P1OFF]==15:
        return 1
    if board[P2OFF]==-15:
        return -1
    return 0

def do_next_board_partial(board, move, player):
    startPip, endPip = move
    if player == 1:
        # if killable move dead piece to BAR
        if board[endPip]==-1:
            board[endPip] = 0
            board[P2BAR] = board[P2BAR]-1

        board[startPip] = board[startPip]-1
        board[endPip] = board[endPip]+1
    else:
        # if killable move dead piece to BAR
        if board[endPip]== 1:
            board[endPip] = 0
            board[P1BAR] = board[P1BAR]+1

        board[startPip] = board[startPip]+1
        board[endPip] = board[endPip]-1

def do_next_board_total(board, move_sequence, player):
    for move in move_sequence:
        do_next_board_partial(board,move,player)

def game_over(board):
    if get_winner(board) == 0:
        return False
    return True
