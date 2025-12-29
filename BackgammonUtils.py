import numpy as np
import torch
from src.MoveSequence import MoveSequence
from src.Move import Move
from src.Board import Board
from src.Colour import Colour


class BackgammonUtils():

    @staticmethod
    def get_prior_idx(move_sequence, dice):
    
        #   0 - 324; Move, Move; Regular dice order
        # 325 - 349; Move, Skip; Regular dice order
        # 350 - 674; Move, Move; Inverse dice order
        # 675 - 699; Move, Skip; Inverse dice order
        # 700;       Skip        

        if not move_sequence:
            return 700

        l = len(move_sequence)

        if l == 1:
            S,_,_ = move_sequence[0]

            if S == Board.P1BAR:
                S = 24

            if move_sequence[0][2] != dice[0]:
                idx = 675 + S
            else:
                idx = 325 + S
            
        else:
            S1,_,_ = move_sequence[0]
            S2,_,_ = move_sequence[1]

            if S1 == Board.P1BAR:
                S1 = 24
            if S2 == Board.P1BAR:
                S2 = 24

            if move_sequence[0][2] != dice[0]:
                idx = 350 + (S1 * (S1 + 1) // 2) + S2
            else:
                idx = 0 + (S1 * (S1 + 1) // 2) + S2

        return idx

    @staticmethod
    def get_internal_board(board:Board):
        return np.array([tile.p_count if tile.colour == Colour.RED else -tile.p_count for tile in board._tiles],dtype=np.int8)

    @staticmethod
    def get_internal_move(movesequence:MoveSequence):
        moves = movesequence.get_moves()
        return [(move.start,move.end,move.die) for move in moves]

    @staticmethod
    def get_external_movesequence(movesequence:list[tuple[int,int]]):
        
        movesequence = movesequence + [None]*4
        A = Move(movesequence[0][0],movesequence[0][1],movesequence[0][2]) if movesequence[0] else None
        B = Move(movesequence[1][0],movesequence[1][1],movesequence[1][2]) if movesequence[1] else None
        C = Move(movesequence[2][0],movesequence[2][1],movesequence[2][2]) if movesequence[2] else None
        D = Move(movesequence[3][0],movesequence[3][1],movesequence[3][2]) if movesequence[3] else None
        return MoveSequence(A,B,C,D)

    @staticmethod
    def encode_board(board,dice,type,player):

        perspective_board = BackgammonUtils.get_perspective_board(board,player)
        diceplayer = np.append(np.array(dice),np.array(type))
        arr = np.append(perspective_board,diceplayer)

        return torch.tensor(arr, dtype=torch.float)

    @staticmethod
    def decode_move(move, player):
        start, end, die = move
        if player == -1:
            if start == Board.P1BAR:
                start = Board.P2BAR
            else:
                start = 23 - start

            if end == Board.P1OFF:
                end = Board.P2OFF
            else:
                end = 23 - end

        return (start,end,die)

    @staticmethod
    def get_perspective_board(board, player):
        if player == 1:
            return board.copy()
        else:
            temp_board = board.copy()
            
            temp_board[0:24] = -temp_board[0:24][::-1]

            temp_board[Board.P1BAR],temp_board[Board.P2BAR] = temp_board[Board.P2BAR], temp_board[Board.P1BAR]
            temp_board[Board.P1OFF],temp_board[Board.P2OFF] = temp_board[Board.P2OFF], temp_board[Board.P1OFF]

            return temp_board

    @staticmethod
    def _movesequence_key(move_sequence: list[tuple[int]]) -> tuple[tuple[int,...]]:
        return tuple(move_sequence)

    @staticmethod
    def get_legal_moves(board, die, player):
        """
        Returns:
            List[MoveSequence[Move]]
            Move = (start,end,die)
            
        """
        
        moveSet = set()

        if bool(board[Board.P1OFF]==15 or board[Board.P2OFF]==-15):
            return moveSet

        if player == -1:
            if board[Board.P2BAR] < 0: # Piece on Bar
                end_pip = 24-die
                if board[end_pip] < 2:
                    moveSet.add((int(Board.P2BAR),int(end_pip),int(die))) 
            else: # No Piece on Bar & game not over
                if sum(board[6:24] < 0) == 0: # Bearing off is legal
                    start_pip = die - 1
                    if (board[start_pip] < 0): # Exact Throw; move off
                        moveSet.add((int(start_pip),int(Board.P2OFF),int(die)))

                    else: # No Exact Throw, Bear off furthest if possible
                        start_pip = np.max(np.where(board[0:7]<0)[0]) # Get Furthest Piece
                        if start_pip < die: # If Can Move Off; move off
                            moveSet.add((int(start_pip),int(Board.P2OFF),int(die)))
                
                # all other legal options
                possible_start_pips = np.where(board[0:24]<0)[0]
                
                start_pips = possible_start_pips
                end_pips = start_pips - die

                mask = (end_pips >= 0) & (end_pips < 24) & ((board[end_pips.clip(0,23)]) < 2)

                masked_start_pips = start_pips[mask]
                masked_end_pips = end_pips[mask]

                new_moves = [(int(start_pip),int(end_pip),int(die)) for start_pip, end_pip in zip(masked_start_pips,masked_end_pips)]
                moveSet.update(new_moves)

        elif player == 1:
            if board[Board.P1BAR] > 0: # Piece on Bar
                end_pip = die-1
                if board[end_pip] > -2:
                    moveSet.add((int(Board.P1BAR),int(end_pip),int(die))) 
            else: # No Piece on Bar & game not over
                if sum(board[0:18] > 0) == 0: # Bearing off is legal
                    start_pip = 24 - die
                    if (board[start_pip] > 0): # Exact Throw; move off
                        moveSet.add((int(start_pip),int(Board.P1OFF),int(die)))

                    else: # No Exact Throw, Bear off furthest if possible
                        start_pip = 23 - np.max(np.where(board[23:17:-1]>0)[0]) # Get Furthest Piece idx
                        if start_pip > 24-die: # If Can Move Off; move off
                            moveSet.add((int(start_pip),int(Board.P1OFF),int(die)))
                
                # all other legal options
                possible_start_pips = np.where(board[0:24]>0)[0]
                
                start_pips = possible_start_pips
                end_pips = start_pips + die

                mask = (end_pips >= 0) & (end_pips < 24) & ((board[end_pips.clip(0,23)]) > -2)

                masked_start_pips = start_pips[mask]
                masked_end_pips = end_pips[mask]

                new_moves = [(int(start_pip),int(end_pip),int(die)) for start_pip, end_pip in zip(masked_start_pips,masked_end_pips)]
                moveSet.update(new_moves)
                

        if len(moveSet) > 1:
            move_sequences = [(s,e,d) for s,e,d in moveSet]
        else:
            move_sequences = []
            

        return list(moveSet)

    @staticmethod
    def get_legal_move_sequences(board,dice:list[int,int], player:int):
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
        def get_board_and_legal(board,m,player):
            temp_board = board.copy()
            BackgammonUtils.do_next_board_partial(temp_board,m,player)
            return temp_board

        moveSequenceSet = set()
        moveSequenceList = []
        dice.sort()

        if dice[0] == dice[1]: # For Doubles
            # try using the first dice, then the second dice, then third, then fourth (i hate doing this but oh well)
            possible_first_moves = BackgammonUtils.get_legal_moves(board, dice[0],player)
            if possible_first_moves:
                for m1 in possible_first_moves:
                    temp_board1 = get_board_and_legal(board,m1,player)
                    possible_second_moves = BackgammonUtils.get_legal_moves(temp_board1, dice[0], player)
                    if possible_second_moves:
                        for m2 in possible_second_moves:
                            temp_board2 = get_board_and_legal(temp_board1,m2,player)
                            possible_third_moves = BackgammonUtils.get_legal_moves(temp_board2, dice[0], player)
                            if possible_third_moves:
                                for m3 in possible_third_moves:
                                    temp_board3 = get_board_and_legal(temp_board2,m3,player) 
                                    possible_fourth_moves = BackgammonUtils.get_legal_moves(temp_board3, dice[0], player)
                                    if possible_fourth_moves:
                                        for m4 in possible_fourth_moves:
                                            if not (m1,m2,m3,m4) in moveSequenceSet:
                                                moveSequenceSet.add((m1,m2,m3,m4))
                                                moveSequenceList.append([m1,m2,m3,m4])
                                            
                                    else:
                                        if not (m1,m2,m3) in moveSequenceSet:
                                            moveSequenceSet.add((m1,m2,m3))
                                            moveSequenceList.append([m1,m2,m3])
                            else:
                                if not (m1,m2) in moveSequenceSet:
                                    moveSequenceSet.add((m1,m2))
                                    moveSequenceList.append([m1,m2])
                    else:
                        if not (m1) in moveSequenceSet:
                            moveSequenceSet.add((m1))
                            moveSequenceList.append([m1])
            else:
                if not () in moveSequenceSet:
                    moveSequenceSet.add(())
                    moveSequenceList.append([])
        else: # For Non-Doubles
            # try using the first dice, then the second dice
            possible_first_moves = BackgammonUtils.get_legal_moves(board, dice[0], player)
            for m1 in possible_first_moves:
                temp_board1 = get_board_and_legal(board,m1,player)
                possible_second_moves = BackgammonUtils.get_legal_moves(temp_board1,dice[1], player)
                if possible_second_moves:
                    for m2 in possible_second_moves:
                        temp_board2 = get_board_and_legal(temp_board1,m2,player)
                        if not (m1,m2) in moveSequenceSet:
                            moveSequenceSet.add((m1,m2))
                            moveSequenceList.append([m1,m2])
                else: 
                    if not (m1) in moveSequenceSet:
                        moveSequenceSet.add((m1))
                        moveSequenceList.append([m1])
                
            # try using the second dice, then the first one
            possible_first_moves = BackgammonUtils.get_legal_moves(board, dice[1], player)
            for m1 in possible_first_moves:
                temp_board1 = get_board_and_legal(board,m1,player)
                possible_second_moves = BackgammonUtils.get_legal_moves(temp_board1,dice[0], player)
                if possible_second_moves:
                    for m2 in possible_second_moves:
                        temp_board2 = get_board_and_legal(temp_board1,m2,player)
                        if not (m1,m2) in moveSequenceSet:
                            moveSequenceSet.add((m1,m2))
                            moveSequenceList.append([m1,m2])
                else: 
                    if not (m1) in moveSequenceSet:
                        moveSequenceSet.add((m1))
                        moveSequenceList.append([m1])

        if len(moveSequenceSet) == 0:
            if not () in moveSequenceSet:
                moveSequenceSet.add(())
                moveSequenceList.append([])

        return moveSequenceList

    @staticmethod
    def TwoMove_get_legal_move_sequences(board,dice:list[int,int], player:int):
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
        def get_board_and_legal(board,m,player):
            temp_board = board.copy()
            BackgammonUtils.do_next_board_partial(temp_board,m,player)
            return temp_board

        moveSequenceSet = set()
        moveSequenceList = []
        dice.sort()

        if dice[0] == dice[1]: # For Doubles
            # try using the first dice, then the second dice, then third, then fourth (i hate doing this but oh well)
            possible_first_moves = BackgammonUtils.get_legal_moves(board, dice[0],player)
            if possible_first_moves:
                for m1 in possible_first_moves:
                    temp_board1 = get_board_and_legal(board,m1,player)
                    possible_second_moves = BackgammonUtils.get_legal_moves(temp_board1, dice[0], player)
                    if possible_second_moves:
                        for m2 in possible_second_moves:
                            if not tuple(sorted([m1,m2])) in moveSequenceSet:
                                moveSequenceSet.add(tuple(sorted([m1,m2])))
                                moveSequenceList.append([m1,m2])
                    else:
                        if not (m1) in moveSequenceSet:
                            moveSequenceSet.add((m1))
                            moveSequenceList.append([m1])
            else:
                if not () in moveSequenceSet:
                    moveSequenceSet.add(())
                    moveSequenceList.append([])
        else: # For Non-Doubles
            # try using the first dice, then the second dice
            possible_first_moves = BackgammonUtils.get_legal_moves(board, dice[0], player)
            for m1 in possible_first_moves:
                temp_board1 = get_board_and_legal(board,m1,player)
                possible_second_moves = BackgammonUtils.get_legal_moves(temp_board1,dice[1], player)
                if possible_second_moves:
                    for m2 in possible_second_moves:
                        temp_board2 = get_board_and_legal(temp_board1,m2,player)
                        if not tuple(sorted([m1,m2])) in moveSequenceSet:
                            moveSequenceSet.add(tuple(sorted([m1,m2])))
                            moveSequenceList.append([m1,m2])
                else: 
                    if not (m1) in moveSequenceSet:
                        moveSequenceSet.add((m1))
                        moveSequenceList.append([m1])
                
            # try using the second dice, then the first one
            possible_first_moves = BackgammonUtils.get_legal_moves(board, dice[1], player)
            for m1 in possible_first_moves:
                temp_board1 = get_board_and_legal(board,m1,player)
                possible_second_moves = BackgammonUtils.get_legal_moves(temp_board1,dice[0], player)
                if possible_second_moves:
                    for m2 in possible_second_moves:
                        temp_board2 = get_board_and_legal(temp_board1,m2,player)
                        if not tuple(sorted([m1,m2])) in moveSequenceSet:
                            moveSequenceSet.add(tuple(sorted([m1,m2])))
                            moveSequenceList.append([m1,m2])
                else: 
                    if not (m1) in moveSequenceSet:
                        moveSequenceSet.add((m1))
                        moveSequenceList.append([m1])

        if len(moveSequenceSet) == 0:
            if not () in moveSequenceSet:
                moveSequenceSet.add(())
                moveSequenceList.append([])

        return moveSequenceList

    @staticmethod
    def do_next_board_partial(board, move, player):
        if move:
            startPip, endPip, die = move
            if player == 1:
                # if killable move dead piece to BAR
                if board[endPip]==-1:
                    board[endPip] = 0
                    board[Board.P2BAR] = board[Board.P2BAR]-1

                board[startPip] = board[startPip]-1
                board[endPip] = board[endPip]+1
            else:
                # if killable move dead piece to BAR
                if board[endPip]== 1:
                    board[endPip] = 0
                    board[Board.P1BAR] = board[Board.P1BAR]+1

                board[startPip] = board[startPip]+1
                board[endPip] = board[endPip]-1

    @staticmethod
    def do_next_board_total(board, move_sequence, player):
        for move in move_sequence:
            BackgammonUtils.do_next_board_partial(board,move,player)

    @staticmethod
    def game_over(board) -> bool:
        return (board[Board.P1OFF] == 15 or board[Board.P2OFF] == -15)
    
    @staticmethod
    def has_won(board) -> int:
        if board[Board.P1OFF] == 15:
            return 1
        elif board[Board.P2OFF] == -15:
            return -1

    @staticmethod
    def heuristic_evaluation(board: np.ndarray, player: int) -> float:
        """
        Heuristic evaluation of a Backgammon board.

        """
        def pip_count(board, player):
            """Get diff average pip distance from OFF"""
            p1_pips = 0
            p2_pips = 0

            for i in range(24):
                n = int(board[i])
                if n > 0:  # P1
                    p1_pips += n * (23 - i)
                elif n < 0:  # P2
                    p2_pips += (-n) * i

            # Bar checkers add full board distance
            p1_pips += int(board[Board.P1BAR]) * 24
            p2_pips += int(-board[Board.P2BAR]) * 24

            pip_advantage = p2_pips - p1_pips if player == 1 else p1_pips - p2_pips

            return pip_advantage

        def blot(board,player):
            """Get diff number of single Pips"""
            blot_penalty = 0
            for i in range(24):
                if board[i] == player:
                    blot_penalty -= 1
                elif board[i] == -player:
                    blot_penalty += 1

            return blot_penalty

        def anchor(board, player):
            """Get diff number of SAFE tiles in home"""
            anchor_bonus = 0

            if player == 1:
                opp_home = range(0, 6)
            else:
                opp_home = range(18, 24)

            for i in opp_home:
                if board[i] * player >= 2:
                    anchor_bonus += 1
                elif board[i] * player <= -2:
                    anchor_bonus -= 1

            return anchor_bonus

        def primes(board,player):
            """Get diff number of safe runs"""
            primes_self = run_self = 0
            primes_opp = run_opp = 0
            for i in range(24):

                if board[i] * -player >= 2:
                    run_opp += 1
                    primes_opp = max(primes_opp, run_opp)
                else:
                    run_opp = 0

                if board[i] * player >= 2:
                    run_self += 1
                    primes_self = max(primes_self, run_self)
                else:
                    run_self = 0

            prime_diff = primes_self - primes_opp

            return prime_diff

        def bar(board,player):
            """Get diff number of tiles on BAR"""
            bar_penalty = 0
            bar_penalty -= board[Board.P1BAR] if player == 1 else -board[Board.P2BAR]
            bar_penalty += -board[Board.P2BAR] if player == 1 else board[Board.P1BAR]

            return bar_penalty

        def off(board,player):
            """Get diff number of tiles OFF"""
            off_bonus = 0
            off_bonus += board[Board.P1OFF] if player == 1 else -board[Board.P2OFF]
            off_bonus -= -board[Board.P2OFF] if player == 1 else board[Board.P1OFF]

            return off_bonus

        assert player in (1, -1)
        assert board.shape[0] == 28

        Pi = 0.1 * pip_count(board,player)
        Bl = 1.5 * blot(board,player)
        An = 2.0 * anchor(board,player)
        Pr = 3.0 * primes(board,player)
        Ba = 7.0 * bar(board,player)
        Of = 5.0 * off(board,player)

        score = Pi+Bl+An+Pr+Ba+Of

        return float(score/500)

