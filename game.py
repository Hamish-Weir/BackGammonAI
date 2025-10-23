from copy import deepcopy
from itertools import permutations


class BackGammonGame:
    """
    Game of BackGammon in which we have:
        Doubling Dice
        home_board_length: 6
        out_board_length: 6

        Starting Configuration:
        Red Home Board  | Outer Board
        2w 0 0 0 0 6r | 0 3r 0 0 0 6w
        2r 0 0 0 0 6w | 0 3w 0 0 0 6r
        White Home Board| Outer Board

                        <--White Going | Red Going-->
        White Home        Out            out            Red Home
        [-2,0,0,0,0,6,    0,3,0,0,0,-6,  6,0,0,0,-3,0,  -6,0,0,0,0,2]
    """
    
    def __init__(self, board: list[int], bar: dict[str:int], off: dict[str:int], player: str):

        # Validate Board
        if not isinstance(board, list):
            raise TypeError("board must be a list")
        if len(board) != 24:
            raise ValueError(f"board must have length 24")
        if not all(isinstance(x, int) for x in board):
            raise TypeError("all elements be int")

        # Validate Bar
        if not (isinstance(bar, dict)):
            raise TypeError()
        if not set(bar.keys()) == {"white", "red"}:
            raise ValueError()
        if not (all(isinstance(bar[color], int) for color in ("white", "red"))):
            raise TypeError("bar must be a dict of {'white': int, 'red': int}")

        # Validate Off
        if not (isinstance(off, dict)):
            raise TypeError()
        if not set(off.keys()) == {"white", "red"}:
            raise ValueError()
        if not (all(isinstance(off[color], int) for color in ("white", "red"))):
            raise TypeError("off must be a dict of {'white': int, 'red': int}")
        
        # Validate Player
        if not isinstance(player, str):
            raise TypeError("player must be a string")
        if player not in ("white", "red"):
            raise ValueError("player must be either 'white' or 'red'")

        #                                   <--White Going | Red Going-->
        #                  White Home        Out            out            Red Home
        self.board = [-2,0,0,0,0,6,    0,3,0,0,0,-6,  6,0,0,0,-3,0,  -6,0,0,0,0,2]
        self.bar = {'white':0, 'red':0}
        self.off = {'white':0, 'red':0}
        self.player = player

    @classmethod
    def new_default(cls):
        """init default game"""
        return cls([-2,0,0,0,0,6,    0,3,0,0,0,-6,  6,0,0,0,-3,0,  -6,0,0,0,0,2], {"white": 0, "red": 0}, {"white": 0, "red": 0}, "white")

    def get_valid_move_sequences(board: list[int], bar: dict[str, int], off: dict[str, int], player: str, dice: tuple[int, int]):
            '''
            Returns all valid move sequences from a game state and dice roll

            Args
            - board: list[int] length 24
            - dice: list[int] either [d1,d2] or [d] for a double (we will expand doubles to 4)
            - player: 'white' or 'red'
            - bar: {'white': int, 'red': int}
            - off: {'white': int, 'red': int}

            Returns
            - list[list[]]
            '''

            #Double dice if double is rolled
            if len(dice) == 1:
                dice_seq = [dice[0]] * 4 #Idk how ill be passing doubles in yet
            elif len(dice) == 2 and dice[0] == dice[1]:
                dice_seq = [dice[0]] * 4 #Idk how ill be passing doubles in yet
            else:
                dice_seq = dice[:]  # two different dice
        
            #Get all possible dice orders (will de-dupe duplicates later)
            all_orders = set(permutations(dice_seq))
        
            direction = -1 if player == 'white' else 1
            opponent = 'red' if player == 'white' else 'white'

            def point_has_player_checker(temp_bd, p):
                    return temp_bd[p] > 0 if player == 'white' else temp_bd[p] < 0
            
            def can_land_on(temp_bd, dest):
                """True if player may land on dest (empty, own, or exactly 1 opponent)."""
                val = temp_bd[dest]
                if player == "white": 
                    return val >= -1 # 1 or less red
                else:
                    return val <= 1 # 1 or less white

            def enter_from_bar_dest_index(die):
                """If moving a checker from the bar, destination index for a die."""
                # For white: entry points are adjusted
                # die roll: 1 | 2 | 3 | 4 | 5 | 6
                # entry   : 0 | 1 | 2 | 3 | 4 | 5 or 23 | 22 | 21 | 20 | 19 | 18
                # for       red                  and white                       respectively
                return 24 - die if player == "white" else die - 1

            def all_checkers_in_home(temp_bd):
                """Return True if all player's checkers are in the home board (for bearing off)."""
                if player == 'white':
                    # white home points are indices 0..5
                    for i in range(6,24):
                        if temp_bd[i] > 0:
                            return False
                    return True
                else:
                    # red home points are indices 18..23
                    for i in range(0,18):
                        if temp_bd[i] < 0:
                            return False
                    return True

            def can_bear_off_from(temp_bd, point, die):
                """Check if a checker at 'point' may bear off using die."""

                dest = point + die * direction

                if player == 'white':
                    # dest < 0 = bearing off
                    if dest < 0:
                        if (die ==  point + 1): # Exact Roll, Off
                            return True
                        if (die > point + 1): # Higher Roll, No higher Pieces, Off
                            for i in range(point+1,6):
                                if temp_bd[i] > 0:
                                    return False
                            return True
                    else:
                        return False # Doesnt Bear Off
                else:  # red
                    # dest > 23 = bearing off
                    if dest > 23:
                        if (die ==  24 - point): # Exact Roll, Off
                            return True
                        if (die > 24 - point): # Higher Roll, No higher Pieces, Off
                            for i in range(point-1,17,-1):
                                if temp_bd[i] < 0:
                                    return False
                            return True
                    else:
                        return False # Doesnt Bear Off
                    
            def generate_moves_for_die(temp_board, temp_bar, temp_off, die):
                """
                Given Game State, return all single moves available for THIS die.
                move given as (from_index OR 'bar', to_index OR 'off', die).
                """
                moves = []
                # If checker on bar, only bar moves allowed
                if temp_bar[player] > 0:
                    dest = enter_from_bar_dest_index(die)
                    if 0 <= dest < 24 and can_land_on(temp_board, dest):
                        moves.append(('bar', dest, die))
                    return moves
                
                # Normal moves from points that have player's checkers
                else:
                    for i in range(0,24):
                        if not point_has_player_checker(temp_board, i):
                            continue # No piece here

                        dest = i + die * direction
                        
                        # Bearing off
                        if dest < 0 or dest > 23:
                            if all_checkers_in_home(temp_board) and can_bear_off_from(temp_board, i, die):
                                moves.append((i, 'off', die))
                            # can't bear off with this die, from this point
                            continue
                        
                        # Regular destination & move
                        if can_land_on(temp_board, dest):
                            moves.append((i, dest, die))

                    return moves
            
            def apply_move(temp_bd, temp_bar, temp_off, move):
                """
                Apply a SINGLE move to copies of (temp_bd, temp_bar, temp_off) and return them.
                move recieved as (from_index OR 'bar', to_index OR 'off', die).
                """
                temp_board2 = temp_bd[:]  # shallow copy
                temp_bar2 = dict(temp_bar)
                temp_off2 = dict(temp_off)

                src, dest, die = move

                # Remove checker from source
                if src == 'bar':
                    temp_bar2[player] -= 1
                else:
                    # Remove player's checker on source
                    if player == 'white':
                        temp_board2[src] -= 1
                    else:
                        temp_board2[src] += 1

                # Place at Destination/Off
                if dest == 'off':
                    temp_off2[player] += 1
                else:
                    # If hitting opponent, send it to bar
                    if player == 'white':
                        if temp_board2[dest] == -1:
                            temp_board2[dest] = 1
                            temp_bar2[opponent] += 1
                        else:
                            temp_board2[dest] += 1
                    else:
                        if temp_board2[dest] == 1:
                            temp_board2[dest] = -1
                            temp_bar2[opponent] += 1
                        else:
                            temp_board2[dest] -= 1

                return temp_board2, temp_bar2, temp_off2
            
            # Recursion to build sequences for a dice order
            # In BG you must use as many die as possible
            def sequences_for_order(dice_order):
                results = set()   # store sequences as tuples to delete duplicates
                max_moves_used = 0

                def backtrack(temp_board, temp_bar, off_state, die_pointer, move_seq):
                    nonlocal max_moves_used
                    # die_pointer is the next die index to play in dice_order

                    # Base Case: No More Die
                    if die_pointer >= len(dice_order):
                        if len(move_seq) >= max_moves_used:
                            if len(move_seq) > max_moves_used: # If current move_seq longer than existing, Use it (delete previous)
                                results.clear()
                                max_moves_used = len(move_seq)
                            results.add(tuple(move_seq)) # Add current seq (same length, or longest found)
                        return

                    die = dice_order[die_pointer]
                    moves = generate_moves_for_die(temp_board, temp_bar, off_state, die)

                    # Base Case: Cant Play Die
                    if not moves:
                        if len(move_seq) >= max_moves_used:
                            if len(move_seq) > max_moves_used:
                                results.clear()
                                max_moves_used = len(move_seq)
                            results.add(tuple(move_seq))
                        return

                    # Step Case: For each legal move, apply and continue
                    for move in moves:
                        temp_board2, temp_bar2, temp_off2 = apply_move(temp_board, temp_bar, off_state, move)
                        backtrack(temp_board2, temp_bar2, temp_off2, die_pointer + 1, move_seq + [move])

                # Start recursion with copies
                backtrack(deepcopy(board), deepcopy(bar), deepcopy(off), 0, [])
                return results, max_moves_used
            
            # Collect sequences across all dice orders
            all_sequences = set()
            overall_max_moves = 0
            for order in all_orders:
                seqs, moves_used = sequences_for_order(order)
                if moves_used > overall_max_moves:
                    overall_max_moves = moves_used
                    all_sequences = set(seqs)
                elif moves_used == overall_max_moves:
                    all_sequences.update(seqs)

            # Convert sequences (tuples) back to list-of-moves lists
            final_sequences = [list(seq) for seq in all_sequences]

            # Sort for deterministic output (optional)
            final_sequences.sort(key=lambda s: (len(s), s), reverse=True)

            return final_sequences

    def is_win(board: list[int], bar: dict[str, int], off: dict[str, int], player: str):
        '''
        Return if the current player has won the game

        Args
        - board: list[int] length 24
        - dice: list[int] either [d1,d2] or [d] for a double (we will expand doubles to 4)
        - player: 'white' or 'red'
        - bar: {'white': int, 'red': int}
        - off: {'white': int, 'red': int}

        Returns
        - Bool
        '''
        
        if (player == "white"):
            if off["white"] > 0:
                return False
            
            for i in enumerate(board):
                if i > 0:
                    return False
        else:
            if off["red"] > 0:
                return False
            
            for i in enumerate(board):
                if i < 0:
                    return False
        return True
        
    def get_next_state(board: list[int], bar: dict[str, int], off: dict[str, int], player: str, move_seq: list[tuple]):
        '''Note: This does not block illegal move sequences, and so may break if it recieves any'''

        opponent = 'red' if player == 'white' else 'white'

        def apply_move(board, bar, off, move):
                """
                Apply a SINGLE move to copies of (temp_bd, temp_bar, temp_off) and return them.
                move recieved as (from_index OR 'bar', to_index OR 'off', die).
                """
                src, dest, die = move

                # Remove checker from source
                if src == 'bar':
                    bar[player] -= 1
                else:
                    # Remove player's checker on source
                    if player == 'white':
                        board[src] -= 1
                    else:
                        board[src] += 1

                # Place at Destination/Off
                if dest == 'off':
                    off[player] += 1
                else:
                    # If hitting opponent, send it to bar
                    if player == 'white':
                        if board[dest] == -1:
                            board[dest] = 1
                            bar[opponent] += 1
                        else:
                            board[dest] += 1
                    else:
                        if board[dest] == 1:
                            board[dest] = -1
                            bar[opponent] += 1
                        else:
                            board[dest] -= 1

                return board, bar, off
        
        for move in move_seq:
            board, bar, off = apply_move(board,bar,off,move)

        return board, bar, off, opponent
    
    def get_reward_for_player(self, board: list[int], bar: dict[str, int], off: dict[str, int], player: str):
        # return None if not ended, 1 if player 1 wins, -1 if player 1 lost

        opponent = 'red' if player == 'white' else 'white'

        if self.is_win(board, bar,off,player):
            return 1
        if self.is_win(board, bar,off,opponent):
            return -1
        if self.has_legal_moves(board):
            return None

        return 0
    
    def __str__(self):
        
        board_str = f"""
            +----+----+----+----+----+----+    +----+----+----+----+----+----+
            +----+----+----+----+----+----+    +----+----+----+----+----+----+
            +----+----+----+----+----+----+    +----+----+----+----+----+----+
            """
    
        return board_str