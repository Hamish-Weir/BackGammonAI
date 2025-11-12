from copy import copy, deepcopy
from itertools import permutations


class BackGammonGameState:
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
        [-2,0,0,0,0,5,    0,3,0,0,0,-5,  5,0,0,0,-3,0,  -5,0,0,0,0,2]
    """

    def __init__(
        self, board: list[int], bar: dict[str:int], off: dict[str:int], player: int
    ):

        if board:
            # Validate Board
            if not isinstance(board, list):
                raise TypeError("board must be a list")
            if len(board) != 24:
                raise ValueError("board must have length 24")
            if not all(isinstance(x, int) for x in board):
                raise TypeError("all elements be int")
            self.board = board
        else:
            #                                   <--White Going | Red Going-->
            #                  White Home        Out            out            Red Home
            self.board = [-2,0,0,0,0,5,    0,3,0,0,0,-5,  5,0,0,0,-3,0,  -5,0,0,0,0,2]

        if bar:
            # Validate Bar
            if not (isinstance(bar, dict)):
                raise TypeError()
            if not set(bar.keys()) == {1, -1}:
                raise ValueError()
            if not (all(isinstance(bar[color], int) for color in (1, -1))):
                raise TypeError(
                    "bar must be a dict of {1: int, -1: int}")
            self.bar = bar
        else:
            self.bar = {1: 0, -1: 0}

        if off:
            # Validate Off
            if not (isinstance(off, dict)):
                raise TypeError()
            if not set(off.keys()) == {1, -1}:
                raise ValueError()
            if not (all(isinstance(off[color], int) for color in (1, -1))):
                raise TypeError(
                    "off must be a dict of {1: int, -1: int}")
            self.off = off
        else:
            self.off = {1: 0, -1: 0}

        if player:
            # Validate Player
            if not isinstance(player, int):
                raise TypeError("player must be an int")
            if player not in (1, -1):
                raise ValueError("player must be either '1' or '-1'")
            self.player = player
            
        else:
            self.player = 1

        # #                                   <--White Going | Red Going-->
        # #                  White Home        Out            out            Red Home
        # self.board = [-2,0,0,0,0,6,    0,3,0,0,0,-6,  6,0,0,0,-3,0,  -6,0,0,0,0,2]
        # self.bar = {1:0, -1:0}
        # self.off = {1:0, -1:0}
        # self.player = player

    @classmethod
    def new_default(cls):
        """init default game"""
        return cls([-2,0,0,0,0,5,    0,3,0,0,0,-5,  5,0,0,0,-3,0,  -5,0,0,0,0,2], {1: 0, -1: 0}, {1: 0, -1: 0}, 1)


    def get_valid_move_sequences(self, dice: tuple[int, int]):
        """
        Returns all valid move sequences from a game state and dice roll

        Args
        - board: list[int] length 24
        - dice: list[int] either [d1,d2] or [d] for a double (we will expand doubles to 4)
        - player: '1' or '-1'
        - bar: {1: int, -1: int}
        - off: {1: int, -1: int}

        Returns
        - list[list[]]
        """
        game_copy = deepcopy(self)

        # Double dice if double is rolled
        if len(dice) == 1:
            dice_seq = [dice[0]] * 4  # Idk how ill be passing doubles in yet
        elif len(dice) == 2 and dice[0] == dice[1]:
            dice_seq = [dice[0]] * 4  # Idk how ill be passing doubles in yet
        else:
            dice_seq = dice[:]  # two different dice

        # Get all possible dice orders (will de-dupe duplicates later)
        all_orders = set(permutations(dice_seq))

        opponent = -game_copy.player

        def point_has_player_checker(temp_bd, player, p):
            return temp_bd[p]*player > 0

        def can_land_on(temp_bd, player, dest):
            """True if player may land on dest (empty, own, or exactly 1 opponent)."""
            val = temp_bd[dest]
            if player == 1:
                return val >= -1  # 1 or less red
            else:
                return val <= 1  # 1 or less white

        def get_dest(src,die,player):
            if src == "bar":
                return 24 - die if player == 1 else die - 1
            
            dest = src - die*player
            return dest

        def enter_from_bar_dest_index(die):
            """If moving a checker from the bar, destination index for a die."""
            # For white: entry points are adjusted
            # die roll: 1 | 2 | 3 | 4 | 5 | 6
            # entry   : 0 | 1 | 2 | 3 | 4 | 5 or 23 | 22 | 21 | 20 | 19 | 18
            # for       red                  and white                       respectively
            return 24 - die if self.player == 1 else die - 1

        def all_checkers_in_home(temp_bd, player):
            """Return True if all player's checkers are in the home board (for bearing off)."""
            if player == 1:
                # white home points are indices 0..5
                for i in range(6, 24):
                    if temp_bd[i] > 0:
                        return False
                return True
            else:
                # red home points are indices 18..23
                for i in range(0, 18):
                    if temp_bd[i] < 0:
                        return False
                return True

        def can_bear_off_from(temp_bd, player, src, die):
            """Check if a checker at 'point' may bear off using die."""

            dest = get_dest(src, die, player)

            if player == 1:
                # dest < 0 = bearing off
                if dest < 0:
                    if die == src + 1:  # Exact Roll, Off
                        return True
                    if die > src + 1:  # Higher Roll, No higher Pieces, Off
                        for i in range(src + 1, 6):
                            if temp_bd[i] > 0:
                                return False
                        return True
                else:
                    return False  # Doesnt Bear Off
            else:  # red
                # dest > 23 = bearing off
                if dest > 23:
                    if die == 24 - src:  # Exact Roll, Off
                        return True
                    if die > 24 - src:  # Higher Roll, No higher Pieces, Off
                        for i in range(src - 1, 17, -1):
                            if temp_bd[i] < 0:
                                return False
                        return True
                else:
                    return False  # Doesnt Bear Off

        def generate_moves_for_die(temp_game, die):
            """
            Given Game State, return all single moves available for THIS die.
            move given as (from_index OR 'bar', to_index OR 'off', die).
            """
            moves = []
            # If checker on bar, only bar moves allowed
            if temp_game.bar[temp_game.player] > 0:
                dest = get_dest("bar",die,temp_game.player)

                if 0 <= dest < 24 and can_land_on(temp_game.board, temp_game.player, dest):
                    moves.append(("bar", die))
                return moves

            # Normal moves from points that have player's checkers
            else:
                for src in range(0, 24):
                    if not point_has_player_checker(
                        temp_game.board, temp_game.player, src
                    ):
                        continue  # No piece here

                    dest = get_dest(src, die, temp_game.player)

                    # Bearing off
                    if dest < 0 or dest > 23:
                        if all_checkers_in_home(temp_game.board, temp_game.player) and can_bear_off_from(
                            temp_game.board,temp_game.player, src, die
                        ):
                            moves.append((src, die))
                        # can't bear off with this die, from this point
                        continue

                    # Regular destination & move
                    if can_land_on(temp_game.board, temp_game.player, dest):
                        moves.append((src, die))

                return moves

        def apply_move(temp_game, move):
            """
            Apply a SINGLE move to copies of (temp_bd, temp_bar, temp_off) and return them.
            move recieved as (from_index OR 'bar', to_index OR 'off', die).
            """
            temp_game2 = deepcopy(temp_game)

            src, die = move

            dest = get_dest(src, die, temp_game.player)
            
            # Remove checker from source
            if src == "bar":
                temp_game2.bar[temp_game2.player] -= 1
            else:
                # Remove player's checker on source
                if temp_game2.player == 1:
                    temp_game2.board[src] -= 1
                else:
                    temp_game2.board[src] += 1

            # Place at Destination/Off
            if dest < 0 or dest > 23:
                temp_game2.off[temp_game2.player] += 1
            else:
                # If hitting opponent, send it to bar
                if temp_game2.player == 1:
                    if temp_game2.board[dest] == -1:
                        temp_game2.board[dest] = 1
                        temp_game2.bar[opponent] += 1
                    else:
                        temp_game2.board[dest] += 1
                else:
                    if temp_game2.board[dest] == 1:
                        temp_game2.board[dest] = -1
                        temp_game2.bar[opponent] += 1
                    else:
                        temp_game2.board[dest] -= 1

            return temp_game2

        # Recursion to build sequences for a dice order
        # In BG you must use as many die as possible
        def sequences_for_order(dice_order):
            results = set()  # store sequences as tuples to delete duplicates
            max_moves_used = 0

            def backtrack(temp_game, die_pointer, move_seq):
                nonlocal max_moves_used
                # die_pointer is the next die index to play in dice_order

                # Base Case: No More Die
                if die_pointer >= len(dice_order):
                    if len(move_seq) >= max_moves_used:
                        if (
                            len(move_seq) > max_moves_used
                            # If current move_seq longer than existing, Use it (delete previous)
                        ):
                            results.clear()
                            max_moves_used = len(move_seq)
                        results.add(
                            tuple(move_seq)
                        )  # Add current seq (same length, or longest found)
                    return

                die = dice_order[die_pointer]
                moves = generate_moves_for_die(temp_game, die)

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
                    temp_game2 = apply_move(
                        temp_game, move
                    )
                    backtrack(
                        temp_game2,
                        die_pointer + 1,
                        move_seq + [move],
                    )

            # Start recursion with copies
            backtrack(game_copy, 0, [])
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
        def normalize_value(v):
            return -1 if v in ("off", "bar") else v

        final_sequences.sort(
            key=lambda s: (
                len(s),
                [tuple(normalize_value(x) for x in move) for move in s]
            ),
            reverse=True)
        
        return final_sequences

    def is_win(self, player):
        """
        Return if the current player has won the game

        Args
        - board: list[int] length 24
        - dice: list[int] either [d1,d2] or [d] for a double (we will expand doubles to 4)
        - player: '1' or '-1'
        - bar: {'1': int, '-1': int}
        - off: {'1': int, '-1': int}

        Returns
        - Bool
        """

        if self.bar[player] > 0: # Player has pieces on the bar
            return False

        for i in self.board: # Player has pieces on the board
            if i * player > 0:
                return False
            
        return True

    def make_move_sequence(self, move_seq: list[tuple],):
        """Note: This does not block illegal move sequences, and so may break if it recieves any"""

        opponent = -self.player

        
        def make_move(self, move):
            """
            Apply a SINGLE move to copies of (temp_bd, temp_bar, temp_off) and return them.
            move recieved as (from_index OR 'bar', to_index OR 'off', die).
            """

            def get_dest(src,die,player):
                if src == "bar":
                    return 24 - die if player == 1 else die - 1
                
                dest = src - die*player

                if dest < 0 or dest > 23:
                    return "off"
            
                return dest
            
            src, die = move

            dest = get_dest(src, die, self.player)

            # Remove checker from source
            if src == "bar":
                self.bar[self.player] -= 1
            else:
                # Remove player's checker on source
                if self.player == 1:
                    self.board[src] -= 1
                else:
                    self.board[src] += 1

            # Place at Destination/Off
            if dest == "off":
                self.off[self.player] += 1
            else:
                # If hitting opponent, send it to bar
                if self.player == 1:
                    if self.board[dest] == -1:
                        self.board[dest] = 1
                        self.bar[opponent] += 1
                    else:
                        self.board[dest] += 1
                else:
                    if self.board[dest] == 1:
                        self.board[dest] = -1
                        self.bar[opponent] += 1
                    else:
                        self.board[dest] -= 1

            return self

        for move in move_seq:
            make_move(self, move)

        self.player = -self.player
        
    def has_legal_moves(self):
        if (self.get_valid_move_sequences() and not (self.game_over() == None)):
            return True
        else:
            return False

    def get_reward_for_player(self):
        # return None if not ended, 1 if player 1 wins, -1 if player 1 lost

        if self.is_win(self.board, self.bar, self.off, 1):
            return 1
        elif self.is_win(self.board, self.bar, self.off, -1):
            return -1

        return 0

    def game_over(self):
        if self.is_win(1):
            return "White"
        elif self.is_win(-1):
            return "Red"
        return None

    def __str__(self):
        RED = "\033[91m"
        WHITE = "\033[0m"
        YELLOW = "\033[1;33m"

        BOARD_COLOUR = YELLOW
        PLAYER_COLOUR = WHITE if self.player == 1 else RED
        TO_PLAY = "White" if self.player == 1 else "Red"
        
        def get_spaced_str(list):
            return " ".join([f"{n:4d}" for n in list])

        def get_spaced_str_board(list):
            list2 = []
            for n in list:
                if n == 0:
                    list2.append(f"    ")
                elif n > 0:
                    list2.append(f"{WHITE}{n:4d}{WHITE}")
                else:
                    list2.append(f"{RED}{abs(n):4d}{WHITE}")
            return " ".join(list2)

        board_str = f"""                {WHITE}White Home                         {WHITE}Out
        {BOARD_COLOUR}OFF     {get_spaced_str([1, 2, 3, 4, 5, 6])}      {get_spaced_str([7, 8, 9, 10, 11, 12])}     BAR
                {BOARD_COLOUR}+----+----+----+----+----+----+    +----+----+----+----+----+----+
        {WHITE}{self.off[1]:4d}    {get_spaced_str_board(self.board[0:6])}      {get_spaced_str_board(self.board[6:12])}   {WHITE}{self.bar[1]:4d}
                {BOARD_COLOUR}+----+----+----+----+----+----+    +----+----+----+----+----+----+
        {RED}{self.off[-1]:4d}    {get_spaced_str_board(self.board[-1:-7:-1])}      {get_spaced_str_board(self.board[-7:-13:-1])}   {RED}{self.bar[-1]:4d}
                {BOARD_COLOUR}+----+----+----+----+----+----+    +----+----+----+----+----+----+
                {get_spaced_str([24, 23, 22, 21, 20, 19])}      {get_spaced_str([18, 17, 16, 15, 14, 13])}
                {RED}Red Home                           {RED}Out

            {BOARD_COLOUR}To Play: {PLAYER_COLOUR}{TO_PLAY}{WHITE}"""

        return board_str

