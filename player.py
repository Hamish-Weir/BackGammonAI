from game import BackGammonGame


class Player:

    def get_next_move(self,game: BackGammonGame, dice: tuple[int, int]):
        valid_action_sequences = game.get_valid_move_sequences(dice)
        while True:
            action_sequence = []

            print("Enter Move (Position, Die): ",end='')
            input_move = input()

            if action_sequence in valid_action_sequences:
                return action_sequence