from game import BackGammonGame 
from player import Player
from random import randrange
from random import seed

def printturn(game,dice):
    remaining_dice_str = "".join(f"{die:4d}" for die in dice)
    
    print(game)
    print()
    print(f"Die Rolled: {remaining_dice_str}")

game = BackGammonGame.new_default()

player1 = Player()
player2 = Player()

current_player = player1
other_player = player2

seed(0)

while(game.get_winner() is None):
    
    die = (randrange(1,7),randrange(1,7))
    printturn(game,die)

    move_sequence = current_player.get_next_move(game,die)

    if move_sequence in game.get_valid_move_sequences(die):
        game.make_move_sequence(move_sequence)
    else:
        raise ValueError("Invalid Move: Exiting")
    
    current_player, other_player = other_player, current_player

# moves = BackGammonGame.get_valid_move_sequences(new_game, (2,1))

# for move in moves:
#     print(move)
