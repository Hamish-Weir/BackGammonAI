import argparse
import cProfile
import importlib
import pstats
import sys

from src.Colour import Colour
from src.Game import Game
from src.Player import Player

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Backgammon",
        description="Run a game of Backgammon. By default, two Random agents will play.",
    )
    parser.add_argument(
        "-p1",
        "--player1",
        default="agents.DefaultAgents.RandomAgent RandomAgent",
        type=str,
        help="Specify the player 1 agent, format: agents.AgentFile AgentClassName",
    )
    parser.add_argument(
        "-p2",
        "--player2",
        default="agents.DefaultAgents.RandomAgent RandomAgent",
        type=str,
        help="Specify the player 2 agent, format: agents.AgentFile AgentClassName",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "-l",
        "--log",
        nargs="?",
        type=str,
        default=sys.stderr,
        const="game.log",
        help=(
            "Save moves history to a log file,"
            "if the flag is present, the result will be saved to game.log."
            "If a filename is provided, the result will be saved to the provided file."
            "If the flag is not present, the result will be printed to the console, via stderr."
        ),
    )
    parser.add_argument(
        "-t",
        "--timelimitOff",
        action="store_true",
        help=(
            "Disable time limit"
        ),
    )

    args = parser.parse_args()
    p1_path, p1_class = args.player1.split(" ")
    p2_path, p2_class = args.player2.split(" ")
    p1 = importlib.import_module(p1_path)
    p2 = importlib.import_module(p2_path)
    g = Game(
        player1=Player(
            name=p1_class,
            agent=getattr(p1, p1_class)(Colour.RED),
        ),
        player2=Player(
            name=p2_class,
            agent=getattr(p2, p2_class)(Colour.BLUE),
        ),
        logDest=args.log,
        verbose=args.verbose,
        timelimitOff=args.timelimitOff,
    )

    if args.verbose:
        with cProfile.Profile() as pr:
            g.run()
            stats = pstats.Stats(pr)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.print_stats(20)
    else:
        g.run()
