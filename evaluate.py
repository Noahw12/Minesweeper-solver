"""Win rate of a trained agent on a fixed, reproducible set of boards.

    python evaluate.py expert
    python evaluate.py beginner --games 100000
    python evaluate.py intermediate --model models/intermediate/rainbow_intermediate_1.9
"""
import argparse

import minesweeper as ms
#Importing it also registers the env ids with gymnasium
import minesweeper_env
from rainbow import Rainbow
from utils import greedy_win_rate

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('difficulty', choices=['beginner', 'intermediate', 'expert'])
parser.add_argument('--model', help="defaults to models/<difficulty>/<difficulty>_model")
parser.add_argument('--games', type=int, default=10000)
parser.add_argument('--seed', type=int, default=1_000_000, help="board i is seeded with seed + i")
args = parser.parse_args()

path = args.model or f"models/{args.difficulty}/{args.difficulty}_model"
model = Rainbow.load(path)
rate = greedy_win_rate(model, ms.Difficulty[args.difficulty.upper()], args.games, args.seed)
se = (rate * (1 - rate) / args.games) ** 0.5
print(f"{path}: win rate {rate:.4f} +/- {se:.4f} over {args.games:,} games")
