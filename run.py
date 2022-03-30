import click

import torch

from src.sudoku import Sudokus
from src.model import QModel
from src.runner import Runner
from src.data import ReplayBuffer


@click.group()
def cli():
    pass

@cli.command()
def train():
    device = torch.device("cuda")
    sudokus = Sudokus("data/sudoku.csv")
    runner = Runner(device, QModel(), QModel(), ReplayBuffer(10000))
    runner.train(1000, sudokus)

@cli.command()
def test():
    sudokus = Sudokus("data/sudoku.csv")
    puzzle = sudokus.get_next(0)
    print(puzzle["current"])
    print(puzzle["solution"])

if __name__ == "__main__":
    cli()