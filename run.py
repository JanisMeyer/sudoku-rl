import click

import torch

from src.sudoku import Sudokus, is_correct_finished
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
def eval():
    device = torch.device("cuda")
    sudokus = Sudokus("data/sudoku.csv")
    state_dict = torch.load("model.pt")
    
    model = QModel()
    model.load_state_dict(state_dict["model"])
    runner = Runner(device, model, None, None)
    runner.eval(100, sudokus)

@cli.command()
def test():
    device = torch.device("cuda")
    sudokus = Sudokus("data/sudoku.csv")
    puzzle = sudokus.get_random()
    state_dict = torch.load("model.pt")
    
    model = QModel()
    model.load_state_dict(state_dict["model"])
    runner = Runner(device, model, None, None)
    prediction = runner.test(puzzle)
    
    print("Solved: %r" % is_correct_finished(puzzle).item())
    print("Initial puzzle:")
    print(puzzle["current"].numpy())
    print("Prediction:")
    print(prediction["current"].numpy())
    print("Solution:")
    print(puzzle["solution"].numpy())

if __name__ == "__main__":
    cli()