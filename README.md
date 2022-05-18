# Reinforcement Learning Framework to solve sudokus

Very simple framework for the training of a reinforcement learning agent to solve sudokus. Expects a `.csv` file of sudokus with puzzle and solution in `data/sudoku.csv` such as this one found on kaggle [here](https://www.kaggle.com/datasets/bryanpark/sudoku).

Uses Double Q-Learning with a Replay buffer for training and a simple convolutional neural network. However a configuration which actually manages to learn has not been found yet.