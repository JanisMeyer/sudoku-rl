from random import randint

import torch
import numpy as np

class Sudokus:
    def __init__(self, path):
        self.data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=str)

    def get_next(self, idx):
        return {
            "current": torch.from_numpy(np.array([int(d) for d in self.data[idx, 0]])).long().view(9, 9),
            "solution": torch.from_numpy(np.array([int(d) for d in self.data[idx, 1]])).long().view(9, 9)
        }

    def get_random(self):
        return self.get_next(randint(0, len(self.data) - 1))

def is_valid_action(game_state, action):
    if action["value"] == 0:
        return False
    if game_state["current"][action["y"], action["x"]] != 0:
        return False
    if torch.any(game_state["current"][:, action["x"]] == action["value"]):
        return False
    if torch.any(game_state["current"][action["y"], :] == action["value"]):
        return False
    block_x = 3 * torch.div(action["x"], 3, rounding_mode="trunc")
    block_y = 3 * torch.div(action["y"], 3, rounding_mode="trunc")
    if torch.any(game_state["current"][block_y:block_y+3, block_x:block_x+3] == action["value"]):
        return False
    return True

def is_correct(game_state):
    return torch.all(game_state["current"][game_state["current"] != 0] == game_state["solution"][game_state["current"] != 0])

def is_correct_finished(game_state):
    return torch.all(game_state["current"] == game_state["solution"])

def num_blanks(game_state):
    return torch.sum(game_state["current"] == 0)

def is_finished(game_state):
    return num_blanks(game_state).item() == 0

def get_actions(game_state):
    actions = []
    coordinates = torch.nonzero(game_state["current"] == 0)
    state = torch.nn.functional.one_hot(game_state["current"], num_classes=10)
    
    rows = [torch.sum(state[i, ...], dim=0) for i in range(9)]
    cols = [torch.sum(state[:, i, ...], dim=0) for i in range(9)]
    blocks = [[torch.sum(torch.reshape(state[3*i:3*i+3, 3*j:3*j+3], (-1, 10)), dim=0) for j in range(3)] for i in range(3)]
    
    for coordinate in coordinates:
        block_y = torch.div(coordinate[0], 3, rounding_mode="trunc")
        block_x = torch.div(coordinate[1], 3, rounding_mode="trunc")
        values = rows[coordinate[0]] + cols[coordinate[1]] + blocks[block_y][block_x]
        actions += [{
            "y": coordinate[0],
            "x": coordinate[1],
            "value": value
        } for value in (values == 0).nonzero() if value != 0]
    return actions

def apply_action(game_state, action):
    new_state = {
        "solution": torch.clone(game_state["solution"]),
        "current": torch.clone(game_state["current"])
    }
    if is_valid_action(game_state, action):
        new_state["current"][action["y"], action["x"]] = action["value"]
    return new_state

def get_next_states(game_state, actions):
    next_states = []
    for action in actions:
        state = torch.clone(game_state["current"])
        state[action["y"], action["x"]] = action["value"]
        next_states.append(state)
    return torch.stack(next_states, dim=0)