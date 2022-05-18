from random import random, randint
from time import time

import torch

from .sudoku import is_correct_finished, is_finished, is_correct, get_actions, get_next_states, apply_action
from .scheduler import EpsilonScheduler

class Runner:
    def __init__(self, device, model, target_model=None, buffer=None, learning_rate=1e-3, epsilon=EpsilonScheduler(0.9, 0.05, 10000)):
        self.device = device
        self.model = model.to(device)
        if target_model: # Target Model for Q-Learning
            self.target_model = target_model.to(device)
            self.target_model.load_state_dict(self.model.state_dict()) # Copy Initialization from actual model
        else:
            self.target_model = None
        # Replay Buffer
        self.buffer = buffer
        
        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.discount = 0.9
        self.epsilon = epsilon
        self.tau = 0.999

    def train(self, num_episodes, sudokus, num_eval=50, report_episodes=100):
        assert self.target_model is not None, "target model needed for Q-Learning"
        assert self.buffer is not None, "Replay Buffer needed for training"
        self.model.train()
        self.target_model.eval() # This seems to work best in the literature/online examples 
        
        total_loss = 0.0
        num_steps = 0
        start_time = time()
        for episode in range(1, num_episodes+1):
            puzzle = sudokus.get_random()
            while not is_finished(puzzle) and is_correct(puzzle):
                action, next_state = self.get_sample(puzzle) # sample next action
                puzzle = apply_action(puzzle, action)
                reward = get_reward(puzzle)
                
                self.buffer.append(next_state, reward) # add new sample to replay buffer
                
                if self.buffer.is_ready(): # if enough samples to fill batch
                    next_states, rewards = self.buffer.get_batch()
                    non_final = [torch.any(next_state == 0) for next_state in next_states]
                    
                    self.optimizer.zero_grad()
                    q_values = self.model(next_states.to(self.device))
                    max_states = []
                    with torch.no_grad():
                        for next_state in next_states[non_final, ...]: # for every state ..
                            possible_actions = get_actions({"current": next_state}) # .. get possible actions ..
                            next_next_states = get_next_states({"current": next_state}, possible_actions).to(self.device) # .. and their states ..
                            max_states.append(next_next_states[torch.argmax(self.model(next_next_states))]) # .. evaluate them and pick the one with maximum value using actual model
                        target_q_values = self.target_model(torch.stack(max_states, dim=0)) # and calculate the evaluation with the target model (Double Q-Learning)
                    # Calculate target
                    target = rewards.to(self.device)
                    target[non_final] += self.discount * target_q_values
                    
                    loss = self.criterion(q_values, target)
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_steps += 1
                    
                    # Update the target model as EMA of actual model
                    for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                        target_param.data.copy_(self.tau * target_param.data + (1 - self.tau) * param.data)
            self.epsilon.step()
            if episode % report_episodes == 0:
                print("Episode %d/%d; Loss: %0.4f; Elapsed: %0.2fs; Num Steps: %d" % (episode, num_episodes, total_loss / num_steps, time() - start_time, num_steps))
                start_time = time()
                total_loss = 0.0
                num_steps = 0
                if num_eval > 0:
                    self.eval(num_eval, sudokus)
                    self.model.train()
                torch.save({"model": self.model.state_dict()}, "model.pt")
        print("Finished training after %d episodes" % num_episodes)

    # evaluate the model using num_episode puzzles
    def eval(self, num_episodes, sudokus):
        self.model.eval()
        num_correct = 0
        num_guesses = 0
        num_finished = 0
        reward = 0
        for _ in range(1, num_episodes + 1):
            puzzle = sudokus.get_random()
            while not is_finished(puzzle) and is_correct(puzzle):
                possible_actions = get_actions(puzzle)
                next_states = get_next_states(puzzle, possible_actions).to(self.device)
                with torch.no_grad():
                    q_values = self.model(next_states)
                action = possible_actions[torch.argmax(q_values, dim=0)]
                puzzle = apply_action(puzzle, action)
                reward += get_reward(puzzle)
                num_guesses += 1
                if is_correct(puzzle):
                    num_correct += 1
                if is_correct_finished(puzzle):
                    num_finished += 1
        print("Evaluation on %d episodes -- Avg. reward: %0.2f -- Accuracy: %0.4f -- Perc. Finished: %0.4f" % 
              (num_episodes, reward / num_guesses, num_correct / num_guesses, num_finished / num_episodes))
    
    # Test the model for a single puzzle and return the prediction
    def test(self, puzzle):
        while not is_finished(puzzle) and is_correct(puzzle):
            possible_actions = get_actions(puzzle)
            possible_actions = get_actions(puzzle)
            next_states = get_next_states(puzzle, possible_actions).to(self.device)
            with torch.no_grad():
                q_values = self.model(next_states)
            action = possible_actions[torch.argmax(q_values, dim=0)]
            puzzle = apply_action(puzzle, action)
        return puzzle
    
    # Get next Action based on epsilon-greedy exploration strategy
    def get_sample(self, puzzle):
        possible_actions = get_actions(puzzle)
        if random() > self.epsilon.get_value(): # Exploitation (best action) 
            next_states = get_next_states(puzzle, possible_actions).to(self.device)
            with torch.no_grad():
                q_values = self.model(next_states)
            idx = torch.argmax(q_values, dim=0)
            action, next_state = possible_actions[idx], next_states[idx]
        else: # Exploration, random action
            action = possible_actions[randint(0, len(possible_actions) - 1)]
            next_state = get_next_states(puzzle, [action])[0].to(self.device)
        return action, next_state

# very simple reward function
def get_reward(puzzle):
    if is_correct_finished(puzzle):
        return 10.0
    elif is_correct(puzzle):
        return 5.0
    else:
        return -1.0