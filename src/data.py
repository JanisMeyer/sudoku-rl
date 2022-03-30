from random import sample

import torch

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size=20):
        self.buffer = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        self.buffer_index = 0

    def append(self, next_state, reward):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((
                next_state,
                reward
            ))
        else:
            self.buffer[self.buffer_index] = (
                next_state,
                reward
            )
            self.buffer_index = (self.buffer_index + 1) % self.buffer_size
            
    def is_ready(self):
        return len(self.buffer) >= self.batch_size

    def get_batch(self):
        batch = sample(self.buffer, self.batch_size)
        batched_next = torch.stack([
            sample[0] for sample in batch
        ], dim=0)
        batched_reward = torch.Tensor([sample[1] for sample in batch])
        return batched_next, batched_reward