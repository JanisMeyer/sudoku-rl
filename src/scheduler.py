import math

class EpsilonScheduler:
    def __init__(self, start_value, end_value, duration):
        self.start_value = start_value
        self.end_value = end_value
        self.duration = duration

        self.current_step = 0

    def step(self):
        self.current_step += 1

    def get_value(self):
        return self.end_value + (self.start_value - self.end_value) * math.exp(-1 * self.current_step / self.duration)