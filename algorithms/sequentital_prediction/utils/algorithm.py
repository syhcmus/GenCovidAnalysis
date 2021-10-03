from sys import _current_frames
from utils.result import result

class algorithm:

    def __init__(self, name, use_steps):
        self.use_steps = use_steps
        self.name = name

        if self.use_steps:
            self.steps = []
            self.current_step = -1

        else:
            self.result = result()


    def get_use_steps(self):
        return self.use_steps

    def add_step(self):
        pass

    def get_name(self):
        return self.name

    def get(self, stat):
        if self.use_steps:
            return self.steps[self.current_step].get(stat)
        else:
            return self.result.get(stat)

    def set(self, stat, value):
        if self.use_steps:
            self.steps[self.current_step].set(stat,value)
        else:
            self.result.set(stat,value)