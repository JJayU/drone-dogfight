from abc import ABC, abstractmethod

class Controller(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def compute_control(self):
        pass