import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod
from ..environments.base import Environment


class MobilityModel():
    """
    Base mobility model.
    Owns dynamics, but not the state itself.
    """
    # ---------------------------------------------------------------------------------------
    def __init__(self, environment: Environment):
        self.environment = environment

    # ---------------------------------------------------------------------------------------
    @abstractmethod
    def move(self, position: np.ndarray) -> Tuple[np.ndarray, float, float]: pass




class NonMarkovian(MobilityModel):
    # ---------------------------------------------------------------------------------------
    def __init__(self, environment: Environment, speed: float=0.2025, randomness: float=0.4):
        super().__init__(environment)
        
        self.speed = speed                  # or stride = 0.01*(self.L**2/self.N)
        self.randomness = randomness
        self.radian = np.random.uniform(-np.pi, np.pi)

    # ---------------------------------------------------------------------------------------
    def move(self, position: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Simple non-Markovian random motion with reflective boundary
        if np.random.rand() > self.randomness:
            dx = self.speed * np.cos(self.radian)
            dy = self.speed * np.sin(self.radian)

        else: 
            dx = self.speed * (np.random.random() - 0.5) * 2
            dy = self.speed * (np.random.random() - 0.5) * 2

        new_position = position + np.array([dx, dy])

        # apply space constraints 
        position, self.radian = self.environment.apply_constraints(new_position, position, self.radian, self.speed)

        return position, self.speed, self.radian




# # -------------------------------------------------------------------------------------------
# class MobilityModel():
#     # ---------------------------------------------------------------------------------------
#     def __init__(self, movement_model: str, environment: Environment, 
#                  speed: float=0.2025, randomness: float=0.4):
#         if movement_model == 'NonMarkovian': self.movement_model = NonMarkovian(environment, speed, randomness)
#         elif ...
#         else: raise ValueError(...)
#         self.environment = environment

#     # ---------------------------------------------------------------------------------------
#     def __call__(self): return self.movement_model
