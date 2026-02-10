import numpy as np
from .base import BasePolicy
from typing import TYPE_CHECKING
if TYPE_CHECKING: from ..agent import Agent


class MinDegreePolicy(BasePolicy):
    def __init__(self, agent: "Agent", k: int=5):
        super().__init__(agent)
        self.k = k

    def act(self) -> float:
        dim = self.agent.brain.action_dim
        if self.agent.k < self.k: action = np.random.choice([(dim-1)/2, dim-1])
        else: action = (dim-1)/2
        
        return action
    