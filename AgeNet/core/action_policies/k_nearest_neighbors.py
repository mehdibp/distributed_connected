from .base import BasePolicy
from typing import TYPE_CHECKING
if TYPE_CHECKING: from ..agent import Agent


class KNNPolicy(BasePolicy):
    def __init__(self, agent: "Agent", k: int=5):
        super().__init__(agent)
        self.k = k


    def act(self) -> float:
        dim = self.agent.brain.action_dim
        incoming_neighbors = sorted(self.agent.incoming_neighbors, key=lambda x: x[1])

        if len(incoming_neighbors) < self.k: action = dim-1
        else:
            _, d_k = incoming_neighbors[self.k - 1]

            if   self.agent.r < d_k: action = dim-1
            elif self.agent.r > d_k: action = 0
            else: action = (dim-1)/2

        return action
    
    