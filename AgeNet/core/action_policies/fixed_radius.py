from .base import BasePolicy
from typing import TYPE_CHECKING
if TYPE_CHECKING: from ..agent import Agent


class FixedRadiusPolicy(BasePolicy):
    def __init__(self, agent: "Agent"):
        super().__init__(agent)



    def act(self) -> float:
        dim = self.agent.brain.action_dim
        action = (dim-1)/2
        return action

