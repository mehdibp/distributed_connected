import numpy as np
from typing import Tuple

from typing import TYPE_CHECKING
if TYPE_CHECKING: from ..core.agent import Agent


class RadiusController:
    # ---------------------------------------------------------------------------------------
    def __init__(self, agent: "Agent"):
        self.agent = agent
    

    # ---------------------------------------------------------------------------------------
    def apply_action(self, action: float) -> Tuple[float, float]:
        radius = self.agent.r

        action_dim  = self.agent.brain.action_dim
        action_dim_ = (action_dim-1)/2
        action = (action - action_dim_)/(action_dim_)   # between -1 and +1

        # rho = self.agent.rho if self.agent.rho != 0 else 1
        # delta_r = action* np.sqrt(1/rho)/4 *np.random.random()
        delta_r = action* np.sqrt(1/self.agent.rho)/4 *np.random.random()
        radius  = max(1e-10, radius + delta_r)
        if radius == 1e-10: radius = 1

        return radius, delta_r

    # ---------------------------------------------------------------------------------------
    def flip(self, delta_r: float, delta_H: float) -> float:
        radius = self.agent.r
        delta_H_clip = np.clip(-delta_H / 4, -20, 2)
        if np.exp(delta_H_clip) < np.random.random():
            radius -= delta_r * np.random.random()
            radius  = max(1e-10, radius)

        return radius
