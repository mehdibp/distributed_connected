import numpy as np
from ..communication.channel import ChannelModel

from typing import TYPE_CHECKING
if TYPE_CHECKING: from ..core.agent import Agent


class StateExtractor:
    # ---------------------------------------------------------------------------------------    
    def __init__(self, agent: "Agent", channel: ChannelModel):
        self.agent   = agent
        self.channel = channel

    # ---------------------------------------------------------------------------------------    
    def mystate(self, obstacles) -> np.ndarray:
        k = 0
        rho = 0

        for other, distance in self.agent.neighbors:
            other: Agent
            blocked    = self.channel.intersect  ( other.position, obstacles )
            attenuated = self.channel.attenuation( distance, other.r, blocked )
            if not attenuated: k += 1

        _, d_max = max(self.agent.incoming_neighbors, key=lambda x: x[1], default=(None, self.agent.r))
        rho = (len(self.agent.incoming_neighbors)+1) / (np.pi*d_max**2)
        if len(self.agent.incoming_neighbors) == 0 and rho > 1: rho = 0.1
        
        return np.array([k, self.agent.r, rho], dtype=np.float32)
