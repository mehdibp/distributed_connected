import numpy as np

from typing import List
from ..core.agent import Agent
from ..environments.base import Environment
from ..communication.channel import ChannelModel


class NetworkTopology:
    def __init__(self, environment: Environment):
        if hasattr(environment, "buildings"): self.obstacles = environment._build_bounds
        else: self.obstacles = []


    def adjacency(self, agents: List[Agent]) -> np.ndarray:
        N = len(agents)
        A = np.zeros((N, N))

        for i, agent in enumerate(agents):
            for other, distance in agent.neighbors:
                other: Agent
                j = agents.index(other) if other in agents else None

                channel: ChannelModel = agent.channel
                blocked    = channel.intersect  ( other.position, self.obstacles )
                attenuated = channel.attenuation( distance, other.r, blocked )
                if not attenuated: A[i, j] = A[j, i] = 1

        return A
