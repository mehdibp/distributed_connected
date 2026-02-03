import numpy as np
from typing import Tuple, List
from ..learning.hamiltonian import Hamiltonian

from typing import TYPE_CHECKING
if TYPE_CHECKING: from ..core.agent import Agent


class RadiusRequestPolicy:
    # -------------------------------------------------------------------------------------
    def __init__(self, agent: "Agent", hamilton: Hamiltonian):
        self.agent = agent
        self.hamilton = hamilton

    # -------------------------------------------------------------------------------------
    def decide(self, incoming_neighbors: List[Tuple["Agent", float]]) -> float:
        candidates = []
        radius = self.agent.r

        for incoming_neighbor in incoming_neighbors:
            _, distance = incoming_neighbor

            if distance > radius:
                delta_H = self.hamilton (
                    k = (self.agent.k+1)**2 - self.agent.k**2,
                    r = distance**2 - self.agent.r**2,
                    neighbors = [incoming_neighbor])
                
                candidates.append((distance, delta_H))

        d_best, delta_H = min(candidates, key=lambda x: x[1], default=(radius, 0))
        delta_H_clip = np.clip(-4*delta_H, -20, 2)
        if np.exp(delta_H_clip) > np.random.random():
            radius = d_best

        return radius
