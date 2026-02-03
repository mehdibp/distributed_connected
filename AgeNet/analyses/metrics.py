import numpy as np
import networkx as nx

from typing import List
from ..core.agent import Agent


class NetworkMetrics:
    def compute(self, agents: List[Agent], adjacency: np.ndarray):
        N = len(agents)
        G = nx.from_numpy_array(adjacency)

        hamilton = sum(agent.energy for agent in agents)    # Hamiltonian of the whole system
        energy   = sum([agent.hamiltonian_model.a3*agent.r**2 for agent in agents])   # Wave energy consumption
        edges    = adjacency.sum() / 2                      # Total number of established links between agents
        avg_r    = np.mean([agent.r for agent in agents])   # Average transition radius

        giant    = len(max(nx.connected_components(G), key=len)) / N * 100      # Giant component of network (%)

        return {
            "hamiltonian"       : hamilton,
            "wave_energy"       : energy,
            "edges"             : edges,
            "average_r"         : avg_r,
            "giant_component"   : giant
        }
    
