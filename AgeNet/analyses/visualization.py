import networkx as nx
from celluloid import Camera
import matplotlib.pyplot as plt

from typing import List
from .topology import NetworkTopology
from ..core.agent import Agent
from ..environments.base import Environment


class NetworkVisualizer:
    # ---------------------------------------------------------------------------------------
    def __init__(self, environment: Environment):
        self.environment = environment
        if hasattr(environment, "_buildings"): self._buildings = environment._buildings
        else: self._buildings = []

        self.topology = NetworkTopology(environment)

        fig, self.ax = plt.subplots(figsize=(14,14))
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=None, hspace=None)
        self.camera = Camera(fig)


        # Fixed Axes settings (only once) -----
        _, _, xmax, ymax = self.environment.get_bounds()
        self.ax.set_xlim(-0.02 * xmax, 1.02 * xmax)
        self.ax.set_ylim(-0.02 * ymax, 1.02 * ymax)


    # ---------------------------------------------------------------------------------------
    def draw(self, agents: List[Agent], step):
        adjacency = self.topology.adjacency(agents)
        G = nx.from_numpy_array(adjacency)
        options = { 'node_size': 60, 'width': 1.5, 'node_color': [(0.5,0.0,0.8,1)]*G.number_of_nodes() }

        # for i, agent in enumerate(agents):
        #     if not (i in (list((sorted(nx.connected_components(G), key=len, reverse=True))[0]))): 
        #         plt.gca().add_artist(plt.Circle(agent.position, radius=agent.r, color='#66338033'))


        pos = {i: agent.position for i, agent in enumerate(agents)}
        nx.draw_networkx(G, pos, with_labels=True, ax=self.ax, **options)
        
        self._draw_buildings(step)
        self.camera.snap()

    # ---------------------------------------------------------------------------------------
    def _draw_buildings(self, step: int):
        for build in self._buildings:
            x, y, w, h = build
            self.ax.add_patch( plt.Rectangle((x, y), width=w, height=h, fill=True, color='#146464cc', ec="black") )

        _, _, xmax, ymax = self.environment.get_bounds()
        self.ax.text(0.9*xmax, 1.03*ymax, f'Episode {step}', fontsize=12)
        self.ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        self.ax.grid(alpha=0.3)


