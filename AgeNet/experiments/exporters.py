
import numpy as np
import networkx as nx
from typing import List, Any
from openpyxl import Workbook

from ..core.agent import Agent
from ..environments.base import Environment
from ..analyses.metrics import NetworkMetrics
from ..analyses.topology import NetworkTopology


class ResultExporter:
    """
    Collects, stores, and exports simulation results.
    This class is intentionally dumb: it only aggregates data
    and never affects the simulation itself.
    """

    # ---------------------------------------------------------------------------------------
    def __init__(self, environment: Environment):
        self.averaged_results: List[List[Any]] = []
        self.particle_results: List[List[List[Any]]] = []

        self.topology = NetworkTopology(environment)
        self.metrics  = NetworkMetrics()


    # ---------------------------------------------------------------------------------------
    def collect(self, step: int, agents: List[Agent]):
        """
        Collect results at a single timestep.

        Args:
            step (int): current simulation step
            agents (List[Agent]): list of agents
        """

        adjacency = self.topology.adjacency(agents)
        results   = self.metrics.compute(agents, adjacency)
        last_loss = np.mean([agent.brain.last_loss.numpy() for agent in agents])
        
        # Averaged Results --------------------------------------------------------------
        self.averaged_results.append([
            step,
            results['hamiltonian'],
            results['giant_component'],
            results['edges'],
            results['wave_energy'],
            results['average_r'],
            last_loss
        ])


        # Particle Results --------------------------------------------------------------
        G = nx.from_numpy_array(adjacency)
        components = list(nx.connected_components(G))

        step_agent_results = []
        for i, agent in enumerate(agents):
            component_id = next( (idx for idx, c in enumerate(components) if i in c), -1 )
            connected_to = list(np.where(adjacency[i] == 1)[0])
            last_exp = agent.replay_memory[-1] if agent.replay_memory else None

            step_agent_results.append([
                step,
                agent.id,
                agent.position[0],
                agent.position[1],
                component_id,
                agent.r,
                agent.k,
                agent.rho,
                agent.hamiltonian(),
                str(connected_to),
                str(last_exp[0]) if last_exp else None,   # state
                    last_exp[1]  if last_exp else None,   # action
                    last_exp[2]  if last_exp else None,   # reward
                str(last_exp[3]) if last_exp else None,   # next_state
                agent.brain.last_loss.numpy()
            ])

        self.particle_results.append(step_agent_results)

    # ---------------------------------------------------------------------------------------
    def export_excel(self, averaged_wb: Workbook, particle_wb: Workbook, sheet_name: str):
        """ Export collected results into Excel workbooks. """

        averaged_sheet = averaged_wb.create_sheet(title=sheet_name)
        particle_sheet = particle_wb.create_sheet(title=sheet_name)

        # Headers -----------------------------------------------------------------------
        averaged_sheet.append([ "step", "Hamiltonian", "Giant (%)", "Edges", "Energy", "Average_r", "Loss" ])
        particle_sheet.append([
            "step", "agent_id", "x", "y", "component", "r", "k", "rho", "Hamiltonian",
            "connected_to", "state", "action", "reward", "next_state", "loss"
        ])

        # Write data --------------------------------------------------------------------

        flattened_results = np.array(self.particle_results, dtype=object)
        flattened_results = flattened_results.reshape( -1, flattened_results.shape[-1] )

        for row in self.averaged_results: averaged_sheet.append(row)
        for row in     flattened_results: particle_sheet.append(list(row))

    # ---------------------------------------------------------------------------------------
    def reset(self):
        """ Clear all stored results. """
        self.averaged_results.clear()
        self.particle_results.clear()

