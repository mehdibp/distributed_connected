import numpy as np
from typing import Tuple, List

from typing import TYPE_CHECKING
if TYPE_CHECKING: from ..core.agent import Agent


# Pure geometric neighbor discovery.
class NeighborFinder:
    # ---------------------------------------------------------------------------------------    
    def __init__(self, agent: "Agent"):
        self.agent = agent


    # ---------------------------------------------------------------------------------------
    def outcoming_neighbors(self, all_agents: List["Agent"]) -> List[Tuple["Agent", float]]:
        # Those to whom I can send messages
        outcoming_neighbors = []
        
        for other in all_agents:
            if other is self.agent: continue

            distance = self._distance_to(other)

            # If this condition is true, this agent is in the other communication scope
            if distance <= self.agent.r:
                # radio   = RadioModel(self.agent)
                # massage = radio.send_message(distance)
                # neighbors.append( massage )
                outcoming_neighbors.append( (other, distance) )

        return outcoming_neighbors

    # ---------------------------------------------------------------------------------------
    def incoming_neighbors(self, all_agents: List["Agent"]) -> List[Tuple["Agent", float]]:
        # Those whose message can reach me
        incoming_neighbors = []

        for other in all_agents:
            if other is self.agent: continue
            
            distance = self._distance_to(other)

            # If this condition is true, this agent is in the other communication scope
            if distance <= other.r:
                # radio   = RadioModel(other)
                # massage = radio.send_message(distance)
                # neighbors.append( massage )
                incoming_neighbors.append( (other, distance) )

        return incoming_neighbors
    
    # ---------------------------------------------------------------------------------------
    def neighbors(self, all_agents: List["Agent"]) -> List[Tuple["Agent", float]]:
        # Someone who has a two-way link
        connected_neighbors = []
        incoming_neighbors = self.incoming_neighbors(all_agents)

        for neighbor, distance in incoming_neighbors:
            if distance <= self.agent.r:
                connected_neighbors.append( (neighbor, distance) )

        return connected_neighbors
    
    # ---------------------------------------------------------------------------------------  
    def _distance_to(self, other: "Agent") -> float:
        return np.linalg.norm(self.agent.position - other.position)
