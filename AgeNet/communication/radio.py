import numpy as np
from typing import List, Dict
from typing import TYPE_CHECKING
if TYPE_CHECKING: from ..core.agent import Agent


# Models packet-level communication behavior of an entity.
class RadioModel:
    # ---------------------------------------------------------------------------------------
    def __init__(self, agent: "Agent", base_delay: float = 0.01, max_delay : float = 1.0):
        self.agent = agent
        self.base_delay = base_delay
        self.max_delay  = max_delay

    # ---------------------------------------------------------------------------------------
    def send_message(self, distance: float) -> Dict:
        # Build a message object (Delivery decision is NOT made here)
        
        massage = {
            "sender"   : self.agent.id,
            "position" : self.agent.position,
            "speed"    : self.agent.speed,
            "radius"   : self.agent.r,
            "distance" : distance,
            "loss_prob": self.packet_loss_prob(distance),
            "delay"    : self.delay(distance),
        }

        return massage
    
    # ---------------------------------------------------------------------------------------
    def packet_loss_prob(self, distance: float) -> float:
        # Probability of packet loss as a function of distance and transmission radius

        if distance > self.agent.r: return 1.0
        return distance / self.agent.r

    # ---------------------------------------------------------------------------------------
    def delay(self, distance: float) -> float:
        d = self.base_delay + (distance / self.agent.r) * self.max_delay
        return min(d, self.max_delay)

    # ---------------------------------------------------------------------------------------
    def link_quality(self, distance: float) -> Dict:
        return {
            "distance"         : distance,
            "packet_loss_prob" : self.packet_loss_prob(distance),
            "expected_delay"   : self.delay(distance),
            "reachable"        : distance <= self.agent.r
        }

    # ---------------------------------------------------------------------------------------
    def broadcast(self, all_agents: List["Agent"]):
        for other in all_agents:
            if other is self.agent: continue
            distance = np.linalg.norm(self.agent.position - other.position)
            self.send_message(distance, self.agent.r)
