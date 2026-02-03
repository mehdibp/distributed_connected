from ..environments.geometry import do_intersect

from typing import TYPE_CHECKING
if TYPE_CHECKING: from ..core.agent import Agent


class ChannelModel:
    # --------------------------------------------------------------------------------------- 
    def __init__(self, agent: "Agent", absorption: float):
        self.agent = agent
        self.absorption = absorption

    # --------------------------------------------------------------------------------------- 
    def intersect(self, other_point_position: float, obstacles) -> bool:
        return do_intersect([self.agent.position, other_point_position], obstacles)

    # --------------------------------------------------------------------------------------- 
    def attenuation(self, distance: float, other_point_r: float, blocked: bool) -> bool:
        crossing = (1.0 - self.absorption)

        if not blocked: return False
        if distance <= crossing * self.agent.r and distance <= crossing * other_point_r: return False
        return True
