from typing import Tuple, List

from typing import TYPE_CHECKING
if TYPE_CHECKING: from ..core.agent import Agent


class Hamiltonian:
    # -------------------------------------------------------------------------------------
    def __init__(self, alphas: Tuple[float, float, float, float] = (-0.5, +0.3, 1.0, -1000)):
        self.a1, self.a2, self.a3, self.a4 = alphas

    # -------------------------------------------------------------------------------------
    def __call__(self, k: int, r: float, neighbors: List[Tuple["Agent", float]]) -> float:
        fourth = sum( 1./d for _, d in neighbors )
        H = self.a1 * k**2 + self.a2 * k**3 + self.a3 * r**2 + self.a4 * fourth

        return H
