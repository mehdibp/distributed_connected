import numpy as np
from ..environments.base import Environment


class PhysicalState:
    # ---------------------------------------------------------------------------------------
    def __init__(self, environment: Environment, entity_id: str):
        self.environment = environment

        self._entity_id = entity_id

        self._position : np.ndarray | None=None
        self._speed:     float | None=None
        self._direction: float | None=None
        self._edge:      float | None=None

        self._init_position()

    # getters -------------------------------------------------------------------------------
    def get_entity_id(self) -> str       : return self._entity_id
    def get_position (self) -> np.ndarray: return self._position.copy()
    def get_speed    (self) -> float     : return self._speed
    def get_direction(self) -> float     : return self._direction
    def get_edge     (self) -> float     : return self._edge

    # setters -------------------------------------------------------------------------------
    def set_position (self, position: np.ndarray):
        self._position = position.astype(float)
    def set_direction(self, direction: float):
        self._direction = direction
    def set_speed(self, speed: float):
        self._speed = speed
    def set_edge (self, edge: str):
        self._edge = edge

    # ---------------------------------------------------------------------------------------
    def _init_position(self):
        # Initialize agent's position avoiding buildings.
        while True:
            self._position = np.random.rand(2)*self.environment.L   # Initialize (xᵢ, yᵢ) (positions of agent in plan)
            if self.environment.is_valid_position(self._position): break
