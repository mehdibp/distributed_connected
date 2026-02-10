# ============================================================
# Abstract physical environment interface
# ============================================================

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import numpy as np


# -------------------------------------------------------------------------------------------
class BaseEnvironment(ABC):
    """
    Abstract interface for ALL physical environments.
    Agents are only allowed to see this API.
    """

    # lifecycle -----------------------------------------------------------------------------
    @abstractmethod
    def start(self): 
        pass
    
    @abstractmethod
    def close(self): 
        pass
    
    @abstractmethod
    def reset(self):
        """Reset environment to initial state."""
        pass
    
    @abstractmethod
    def step(self):
        """Advance environment by one simulation step."""
        pass

    # geometry ------------------------------------------------------------------------------
    @abstractmethod
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Return (xmin, ymin, xmax, ymax)."""
        pass

    # agent queries -------------------------------------------------------------------------
    @abstractmethod
    def get_entity_ids(self) -> List[str]:
        """Return IDs of active agents / vehicle."""
        pass

    @abstractmethod
    def get_position(self, entity_id: str) -> np.ndarray:
        """Return (x, y) position."""
        pass

    @abstractmethod
    def get_speed(self, entity_id: str) -> float:
        """Return sqrt(v_x**2, v_y**2) speed."""
        pass

    @abstractmethod
    def get_direction(self, entity_id: str) -> float | None:
        """Return preferred direction of movement."""
        pass

    @abstractmethod
    def get_edge(self, entity_id: str) -> str | None:
        """Return steet or lane it is located on."""
        pass
    
    @abstractmethod
    def get_time(self) -> float:
        """Return world simulation time."""
        pass

    @abstractmethod
    def get_density(self) -> float:
        """Return agent density per unit area."""
        pass




# -------------------------------------------------------------------------------------------
from dataclasses import dataclass
@dataclass
class Entity:
    position  : np.ndarray
    speed     : float
    direction : float | None
    edge      : str   | None


# -------------------------------------------------------------------------------------------
class Environment(BaseEnvironment, ABC):
    """ Base class for environments that expose entity-based physical state. """

    def __init__(self):
        self._entities: Dict[str, Entity] = {}
        self._buildings    = []
        self._build_bounds = []
        self._time: float = 0.0

    # agent queries - Public API (used by controller / state extractor) ---------------------
    def get_entity_ids(self)                 -> List[str]   :
        return list(self._entities.keys())
    def get_position  (self, entity_id: str) -> np.ndarray  : 
        return self._entities[entity_id].position
    def get_speed     (self, entity_id: str) -> float       : 
        return self._entities[entity_id].speed
    def get_direction (self, entity_id: str) -> float | None: 
        return self._entities[entity_id].direction
    def get_edge      (self, entity_id: str) -> str   | None: 
        return self._entities[entity_id].edge 
    def get_time      (self)                 -> float       : 
        return self._time
    
