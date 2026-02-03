# ============================================================
# Handcrafted physical environment (random / regular buildings)
# ============================================================

import numpy as np
from typing import Tuple, List, Dict
from matplotlib.transforms import Bbox

from .base import Environment, Entity


class SimpleEnvironment(Environment):
    # ---------------------------------------------------------------------------------------
    def __init__(self, L: float, buildings_type="random", num_buildings=0, num_streets=0):

        super().__init__()

        self.L = L
        self.buildings    = []
        self.build_bounds = []

        if   buildings_type == "random" : self._random_buildings (num_buildings)
        elif buildings_type == "regular": self._regular_buildings(num_streets)
        else: raise ValueError( "buildings_type is invalid. Allowed values: 'random' or 'regular'." )


        self.physics = SimplePhysics(self.L, self.buildings)        # attach physics module

    # lifecycle -----------------------------------------------------------------------------
    def start(self): 
        self._time = 0.0

    def close(self): 
        self._entities.clear()

    def reset(self):
        self.close()
        self.start()

    def step(self) : 
        self._time += 1


    # entity management ---------------------------------------------------------------------
    def add_entity(self, entity_id: str, position: np.ndarray, speed: float, direction: float, edge: str | None):
        if entity_id in self._entities:
            raise ValueError(f"Entity '{entity_id}' already exists.")
        
        self._entities[entity_id] = Entity(
            position=position.astype(float),
            speed=speed,
            direction=direction,
            edge=edge
        )

    def remove_entity(self, entity_id: str):
        self._entities.pop(entity_id, None)

    def update_entity(self, entity_id: str, position: np.ndarray, speed: float, direction: float, edge: str | None):
        self._entities[entity_id] = Entity(
            position    = position.astype(float),
            speed       = speed,
            direction   = direction,
            edge        = edge
        )


    # geometry ------------------------------------------------------------------------------
    def get_bounds (self) -> Tuple[float, float, float, float]:
        """Return (xmin, ymin, xmax, ymax) of network."""
        return (0.0, 0.0, self.L, self.L)

    def get_density(self) -> float: 
        return len(self._entities)/self.L**2


    # internal ------------------------------------------------------------------------------
    def _random_buildings(self, num_buildings: int):
        while len(self.buildings) < num_buildings:
            x, y = np.random.randint(self.L, size=2)
            w, h = np.random.randint(2     , size=2) + self.L/15    # width, height

            if all(
                x + w < bx or x > bx + bw or
                y + h < by or y > by + bh
                for bx, by, bw, bh in self.buildings
            ):
                self.buildings   .append(np.array([x, y, w, h]))
                self.build_bounds.append(Bbox.from_bounds(x, y, w, h))

    def _regular_buildings(self, num_streets: int):
        if num_streets == 0: return
        
        Ns_Vertical   = int(np.ceil (num_streets/2))
        Ns_Horizontal = int(np.floor(num_streets/2))

        Ls_Vertical   = self.L / (4*Ns_Vertical-1)
        Ls_Horizontal = self.L / (3*Ns_Horizontal)

        for v in range(Ns_Vertical+1):
            for h in range(Ns_Horizontal+1):

                x = (4*v-2)*Ls_Vertical
                y = (3*h-1)*Ls_Horizontal
                w = 3*Ls_Vertical           # width
                h = 2*Ls_Horizontal         # height

                self.buildings   .append(np.array([x, y, w, h]))
                self.build_bounds.append(Bbox.from_bounds(x, y, w, h))


    # physics helpers (delegation) ----------------------------------------------------------
    def is_valid_position(self, position: np.ndarray) -> bool:
        return self.physics.is_valid_position(position)

    def apply_constraints(self, position: np.ndarray, previous_position: np.ndarray, radian: float, speed: float=0.):
        return self.physics.apply_constraints(position, previous_position, radian, speed)

    def boundary_condition(self, position: np.ndarray, radian: float, speed: float):
        return self.physics.boundary_condition(position, radian, speed)
    
    def buildings_boundary(self, position: np.ndarray, previous_position: np.ndarray, radian: float, speed: float):
        return self.physics.buildings_boundary(position, previous_position, radian, speed)
    




class SimplePhysics:
    # ---------------------------------------------------------------------------------------
    def __init__(self, L: float, buildings: List[np.ndarray]):
        self.L = L
        self.buildings = buildings

    # ---------------------------------------------------------------------------------------
    def is_valid_position(self, position: np.ndarray) -> bool:
        x, y = position

        if x < 0 or x > self.L or y < 0 or y > self.L: 
            return False

        for bx, by, w, h in self.buildings:
            if bx <= x <= bx + w and by <= y <= by + h:
                return False

        return True

    # ---------------------------------------------------------------------------------------
    def apply_constraints(self, position: np.ndarray, previous_position: np.ndarray, radian: float, speed: float=0.):
        """
        Apply space constraints to agent movement.

        Args:
            position          : proposed new position
            previous_position : last valid position
            radian            : current movement direction
            speed             : safety margin / agent radius

        Returns: (corrected_position, corrected_radian)
        """
        
        x, y = position
        xp, yp = previous_position

        x, y, radian = self.boundary_condition((x, y), radian, speed)
        x, y, radian = self.buildings_boundary((x, y), (xp, yp), radian, speed)

        return np.array([x, y]), radian

    # ---------------------------------------------------------------------------------------
    def boundary_condition(self, position: np.ndarray, radian: float, speed: float):
        x, y = position
        if (x < 0) or x > (self.L):
            radian = np.pi-radian if radian > 0 else -np.pi-radian
            x = np.clip(x, speed, self.L - speed)
        if (y < 0) or (y > self.L):
            radian = -radian
            y = np.clip(y, speed, self.L - speed)
            
        return x, y, radian
    
    # ---------------------------------------------------------------------------------------
    def buildings_boundary(self, position: np.ndarray, previous_position: np.ndarray, radian: float, speed: float):
        x, y = position
        xp, yp = previous_position

        for bx, by, w, h in self.buildings:
            if bx <= x <= bx+w and by <= y <= by+h:
                if xp < bx:                                                 # left
                    radian = np.pi-radian if radian > 0 else -np.pi-radian
                    return x-speed, y, radian
                if xp > bx+w:                                               # right
                    radian = np.pi-radian if radian > 0 else -np.pi-radian
                    return x+speed, y, radian
                if yp < by:                                                 # buttom
                    radian = -radian
                    return x, y-speed, radian
                if yp > by + h:                                             # top
                    radian = -radian
                    return x, y+speed, radian

        return x, y, radian

