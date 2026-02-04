# ============================================================
# SUMO physical environment adapter (TraCI)
# ============================================================

import numpy as np
from typing import Tuple, List, Dict

import traci
import sumolib

from .base import Environment, Entity


class SumoEnvironment(Environment):
    """
    Adapter layer between SUMO (TraCI) and multi-agent RL system.
    This class is the SINGLE SOURCE OF TRUTH for physical states.
    """

    # ---------------------------------------------------------------------------------------
    def __init__(self, sumo_cfg: str, sumo_binary: str="sumo", step_length: float=0.1, use_gui: bool=False, seed: int|None=None):
        """
        Args:
            sumo_cfg     : path to *.sumocfg file
            sumo_binary  : "sumo" or "sumo-gui"
            step_length  : simulation step length (seconds)
            use_gui      : whether to use SUMO-GUI
            seed         : SUMO random seed
        """

        super().__init__()

        self.sumo_cfg = sumo_cfg
        self.sumo_binary = sumo_binary if not use_gui else "sumo-gui"
        self.step_length = step_length
        self.seed = seed

        self._buildings    = []
        self._build_bounds = []
        self._started = False

        self._net_bounds: Tuple[float, float, float, float] | None = None   # network bounds


    # SUMO lifecycle ------------------------------------------------------------------------
    def start(self):
        """Start SUMO and initialize environment."""
        if self._started: return

        cmd = [ self.sumo_binary, "-c", self.sumo_cfg, "--step-length", str(self.step_length) ]
        if self.seed is not None: cmd += ["--seed", str(self.seed)]

        traci.start(cmd)
        self._started = True

        self._load_network_bounds()
        self.step()  

    def close(self):
        """Close SUMO safely."""
        if self._started:
            traci.close()
            self._started = False

    def reset(self):
        """Restart SUMO simulation."""
        self.close()
        self.start()

    def step(self):
        """Advance simulation by one step."""
        traci.simulationStep()
        self._cache_step_data()


    # geometry ------------------------------------------------------------------------------
    def get_bounds (self) -> Tuple[float, float, float, float]:
        """Return (xmin, ymin, xmax, ymax) of network."""
        return self._net_bounds

    def get_density(self) -> float:
        """Vehicle density per unit area."""
        if self._net_bounds is None: return 0.0

        xmin, ymin, xmax, ymax = self._net_bounds
        area = max((xmax - xmin) * (ymax - ymin), 1e-6)
        return len(self._entities) / area

    
    # internal ------------------------------------------------------------------------------
    def _cache_step_data(self):
        """Cache vehicle positions and ID."""
        self._time  = traci.simulation.getTime()
        vehicle_ids = traci.vehicle.getIDList()

        self._entities.clear()
        for vid in vehicle_ids:
            pos   = traci.vehicle.getPosition(vid)
            speed = traci.vehicle.getSpeed   (vid)
            edge  = traci.vehicle.getRoadID  (vid)
            self._entities[vid] = Entity(np.array(pos, dtype=np.float32), speed, None, edge)

    def _load_network_bounds(self):
        """Extract bounding box of SUMO road network."""
        cfg = next(sumolib.xml.parse(self.sumo_cfg, "configuration"))
        net_file = cfg.input[0].getChild('net-file')[0].value

        if net_file is None: raise RuntimeError("Could not find net-file in sumocfg.")

        path = self.sumo_cfg.removesuffix('simulation.sumocfg')
        net  = sumolib.net.readNet(f"{path}{net_file}")
        xmin, ymin, xmax, ymax = net.getBoundary()
        self._net_bounds = (xmin, ymin, xmax, ymax)
