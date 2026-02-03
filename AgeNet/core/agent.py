import numpy as np
from typing import List, Tuple

# -------- physics --------
from ..physics.mobility import MobilityModel
from ..physics.state import PhysicalState

# -------- communication --------
from ..communication.radio import RadioModel
from ..communication.channel import ChannelModel
from ..communication.neighbor_finder import NeighborFinder
from ..communication.request_policy import RadiusRequestPolicy

# -------- learning --------
from ..learning.brain import RLBrain
from ..learning.state import StateExtractor
from ..learning.hamiltonian import Hamiltonian
from ..learning.radius_controller import RadiusController



class Agent:
    def __init__(
        self, 
        agent_id: str, 
        alphas: Tuple,
        brain_parameters: List,
        communication_parameters: List[float],

        physical_state: PhysicalState, 
        mobility: MobilityModel | None = None
    ):
        
        self.id = agent_id
        
        # -------- physical state --------
        self.physical_state = physical_state
        self.mobility       = mobility

        # -------- communication --------
        absorption, base_delay, max_delay = communication_parameters
        self.radio           = RadioModel(self, base_delay, max_delay)
        self.channel         = ChannelModel(self, absorption)
        self.neighbor_finder = NeighborFinder(self)

        # -------- learning --------
        self.hamiltonian_model  = Hamiltonian(alphas)
        self.radius_policy      = RadiusRequestPolicy(self, self.hamiltonian_model)
        self.radius_controller  = RadiusController(self)
        self.state              = StateExtractor(self, self.channel)
        self.brain              = RLBrain(*brain_parameters)

        # -------- bookkeeping --------
        self.position = self.physical_state.get_position()
        self.speed    = self.physical_state.get_speed()
        self.radian   = self.physical_state.get_direction()
        self.edge     = self.physical_state.get_edge()

        self.neighbors          = []
        self.incoming_neighbors = []
        self.delta_r = 0.
        
        self.r   = 1.0
        self.k   = 0
        self.rho = 0.0


    # ==================================================================================
    # High-level API
    # ==================================================================================

    def move(self):
        if self.mobility is None: 
            raise ValueError( "Mobility/Movement is not defined. (Maybe using the sumo environment)" )

        position, speed, radian = self.mobility.move(self.position)
        return position, speed, radian

    def update_physical_state(self, position, speed, radian, edge):
        self.position, self.speed, self.radian, self.edge = position, speed, radian, edge

        self.physical_state.set_position (position)
        self.physical_state.set_direction(radian)
        self.physical_state.set_speed(speed)
        self.physical_state.set_edge (edge)

    def update_neighbors(self, all_agents: List["Agent"]):
        self.neighbors    = self.neighbor_finder.neighbors(all_agents)
        self.incoming_neighbors = self.neighbor_finder.incoming_neighbors(all_agents)

    def decide_request(self):
        self.r = self.radius_policy.decide(self.incoming_neighbors)

    def flip_radius(self, delta_H: float):
        self.r = self.radius_controller.flip(self.delta_r, delta_H)

    def apply_action(self, action: float):
        self.r, self.delta_r = self.radius_controller.apply_action(action)

    def remember(self, state, action, reward, next_state):
        self.brain.remember(state, action, reward, next_state)

    def learn(self, steps_per_train: int = 10):
        self.brain.train_per(steps_per_train)

    def act(self, state: np.ndarray) -> int:
        return self.brain.act(state)

    def hamiltonian(self) -> float:
        return self.hamiltonian_model(self.k, self.r, self.neighbors)

    def observe(self, obstacles):
        self.k, self.r, self.rho = self.state.mystate(obstacles)
        return np.array([self.k, self.r, self.rho], dtype=np.float32)


    # ==================================================================================
    # Convenience properties
    # ==================================================================================

    @property
    def energy(self): return self.hamiltonian()     # Expose hamiltonian as 'energy'
    @property
    def model(self): return self.brain.model        # Export neural network model
    @property
    def replay_memory(self): return self.brain.replay_memory
    
    # ----------------------------------------------------------------------------------
    def __repr__(self):
        return f"Agent(id={self.id}, pos: {self.position} - (r={self.r:.2f}, k={self.k}, rho={self.rho:.2f}) )"


