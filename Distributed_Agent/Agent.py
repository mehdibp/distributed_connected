import random
import numpy as np
import tensorflow as tf
from collections import deque

from . import Environment


# -------------------------------------------------------------------------------------------
class Agent_in_Env():
    # ---------------------------------------------------------------------------------------
    def __init__(self, environment: Environment.Environment, L: float):
        self.environment = environment
        self.L = L

        self.init_position()                            # Initialize xᵢ, yᵢ (positions of agent in plan)
        
        self.radian = np.random.uniform(-np.pi, np.pi)  # The angle towards which the agent moves more


    # ---------------------------------------------------------------------------------------
    def init_position(self):
        # Initialize agent's position avoiding buildings.
        while True:
            self.x = np.random.rand()*self.L           # Initialize xᵢ (positions of agent in plan)
            self.y = np.random.rand()*self.L           # Initialize yᵢ (positions of agent in plan)
            if not any( (b[0] <= self.x <= b[0]+b[2] and b[1] <= self.y <= b[1]+b[3]) for b in self.environment.buildings ):
                break

    # ---------------------------------------------------------------------------------------
    def get_position(self):
        return np.array([self.x, self.y])
    
    # --------------------------------------------------------------------------------------- 
    def movement(self):
        # Simple non-Markovian random motion with reflective boundary.
        stride = 0.2025 # 0.005*self.L                       # or stride = 0.01*(self.L**2/self.N)  ###
        x_prev, y_prev = self.x, self.y             # Save previous agent location

        # Assume a non-Markovian behavior for each particle and move according to it
        if random.random() < 0.6: self.x +=   stride*np.cos(self.radian)
        else:                     self.x += 2*stride*(random.random() -0.5)

        if random.random() < 0.6: self.y +=   stride*np.sin(self.radian)
        else:                     self.y += 2*stride*(random.random() -0.5)

        # Reflective boundary condition -------------------------------------------------
        self.x, self.y, self.radian = self.environment.BoundaryCondition((self.x, self.y), self.radian, stride)
        self.x, self.y, self.radian = self.environment.BuildingsBoundary((self.x, self.y), (x_prev, y_prev), self.radian, stride)


# -------------------------------------------------------------------------------------------
class Agent_Interaction():
    # ---------------------------------------------------------------------------------------    
    def __init__(self, agent):
        self.agent = agent
        self.N = 1

        self.neighbors = []  # [(other_agent, distance), ...]


    # ---------------------------------------------------------------------------------------   
    def update_neighbors(self, all_agents: list):
        # Compute list of agents that are within each other's connection radius.
        self.neighbors.clear()
        self.N = len(all_agents)

        for other in all_agents:
            if other is self.agent: continue

            distance = np.linalg.norm(self.agent.get_position() - other.get_position())
            
            # If this condition is true, this agent is in the other communication scope
            if distance <= other.r:
                self.neighbors.append((other, distance))

        return self.neighbors
    
    # Potential future extensions: ----------------------------------------------------------
    def exchange_information(self):
        """Send or receive messages (for OMNeT++ connection)."""
        pass

    def build_local_graph(self):
        """Return adjacency or connectivity data."""
        pass


# -------------------------------------------------------------------------------------------
class Distributed_Agent(Agent_in_Env):
    # ---------------------------------------------------------------------------------------
    def __init__(self, environment: Environment.Environment, Parameters: list, Functions: list, model_path: str):
        """
        Args:
            environment (Env) : An object of the "Environment" class that specifies the environment properties for each agent
            Parameters  (list): [N, L, Alphas, learning_rate, gamma, batch_size, steps_per_train]
            Functions   (list): [requesting, moving, training]
            model_path  (str) : Address and name of the initial saved model
        """

        # Enable or disable different functions (different features) ------------------------
        self.requesting, self.moving, self.training = Functions

        # Agent parameters ------------------------------------------------------------------
        [self.L, Alphas, self.lr, self.gamma, self.batch_size, self.steps_per_train] = Parameters
        self.alpha_1, self.alpha_2, self.alpha_3, self.alpha_4, = Alphas


        super().__init__(environment, self.L)
        self.r = 1.                   # Transition reange
        self.k = 0                    # Degree of a vertex (Number of conections)


        # Build model -----------------------------------------------------------------------
        tf.keras.backend.clear_session()
        custom_activation = {'custom_activation': tf.keras.layers.Activation(self.custom_activation)}
        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_activation, compile=False)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, clipnorm=1.0)
        self.loss_fn   = tf.keras.losses.mse
        

        self.replay_memory = deque(maxlen=40)   # A bag for 40 recently viewed (state, action, reward, next_state)
        self.trainCounter = 1                                   # Just to set the train to run every few steps
        self.loss = tf.constant(0.)                             # Just for print and plot loss per step

        # self.interaction = Agent_Interaction(self)


    # ---------------------------------------------------------------------------------------    
    def prediction(self, neighbors, N_density):
        self.neighbors = neighbors
        self.N_density = N_density

        state  = [s+1 for s in self.my_state()]
        action = np.argmax( self.model( np.array([state])).numpy()[0] )
        STEP   = self.step((action-0.5)*2)
        reward = STEP[1]
        next_state = [ns+1 for ns in STEP[0]]

        self.replay_memory.append((state, action, reward, next_state))

        if self.training and self.trainCounter%self.steps_per_train == 0: self.train()
        self.trainCounter += 1

    # ---------------------------------------------------------------------------------------    
    def my_state(self):
        self.k = 0
        rho    = 0

        for OtherAgents, distance in self.neighbors:
            intersect = self.environment.do_intersect([self.get_position(), OtherAgents.get_position()])
            if distance <= self.r and (not intersect or (distance <= 0.5*self.r and distance <= 0.5*OtherAgents.r)):
                self.k += 1

            if distance <= 1.6 and (not intersect or distance <= 0.8):  ### should debug it
                rho += 1

        # rho = np.clip(rho, self.N_density, 8)
        rho = min(rho, 8)                       ### should debug in training
        rho = max(rho, self.N_density)

        return np.array([self.k, self.r, rho], dtype=np.float32)

    # ---------------------------------------------------------------------------------------    
    def step(self, action: int):

        if self.k == 0 and action == -1: action = 0.1
        Hamilton_t0 = self.hamiltonian()

        delta_r = action* np.sqrt(1/self.N_density)/4 *np.random.random()
        self.r += delta_r
        if self.r < 0:             self.r = 0
        if self.r > 2**0.5*self.L: self.r = 2**0.5*self.L
        
        _, _, rho = self.my_state()


        Hamilton_t1 = self.hamiltonian()
        _reward = Hamilton_t1 - Hamilton_t0

        self.flip_radius(delta_r, _reward)
        if self.requesting: self.send_request()
        if self.moving    : self.movement()

        return ([ self.k, self.r, rho ], -_reward) 

    # ---------------------------------------------------------------------------------------    
    def hamiltonian(self):
        fourth = sum(1/distance for _, distance in self.neighbors if distance <= self.r)
        H = self.alpha_1*self.k**2 + self.alpha_2*self.k**3 + self.alpha_3*self.r**2 + self.alpha_4*fourth
        return H
    
    # ---------------------------------------------------------------------------------------    
    def flip_radius(self, delta_r: float, delta_H: float):
        if np.exp(-delta_H/4) < np.random.random():     # delta_H < 0 --> f(x) > 1;
            self.r -= delta_r*np.random.random()

        # for i in range(self.N):
        #     delta_k = (self.k[i] - k_previous[i]) / (k_previous[i]+1e-12)
        #     if delta_k > 0 and np.random.random() < delta_k + 1/2*(0.8-delta_k):
        #         self.r[i] -= delta_r[i]

    # ---------------------------------------------------------------------------------------
    def send_request(self):
        requested = []
        for _, distance in self.neighbors:
            if distance > self.r:
                delta_H = self.alpha_1*((self.k+1)**2 - self.k**2) + self.alpha_2*((self.k+1)**3 - self.k**3) + self.alpha_3*(distance**2 - self.r**2) - self.alpha_4*(1/distance)
                requested.append([ distance, delta_H ])

                ArgMinimum = np.argmin(np.array(requested)[:,1])
                if np.exp(-requested[ArgMinimum][1] * 4) > np.random.random():
                # if requested[ArgMinimum][1] < 0:
                    self.r = requested[ArgMinimum][0]

    # ---------------------------------------------------------------------------------------
    def train(self):
        indices = np.random.randint(len(self.replay_memory), size=self.batch_size)   # 32 random number between[0 - len(replay)]
        batch   = [self.replay_memory[index] for index in indices]              # number in replay_memory[indices]
        
        states, actions, rewards, next_states = [                               # from replay_memory read these and save in...
            np.array([experience[field_index] for experience in batch])
            for field_index in range(4)]
        
        
        for j in range(len(rewards)):
            if rewards[j] < 0.05 and rewards[j] > -0.1: rewards[j] = 0.
            else: rewards[j] = round(rewards[j], ndigits=3)
                
        next_Q_values   = self.model(next_states)                       # 32 predict of 2 actions
        max_next_Q_values = np.max(next_Q_values, axis=1)               # choose higher probiblity of each actions (of each 32)
        target_Q_values = rewards + self.gamma*max_next_Q_values     # Equation 18-5. Q-Learning algorithm
        target_Q_values = target_Q_values.reshape(-1, 1)                # reshape to (32,1) beacuse of Q_values.shape
        
        
        mask = tf.one_hot(actions, 2)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values  = tf.reduce_sum(all_Q_values*mask, axis=1, keepdims=True)
            self.loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(self.loss, self.model.trainable_variables)

        nan = 0
        for g in grads:
            if np.isnan(g.numpy().ravel()).any(): nan = 1
        if nan != 1: self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    # ---------------------------------------------------------------------------------------
    def custom_activation(self, x):
        return 100 - tf.nn.elu(-tf.sqrt(tf.nn.softplus(x)) + 100)



