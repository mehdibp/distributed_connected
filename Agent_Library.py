import random
import numpy as np
import networkx as nx
from celluloid   import Camera
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.transforms import Bbox

import tensorflow as tf


# -------------------------------------------------------------------------------------------
class Distributed_Agent():
    # ---------------------------------------------------------------------------------------
    def __init__(self, environment, Parameters, Functions, model_path):

        self.environment = environment

        # Enable or disable different functions (different features) ------------------------
        self.requesting, self.moving, self.training = Functions

        # Agent parameters ------------------------------------------------------------------
        [self.N, self.L, Alphas, self.learning_rate, self.discount_rate, self.batch_size, self.steps_per_train] = Parameters
        self.alpha_1, self.alpha_2, self.alpha_3, self.alpha_4, = Alphas

        # Initialize xᵢ, yᵢ (positions of agent in plan) ------------------------------------
        self.init_position()


        # Build model -----------------------------------------------------------------------
        tf.keras.backend.clear_session()
        custom_activation = {'custom_activation': tf.keras.layers.Activation(self.custom_activation)}
        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_activation, compile=False)
        
        self.r = 1.                   # Transition reange
        self.k = 0                    # Degree of a vertex (Number of conections)

        self.replay_memory = deque(maxlen=40)   # A bag for 40 recently viewed (state, action, reward, next_state)        
        self.radian = (np.random.randint(360) -180)/180*np.pi   # The angle towards which the agent moves more
        self.trainCounter = 1                                   # Just to set the train to run every few steps
        self.loss = tf.constant(0.)                             # Just for print and plot loss per step


    # ---------------------------------------------------------------------------------------    
    def init_position(self):
        self.x = np.random.rand()*self.L           # Initialize xᵢ (positions of agent in plan)
        self.y = np.random.rand()*self.L           # Initialize yᵢ (positions of agent in plan)

        j = 0
        while j < len(self.environment.buildings):
            build = self.environment.buildings[j]
            if (self.x >= build[0] and self.x <= (build[0] + build[2]) and 
                self.y >= build[1] and self.y <= (build[1] + build[3])):

                self.x = np.random.rand()*self.L
                self.y = np.random.rand()*self.L

                j=0
            else: j += 1

    # ---------------------------------------------------------------------------------------    
    def Prediction(self, OtherAgents):
        self.N = len(OtherAgents) + 1
        self.OtherAgents = OtherAgents

        state  = [s+1 for s in self.MyState()]
        action = np.argmax( self.model( np.array([state])).numpy()[0] )
        STEP   = self.step((action-0.5)*2)
        reward = STEP[1]
        next_state = [ns+1 for ns in STEP[0]]

        self.replay_memory.append((state, action, reward, next_state))

        if self.training and self.trainCounter%self.steps_per_train == 0: self.Train()
        self.trainCounter += 1

    # ---------------------------------------------------------------------------------------    
    def MyState(self):
        self.k = 0
        rho    = 0
        for i in range(self.N-1):
            distance = ( (self.x - self.OtherAgents[i].x)**2 + (self.y-self.OtherAgents[i].y)**2 )**0.5
            if distance <= self.r and distance <= self.OtherAgents[i].r:
                intersect = self.environment.do_intersect([(self.x, self.y), (self.OtherAgents[i].x, self.OtherAgents[i].y)])
                if not intersect or (distance <= 0.5*self.r and distance <= 0.5*self.OtherAgents[i].r):
                    self.k += 1 
            
            
            if distance <= 1.6:                 ### should debug it
                intersect = self.environment.do_intersect([(self.x, self.y), (self.OtherAgents[i].x, self.OtherAgents[i].y)])
                if not intersect or distance <= 0.8: 
                    rho += 1
        if rho >  8: rho = 8                    ### should debug in training
        if rho == 0: rho = self.N / self.L**2

        return([ self.k, self.r, rho ])

    # ---------------------------------------------------------------------------------------    
    def step(self, action):

        if self.k == 0 and action == -1: action = 0.1
        Hamilton_t0 = self.Hamiltonian()

        delta_r = action* np.sqrt(self.L**2/self.N)/4 *np.random.random()
        self.r += delta_r
        if self.r < 0:             self.r = 0
        if self.r > 2**0.5*self.L: self.r = 2**0.5*self.L
        
        _, _, rho = self.MyState()


        Hamilton_t1 = self.Hamiltonian()
        _reward = Hamilton_t1 - Hamilton_t0

        self.Flip_redius(delta_r, _reward)
        if self.requesting: self.SendRequest()
        if self.moving    : self.Movement()

        return ([ self.k, self.r, rho ], -_reward) 

    # ---------------------------------------------------------------------------------------    
    def Hamiltonian(self):
        
        fourth = 0
        for i in range(self.N-1):
            distance = ( (self.x - self.OtherAgents[i].x)**2 + (self.y-self.OtherAgents[i].y)**2 )**0.5
            if distance <= self.r and distance <= self.OtherAgents[i].r:
                fourth += 1/distance
            
        H = self.alpha_1*self.k**2 + self.alpha_2*self.k**3 + self.alpha_3*self.r**2 + self.alpha_4*fourth
        return H
    
    # ---------------------------------------------------------------------------------------    
    def Flip_redius(self, delta_r, delta_H):
        if np.exp(-delta_H/4) < np.random.random():     # delta_H < 0 --> f(x) > 1;
            self.r -= delta_r*np.random.random()

        # for i in range(self.N):
        #     delta_k = (self.k[i] - k_previous[i]) / (k_previous[i]+1e-12)
        #     if delta_k > 0 and np.random.random() < delta_k + 1/2*(0.8-delta_k):
        #         self.r[i] -= delta_r[i]

    # ---------------------------------------------------------------------------------------
    def SendRequest(self):

        requested = []
        for i in range(self.N-1):
            distance = ( (self.x - self.OtherAgents[i].x)**2 + (self.y-self.OtherAgents[i].y)**2 )**0.5
            if distance > self.r and distance <= self.OtherAgents[i].r:
                delta_H = self.alpha_1*((self.k+1)**2 - self.k**2) + self.alpha_2*((self.k+1)**3 - self.k**3) + self.alpha_3*(distance**2 - self.r**2) - self.alpha_4*(1/distance)
                requested.append([ distance, delta_H ])

                ArgMinimum = np.argmin(np.array(requested)[:,1])
                if np.exp(-requested[ArgMinimum][1] * 4) > np.random.random():
                # if requested[ArgMinimum][1] < 0:
                    self.r = requested[ArgMinimum][0]

    # --------------------------------------------------------------------------------------- 
    def Movement(self):
        Rho = 0.02*(self.L**2/self.N)

        x_p, y_p = self.x, self.y                   # Save previous agent location

        # Assume a non-Markovian behavior for each particle and move according to it
        if random.random() < 0.6: self.x += Rho/2*np.cos(self.radian)
        else: self.x += Rho*(random.random() -0.5)

        if random.random() < 0.6: self.y += Rho/2*np.sin(self.radian)
        else: self.y += Rho*(random.random() -0.5) 

        # Reflective boundary condition -------------------------------------------------
        self.x, self.y, self.radian = self.environment.BoundaryCondition((self.x, self.y), self.radian, Rho)
        self.x, self.y, self.radian = self.environment.BuildingsBoundary((self.x, self.y), (x_p, y_p), self.radian, Rho)

    # ---------------------------------------------------------------------------------------
    def Train(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        loss_fn   = tf.keras.losses.mse
        
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
        target_Q_values = rewards + self.discount_rate*max_next_Q_values     # Equation 18-5. Q-Learning algorithm
        target_Q_values = target_Q_values.reshape(-1, 1)                # reshape to (32,1) beacuse of Q_values.shape
        
        
        mask = tf.one_hot(actions, 2)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values  = tf.reduce_sum(all_Q_values*mask, axis=1, keepdims=True)
            self.loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(self.loss, self.model.trainable_variables)

        nan = 0
        for g in grads:
            if np.isnan(g.numpy().ravel()).any(): nan = 1
        if nan != 1: optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    # ---------------------------------------------------------------------------------------
    def custom_activation(self, x):
        xx = (tf.keras.backend.softplus(x))**0.5
        return -tf.keras.backend.elu(-xx + 100) + 100
        



# -------------------------------------------------------------------------------------------
class Environment():
    # ---------------------------------------------------------------------------------------
    def __init__(self, ENV_Parameters):
        self.L, buildings_type, num_buildings, num_streets = ENV_Parameters

        self.buildings = []
        self.build_bounds = []
        # self.border = np.array([[-L, 0, L, L], [0, -L, L, L], [ L, 0, L, L], [ 0, L, L, L]])

        if buildings_type == "random" : self.RandomBuilding (num_buildings)
        if buildings_type == "regular": self.RegularBuilding(num_streets)

    # ---------------------------------------------------------------------------------------
    def BoundaryCondition(self, agent_pos, radian, Rho):
        x, y = agent_pos
        if (x > self.L) or (x < 0): 
            radian = np.pi-radian if radian > 0 else -np.pi-radian
            x = Rho if x < 0 else self.L-Rho
        if (y > self.L) or (y < 0): 
            radian = -radian
            y = Rho if y < 0 else self.L-Rho
        
        return x, y, radian
   
    # ---------------------------------------------------------------------------------------
    def RandomBuilding(self, num_buildings):
        while len(self.buildings) < num_buildings:
            x, y          = np.random.randint(self.L   , size=2)
            # width, height = np.random.randint(self.L/15, size=2) + 1
            width, height = np.random.randint(2     , size=2) + self.L/15

            No_intersection = True
            for b in self.buildings:
                if not (x+width <= b[0] or x >= b[0]+b[2] or y+height <= b[1] or y >= b[1]+b[3]): No_intersection = False
            if No_intersection: 
                self.buildings   .append(np.array([x, y, width, height]))
                self.build_bounds.append(Bbox.from_bounds(x, y, width, height))
    
    # ---------------------------------------------------------------------------------------
    def RegularBuilding(self, num_streets):
        Ns_Vertical   = int(np.ceil (num_streets/2))
        Ns_Horizontal = int(np.floor(num_streets/2))

        Ls_Vertical   = self.L / (4*Ns_Vertical-1)
        Ls_Horizontal = self.L / (3*Ns_Horizontal)

        for v in range(Ns_Vertical+1):
            for h in range(Ns_Horizontal+1):

                x = (4*v-2)*Ls_Vertical
                y = (3*h-1)*Ls_Horizontal
                width  = 3*Ls_Vertical
                height = 2*Ls_Horizontal

                self.buildings.append(np.array([x, y, width, height]))
                self.build_bounds.append(Bbox.from_bounds(x, y, width, height))

    # ---------------------------------------------------------------------------------------
    def do_intersect(self, agents):
        agents = Path(agents)
        for build in self.build_bounds:
            if agents.intersects_bbox(build): return True
        return False

    # ---------------------------------------------------------------------------------------
    def BuildingsBoundary(self, agent_pos, agent_previous_pos, radian, Rho):
        x, y     = agent_pos
        x_p, y_p = agent_previous_pos

        for build in self.buildings:
            if (x >= build[0]+Rho and x <= (build[0]+build[2]+Rho) and 
                y >= build[1]+Rho and y <= (build[1]+build[3]+Rho)):

                if x_p <  build[0]+Rho:                                    # left
                    radian = np.pi-radian if radian > 0 else -np.pi-radian
                    return x-Rho, y, radian

                if x_p > (build[0]+build[2]+Rho):                          # right
                    radian = np.pi-radian if radian > 0 else -np.pi-radian
                    return x+Rho, y, radian
                
                if y_p <  build[1]+Rho:                                    # buttom
                    radian = -radian
                    return x, y-Rho, radian
                
                if y_p > (build[1]+build[3]+Rho):                          # top
                    radian = -radian
                    return x, y+Rho, radian
                
        return x, y, radian




# -------------------------------------------------------------------------------------------
class Plot_Environment():
    # ---------------------------------------------------------------------------------------
    def __init__(self, environment, Agents):
        self.environment = environment

        self.L = self.environment.L
        self.N = len(Agents)
        self.A = np.zeros((self.N, self.N))
        self.Agents = Agents

        # Parameter for calculating tau in Calculate_Result()
        self.P_ij = np.ones((self.N, self.N))*1e-15
        self.Q_ij = np.ones((self.N, self.N))*1e-15
        self.previous_A = np.zeros((self.N, self.N))

        fig, self.ax = plt.subplots(figsize=(14,14))
        self.camera = Camera(fig)


    # ---------------------------------------------------------------------------------------
    def Environmental_Changes(self, Agents):
        self.N = len(Agents)
        self.k = np.zeros(self.N)
        self.A = np.zeros((self.N,self.N))
        if len(self.A) != len(self.P_ij): self.P_ij, self.Q_ij = np.ones((self.N, self.N)), np.ones((self.N, self.N))
        self.Agents = Agents
        
        for i in range(self.N):
            for j in range(i+1, self.N):
                distance = ( (Agents[i].x-Agents[j].x)**2 + (Agents[i].y-Agents[j].y)**2 )**0.5
                if distance <= Agents[i].r and distance <= Agents[j].r:
                    intersect = self.environment.do_intersect([(Agents[i].x, Agents[i].y), (Agents[j].x, Agents[j].y)])
                    if not intersect or (distance <= 0.5*Agents[i].r and distance <= 0.5*Agents[j].r):
                        self.A[i][j] = self.A[j][i] = 1
                        self.k[i]   += 1
                        self.k[j]   += 1

        if len(self.A) != len(self.previous_A): self.previous_A = self.A

    # ---------------------------------------------------------------------------------------
    def Env_Plot(self, step):
        if self.environment.buildings:
            for build in self.environment.buildings:
                x, y, w, h = build
                self.ax.add_patch( plt.Rectangle((x, y), width=w, height=h, fill=True, color='#146464cc', ec="black") ) 

        plt.text(0.9*self.L, 1.03*self.L, f'Episode {step}', fontname='Comic Sans MS', fontsize=12)
        plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.xlim([-0.02*self.L, 1.02*self.L])
        plt.ylim([-0.02*self.L, 1.02*self.L])
        plt.grid(alpha = 0.3)

    # ---------------------------------------------------------------------------------------
    def Animation(self, step):
        G = nx.from_numpy_array(self.A)
        options = { 'node_size': 60, 'width': 1.5, 'node_color': [(0.5,0.0,0.8,1)]*G.number_of_nodes() }

        for i in range(self.N):
            G.add_node(i, pos=(self.Agents[i].x, self.Agents[i].y))
            # if not (i in (list((sorted(nx.connected_components(G), key=len, reverse=True))[0]))): 
            #     plt.gca().add_artist(plt.Circle((self.Agents[i].x, self.Agents[i].y), radius=self.Agents[i].r, color='#66338033'))

        pos = nx.get_node_attributes(G,'pos')
        nx.draw_networkx(G, pos, with_labels=True, **options)

        self.Env_Plot(step)
        self.camera.snap()
    
    # ---------------------------------------------------------------------------------------
    def Calculate_Result(self, Agents, s):
        self.Environmental_Changes(Agents)

        # Hamiltonian of the whole system -----------------------------------------
        hamilton = 0
        for i in range(self.N): hamilton += self.Agents[i].Hamiltonian()
        # -------------------------------------------------------------------------
        edge = self.k.sum()/2
        # -------------------------------------------------------------------------
        energy = 0
        for i in range(self.N): energy += 0.2*self.Agents[i].r**2
        # -------------------------------------------------------------------------
        average_r = 0
        for i in range(self.N): average_r += self.Agents[i].r / self.N
        
        # Giant component of network (%) ------------------------------------------
        G = nx.from_numpy_array(self.A)
        giant = len((sorted(nx.connected_components(G), key=len, reverse=True))[0])/self.N * 100

        # Calculate Tau = 1/sM sigma(P_ij/Q_ij) -----------------------------------
        for i in range(self.N):
            for j in range(i, self.N):
                if self.previous_A[i][j] != self.A[i][j]:
                    self.Q_ij[i][j] += 1
                    self.Q_ij[j][i] += 1

        self.P_ij += self.A
        self.previous_A = self.A
        tau = 1/((s+1e-15)*(self.N*(self.N-1))) * (self.P_ij/self.Q_ij).sum()
        
        return hamilton, edge, energy, average_r, giant, tau



# -------------------------------------------------------------------------------------------
def test():
    print("TEST")


# -------------------------------------------------------------------------------------------

