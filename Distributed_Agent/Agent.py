import random
import numpy as np
import tensorflow as tf
from collections import deque

from . import Environment


# -------------------------------------------------------------------------------------------
class Distributed_Agent():
    # ---------------------------------------------------------------------------------------
    def __init__(self, environment: Environment.Environment, Parameters: list, Functions: list, model_path: str):

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
    def Prediction(self, OtherAgents: list):
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
    def step(self, action: int):

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
    def Flip_redius(self, delta_r: float, delta_H: float):
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
        Rho = 0.01*(self.L**2/self.N)               # Have to change to Rho = 0.005*self.L

        x_p, y_p = self.x, self.y                   # Save previous agent location

        # Assume a non-Markovian behavior for each particle and move according to it
        if random.random() < 0.6: self.x += Rho*np.cos(self.radian)
        else: self.x += 2*Rho*(random.random() -0.5)

        if random.random() < 0.6: self.y += Rho*np.sin(self.radian)
        else: self.y += 2*Rho*(random.random() -0.5) 

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
        

