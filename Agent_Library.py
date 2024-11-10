import numpy as np
import matplotlib.pyplot as plt
import random
from celluloid import Camera
import networkx as nx
from collections import deque

import tensorflow as tf


# -------------------------------------------------------------------------------------------
class Distributed_Agent():
    # ---------------------------------------------------------------------------------------
    def __init__(self, N, L, Alphas=[-0.5,0.1,0.2,-0.5], requesting=False, moving=False, training=False):
        self.L = L
        self.N = N

        # Variables: x , y (positions of agent in plan)
        self.x = np.random.rand()*L           # Initialize xᵢ
        self.y = np.random.rand()*L           # Initialize yᵢ

        self.alpha_1, self.alpha_2, self.alpha_3, self.alpha_4, = Alphas

        # build model
        tf.keras.backend.clear_session()
        self.model = tf.keras.models.load_model('./All Results/Different Model Training/models with alpha4 = -1000/L=45/model_H_best_weight.keras',
                        custom_objects={'custom_activation': tf.keras.layers.Activation(self.custom_activation)},   
                        compile=False)
        
        self.r = 1.                   # Transition reange
        self.k = 0                    # Degree of a vertex (Number of conections)

        self.replay_memory = deque(maxlen=40)   # A bag for 40 recently viewed (state, action, reward, next_state)

        self.training   = training
        self.requesting = requesting
        self.moving     = moving
        self.radian = (np.random.randint(360) -180)/180*np.pi    # The angle towards which the agent moves more
        self.trainCounter = 1

        self.loss = tf.constant(0.)   # Just for print and plot loss per step


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

        if self.training == True and self.trainCounter%10 == 0: self.Train(batch_size=20, discount_rate=0.98, learning_rate=1e-5)
        self.trainCounter += 1

    # ---------------------------------------------------------------------------------------    
    def MyState(self):
        self.k = 0
        rho    = 0
        for i in range(self.N-1):
            distance = ( (self.x - self.OtherAgents[i].x)**2 + (self.y-self.OtherAgents[i].y)**2 )**0.5
            if distance <= self.r and distance <= self.OtherAgents[i].r:
                self.k += 1 
            
            if distance <= 1.6: rho += 1
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
        if self.requesting == True: self.SendRequest()
        if self.moving     == True: self.Movement()

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

        # Assume a non-Markovian behavior for each particle and move according to it
        if random.random() < 0.6: self.x += Rho/2*np.cos(self.radian)
        else: self.x += Rho*(random.random() -0.5)

        if random.random() < 0.6: self.y += Rho/2*np.sin(self.radian)
        else: self.y += Rho*(random.random() -0.5) 

        # Reflective boundary condition
        if (self.x > self.L) or (self.x < 0): 
            self.radian = np.pi-self.radian if self.radian > 0 else -np.pi-self.radian
            self.x = 0+Rho if abs(self.x) < abs(self.x-self.L) else self.L-Rho
        if (self.y > self.L) or (self.y < 0): 
            self.radian = -self.radian
            self.y = 0+Rho if abs(self.y) < abs(self.y-self.L) else self.L-Rho

    # ---------------------------------------------------------------------------------------
    def Train(self, batch_size=32, discount_rate=0.98, learning_rate=1e-3):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        loss_fn   = tf.keras.losses.mse
        
        indices = np.random.randint(len(self.replay_memory), size=batch_size)   # 32 random number between[0 - len(replay)]
        batch   = [self.replay_memory[index] for index in indices]              # number in replay_memory[indices]
        
        states, actions, rewards, next_states = [                               # from replay_memory read these and save in...
            np.array([experience[field_index] for experience in batch])
            for field_index in range(4)]
        
        
        for j in range(len(rewards)):
            if rewards[j] < 0.05 and rewards[j] > -0.1: rewards[j] = 0.
            else: rewards[j] = round(rewards[j], ndigits=3)
                
        next_Q_values   = self.model(next_states)                       # 32 predict of 2 actions
        max_next_Q_values = np.max(next_Q_values, axis=1)               # choose higher probiblity of each actions (of each 32)
        target_Q_values = rewards + discount_rate*max_next_Q_values     # Equation 18-5. Q-Learning algorithm
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
class Plot_Environment():
    # ---------------------------------------------------------------------------------------
    def __init__(self, L, Agents):
        self.L = L
        self.N = len(Agents)
        self.A = np.zeros((self.N, self.N))
        self.Agents = Agents

        # Parameter for calculating tau in Calculate_Result()
        self.P_ij = np.ones((self.N, self.N))*1e-15
        self.Q_ij = np.ones((self.N, self.N))*1e-15
        self.previous_A = np.zeros((self.N, self.N))


    # ---------------------------------------------------------------------------------------
    def Environmental_Changes(self, Agents):
        self.N = len(Agents)
        self.k = np.zeros(self.N)
        self.A = np.zeros((self.N,self.N))
        self.Agents = Agents
        
        for i in range(self.N):
            for j in range(i+1, self.N):
                distance = ( (Agents[i].x-Agents[j].x)**2 + (Agents[i].y-Agents[j].y)**2 )**0.5
                if distance <= Agents[i].r and distance <= Agents[j].r:
                    self.A[i][j] = self.A[j][i] = 1
                    self.k[i]   += 1
                    self.k[j]   += 1
    
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

    # ---------------------------------------------------------------------------------------
    def Animation(self, camera, episode):
        options = { 'node_size': 60, 'width': 1.5 }

        G = nx.from_numpy_array(self.A)
        for i in range(self.N):
            G.add_node(i, pos=(self.Agents[i].x, self.Agents[i].y))
            if not (i in (list((sorted(nx.connected_components(G), key=len, reverse=True))[0]))): 
                plt.gca().add_artist(plt.Circle((self.Agents[i].x, self.Agents[i].y), radius=self.Agents[i].r, color=(0.4,0.2,0.5,0.2)))
        pos = nx.get_node_attributes(G,'pos')
        
        nx.draw_networkx(G, pos, with_labels=True, **options)
        
        plt.text(self.L-0.15*self.L, self.L+0.3, f'Episode {episode}', fontname='Comic Sans MS', fontsize=12)
        plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.xlim(-0.1, self.L+0.1); plt.ylim(-0.1, self.L+0.1)
        plt.grid(alpha = 0.3)
        camera.snap()
    
    # ---------------------------------------------------------------------------------------
    def Static_Plot(self, episode, Hamilton, Edges, Energy, R_avg, Giant, Tau):
        plt.figure(figsize=(16,12))

        plt.subplot(2,2,1)
        plt.title("Hamilton")
        plt.plot(Hamilton)
        plt.text(0.7*episode, 1.1*max(Hamilton), "Min(H): %f" % (min(Hamilton)) )
        plt.text(0.7*episode, 1.2*max(Hamilton), "Arg(H): %f" % (np.argmin(Hamilton)) )
        plt.grid(alpha=0.3)

        plt.subplot(2,2,2)
        plt.title("Giant Component")
        plt.plot(Giant)
        plt.grid(alpha=0.3)

        plt.subplot(2,2,3)
        plt.title("Transition Range")
        plt.plot(R_avg)
        plt.grid(alpha=0.3) 

        plt.subplot(2,2,4)
        plt.title("Tau")
        plt.plot(Tau)
        plt.grid(alpha=0.3) 

        plt.show()



# -------------------------------------------------------------------------------------------
def test():
    print("TEST")


# -------------------------------------------------------------------------------------------

