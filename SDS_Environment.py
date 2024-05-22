import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
import networkx as nx
from collections import deque

import tensorflow as tf


# -------------------------------------------------------------------------------------------
class Distributed_System():
    # ---------------------------------------------------------------------------------------
    def __init__(self, N, L):
        np.random.seed(44)
        self.N = N
        self.L = L
        
        # Variables: x , y (positions of agents in plan)
        self.x = np.random.rand(N)*L           # Initialize xᵢ
        self.y = np.random.rand(N)*L           # Initialize yᵢ
        
        self.r = np.ones(N)                    # Wave sending radius
        self.A = np.zeros((N,N))               # Adjacency Matrix
        self.k = np.zeros(N)                   # Degree of a vertex
        for i in range(N):
            for j in range(i+1,N):
                distance = ( (self.x[i]-self.x[j])**2 + (self.y[i]-self.y[j])**2 )**0.5
                if distance <= self.r[i] and distance <= self.r[j]:
                    self.A[i][j] = self.A[j][i] = 1
                    self.k[i]   += self.A[i][j]
                    self.k[j]   += self.A[i][j]
    
    # ---------------------------------------------------------------------------------------    
    def step(self, action, episode):

        if (episode+1)%100 == 0:
            action = self.Turning_off_or_on(action)

        delta_r = action*(0.16*self.L**2/self.N)*np.random.random(size=self.N)
        _reward = self.calc_rewards(delta_r)
        self.r += delta_r
        self.check(delta_r, _reward)
        
        self.A = np.zeros((self.N,self.N))
        self.k = np.zeros(self.N)
        rho    = np.zeros(self.N)
        
        for i in range(self.N):
            
            if self.r[i] < 0:     self.r[i] = 0
            if self.r[i] > 2**0.5*self.L: self.r[i] = 2**0.5*self.L

            for j in range(i+1, self.N):
                distance = ( (self.x[i]-self.x[j])**2 + (self.y[i]-self.y[j])**2 )**0.5
                if distance <= self.r[i] and distance <= self.r[j]:
                    self.A[i][j] = self.A[j][i] = 1
                    self.k[i]   += self.A[i][j]
                    self.k[j]   += self.A[i][j]
                
                if distance <= 1.6: 
                    rho[i] += 1
                    rho[j] += 1
                if rho[i] > 8: rho[i] = 8       ### should debug in training

        return ([ self.k+1, self.r+1, rho+1 ], -_reward)

    # ---------------------------------------------------------------------------------------    
    def check(self, delta_r, delta_H):
        for i in range(self.N):
            if np.exp(-delta_H[i]/4) < np.random.random():     # delta_H < 0 --> f(x) > 1;
                self.r[i] -= delta_r[i]*np.random.random()

        # for i in range(self.N):
        #     delta_k = (self.k[i] - k_previous[i]) / (k_previous[i]+1e-12)
        #     if delta_k > 0 and np.random.random() < delta_k + 1/2*(0.8-delta_k):
        #         self.r[i] -= delta_r[i]

    # ---------------------------------------------------------------------------------------    
    def calc_rewards(self, delta_r):
        
        Hamilton_t0 = np.zeros(self.N)
        Hamilton_t1 = np.zeros(self.N)
        for i in range(self.N): Hamilton_t0[i] = self.Hamiltonian(i)

        self.A = np.zeros((self.N,self.N))
        self.k = np.zeros(self.N)
        radius = np.copy(self.r)

        for i in range(self.N):

            self.r = np.copy(radius)
            self.r[i] += delta_r[i]
            if self.r[i] < 0:     self.r[i] = 0
            if self.r[i] > 2**0.5*self.L: self.r[i] = 2**0.5*self.L

            for j in range(self.N):
                if i != j:
                    distance = ( (self.x[i]-self.x[j])**2 + (self.y[i]-self.y[j])**2 )**0.5
                    if distance <= self.r[i] and distance <= self.r[j]:
                        self.A[i][j] = 1
                        self.k[i]   += self.A[i][j]

            Hamilton_t1[i] = self.Hamiltonian(i)

        self.r = np.copy(radius)
        reward = Hamilton_t1 - Hamilton_t0
        # self.check(delta_r, reward, k_previous)
        return reward

    # ---------------------------------------------------------------------------------------    
    def Hamiltonian(self, i):
        alfa_1 = -0.5
        alfa_2 = +0.1
        alfa_3 = +0.2
        alfa_4 = -0.5
        
        fourth = np.zeros(self.N)
        for j in range(self.N):
            if i != j:
                fourth[i] += (self.A[i][j] / (( (self.x[i]-self.x[j])**2 + (self.y[i]-self.y[j])**2 )**0.5 ))
            
        H = alfa_1*self.k[i]**2 + alfa_2*self.k[i]**3 + alfa_3*self.r[i]**2 + alfa_4*fourth[i]
        return H
    
    # --------------------------------------------------------------------------------------- 
    def Turning_off_or_on(self, action):
        
        random = np.random.random()
        for _ in range(10):
            if random < 0.5:
                choice = np.random.choice(self.N)
                self.x = np.delete(self.x, choice)
                self.y = np.delete(self.y, choice)
                self.r = np.delete(self.r, choice)
                self.k = np.delete(self.k, choice)
                self.A = np.delete(np.delete(self.A, choice, 0), choice, 1)
                action = np.delete(action, choice)
                self.N -= 1
            else:
                self.x = np.append(self.x, np.random.random()*self.L) 
                self.y = np.append(self.y, np.random.random()*self.L) 
                self.r = np.append(self.r, 1)
                self.k = np.append(self.k, 0)
                self.A = np.append(np.append(self.A, np.zeros((1,self.N)), 0), np.zeros((self.N+1,1)), 1)
                action = np.append(action, 0)
                self.N += 1
        
        return action
    
    # --------------------------------------------------------------------------------------- 
    def move(self):
        self.x = self.x + (0.16*self.L**2/self.N)*(np.random.random(size=self.N) -0.5)
        self.y = self.y + (0.16*self.L**2/self.N)*(np.random.random(size=self.N) -0.5) 

    # ---------------------------------------------------------------------------------------
    def Plot(self, camera, episode):
        options = { 'node_size': 60, 'width': 0.3 }
        
        G = nx.from_numpy_array(self.A)
        for i in range(self.N):
            G.add_node(i, pos=(self.x[i], self.y[i]))
        pos = nx.get_node_attributes(G,'pos')
        
        nx.draw_networkx(G, pos, with_labels=True, **options)
        giant = len((sorted(nx.connected_components(G), key=len, reverse=True))[0])/self.N * 100
        
        plt.text(self.L-0.15*self.L, self.L+0.3, f'Episode {episode}', fontname='Comic Sans MS', fontsize=12)
        plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.xlim(-0.1, self.L+0.1); plt.ylim(-0.1, self.L+0.1)
        plt.grid(alpha = 0.3)
        camera.snap()
        return giant
    

# -------------------------------------------------------------------------------------------
def training_step(i, model, n_outputs, replay_memory, batch_size, discount_rate):
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-2)
    loss_fn   = tf.keras.losses.mean_squared_error
    
    indices = np.random.randint(len(replay_memory[i]), size=batch_size)   # 32ta random number between[0 - len(replay)]
    batch   = [replay_memory[i][index] for index in indices]              # number in replay_memory[indices]
    
    states, actions, rewards, next_states = [                             # from replay_memory read these and save in...
        np.array([experience[field_index] for experience in batch])
        for field_index in range(4)]
    
    
    for j in range(len(rewards)):
        if rewards[j] < 0.05 and rewards[j] > -0.1: rewards[j] = 0.
        else: rewards[j] = round(rewards[j], ndigits=3)
            
    next_Q_values   = model[0].predict(next_states, verbose=0)      # 32 predict of 2 actions
    max_next_Q_values = np.max(next_Q_values, axis=1)               # choose higher probiblity of each actions (of each 32)
    target_Q_values = rewards + discount_rate*max_next_Q_values     # Equation 18-5. Q-Learning algorithm
    target_Q_values = target_Q_values.reshape(-1, 1)                # reshape to (32,1) beacuse of Q_values.shape
    
    
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model[0](states)
        Q_values = tf.reduce_sum(all_Q_values*mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model[0].trainable_variables)

    nan = 0
    for g in grads:
        if np.isnan(g.numpy().ravel().any()): nan = 1
    if nan != 1: optimizer.apply_gradients(zip(grads, model[0].trainable_variables))
        
    return all_Q_values, next_Q_values, Q_values, target_Q_values, loss, grads, model[0].trainable_variables       ### 


# -------------------------------------------------------------------------------------------
def test():
    print("TEST")


# -------------------------------------------------------------------------------------------

