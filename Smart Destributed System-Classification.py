import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from celluloid import Camera
import networkx as nx
import time

import tensorflow as tf
from tensorflow import keras
# ------------------------------------------------------------------

class Distributed_System():
    def __init__(self, N, L):
        np.random.seed(42)
        self.N = N
        self.L = L

        # Variables: x , y (positions of agents in plan)
        self.x = np.random.rand(N) * L  # Initialize xᵢ
        self.y = np.random.rand(N) * L  # Initialize yᵢ

        self.x[0] = 0; self.x[1] = 1; self.x[2] = 0; self.x[3] = 1.5
        self.y[0] = 0; self.y[1] = 0; self.y[2] = 1; self.y[3] = 1.5

        #         delta = L/(N**0.5)
        #         for i in range(N):
        #             self.x[i] = 0.5+ int(i/ round(N**0.5)) *delta + np.random.rand()*0.2*delta
        #             self.y[i] = 0.5+ int(i% round(N**0.5)) *delta + np.random.rand()*0.2*delta

        self.r = np.ones(N)  # Wave sending radius
        self.A = np.zeros((N, N))  # Adjacency Matrix
        self.k = np.zeros(N)  # Degree of a vertex
        for i in range(N):
            for j in range(i + 1, N):
                if ((self.x[i] - self.x[j]) ** 2 + (self.y[i] - self.y[j]) ** 2) ** 0.5 < self.r[i]:
                    self.A[i][j] = self.A[j][i] = 1
                    self.k[i] += self.A[i][j]
                    self.k[j] += self.A[i][j]

    def step(self, action):
        Hamilton_t0 = np.zeros(self.N)
        Hamilton_t1 = np.zeros(self.N)
        for i in range(self.N):
            Hamilton_t0[i], e = self.Hamiltonian(i)  ###

        self.r += action * (0.25 * self.L ** 2 / self.N) * np.random.rand()
        self.A = np.zeros((N, N))
        self.k = np.zeros(N)
        rho = np.zeros(N)

        for i in range(self.N):
            if self.r[i] < 0:     self.r[i] = 0
            if self.r[i] > 2 ** 0.5 * self.L: self.r[i] = 2 ** 0.5 * self.L

        for i in range(self.N):
            for j in range(i + 1, self.N):
                distance = ((self.x[i] - self.x[j]) ** 2 + (self.y[i] - self.y[j]) ** 2) ** 0.5
                if distance <= self.r[i] and distance <= self.r[j]:
                    self.A[i][j] = self.A[j][i] = 1
                    self.k[i] += self.A[i][j]
                    self.k[j] += self.A[i][j]

                if distance <= 1.2 * self.N / self.L ** 2:
                    rho[i] += 1
                    rho[j] += 1

        for i in range(self.N):
            Hamilton_t1[i], e = self.Hamiltonian(i)  ###
        reward = Hamilton_t0 - Hamilton_t1
        return ([self.k, self.r, rho], reward)

    def Hamiltonian(self, i):
        alfa_1 = -0.5
        alfa_2 = +0.1
        alfa_3 = +0.2
        alfa_4 = -0.5

        fourth = np.zeros(self.N)
        for j in range(self.N):
            if i != j:
                fourth[i] += (self.A[i][j] / (((self.x[i] - self.x[j]) ** 2 + (self.y[i] - self.y[j]) ** 2) ** 0.5))

        H = alfa_1 * self.k[i] ** 2 + alfa_2 * self.k[i] ** 3 + alfa_3 * self.r[i] ** 2 + alfa_4 * fourth[i]
        return H, fourth[i]  ###

    def Plot(self, episode):
        options = {'node_size': 60, 'width': 0.3}

        G = nx.from_numpy_array(self.A)
        for i in range(self.N):
            G.add_node(i, pos=(self.x[i], self.y[i]))
        pos = nx.get_node_attributes(G, 'pos')

        nx.draw_networkx(G, pos, with_labels=False, **options)

        plt.text(self.L - 0.15 * self.L, self.L + 0.3, f'Episode {episode}', fontname='Comic Sans MS', fontsize=12)
        plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.xlim(-0.1, self.L + 0.1); plt.ylim(-0.1, self.L + 0.1); plt.grid(alpha=0.3)
        camera.snap()


# ------------------------------------------------------------------
N = 4                     # Number of agents
L = 2                     # The length of the simulation box

env = Distributed_System(N,L)

# ------------------------------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)

input_shape = [3]              # == env.observation_space.shape
n_outputs = 3                  # == env.action_space.n

model = []                     # make a model for each agent
for i in range(N):
    model.append(keras.models.Sequential([
        keras.layers.Dense(2, activation="relu", input_shape=input_shape),
        keras.layers.Dense(2, activation="relu"),
        keras.layers.Dense(n_outputs)
    ]))

# ------------------------------------------------------------------
def play_one_step(env, state, epsilon=0.0):
    action = np.zeros(N)

    for i in range(N):
        if np.random.rand() < epsilon:
            action[i] = np.random.randint(n_outputs)
        else:
            actionnnn = model[0].predict(np.reshape(state[i], (3)).reshape(1, -1), verbose=0)[0]
            action[i] = np.argmax(actionnnn)

    next_state, reward = env.step((action - 0.5) * 2)
    next_state = [[next_state[j][i] for j in range(3)] for i in range(N)]

    for i in range(N):
        replay_memory[i].append((state[i], action[i], reward[i], next_state[i]))
    return next_state, reward

# ------------------------------------------------------------------
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
loss_fn = tf.keras.losses.mean_squared_error


def training_step(i, batch_size, discount):
    discount_rate = discount

    indices = np.random.randint(len(replay_memory[i]), size=batch_size)  # 32ta random number between[0 - len(replay)]
    batch = [replay_memory[i][index] for index in indices]  # number in replay_memory[indices]

    states, actions, rewards, next_states = [  # from replay_memory read these and save in...
        np.array([experience[field_index] for experience in batch])
        for field_index in range(4)]

    for j in range(len(rewards)):
        if rewards[j] < 0.05 and rewards[j] > -0.1: rewards[j] = 0

    next_Q_values = model[0].predict(next_states, verbose=0)  # 32 predict of 2 actions
    max_next_Q_values = np.max(next_Q_values, axis=1)  # choose higher probiblity of each actions (of each 32)
    if max_next_Q_values < 0: discount_rate = 1 / discount
    target_Q_values = rewards + discount_rate * max_next_Q_values  # Equation 18-5. Q-Learning algorithm
    target_Q_values = target_Q_values.reshape(-1, 1)  # reshape to (32,1) beacuse of Q_values.shape

    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model[0](states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model[0].trainable_variables)
    optimizer.apply_gradients(zip(grads, model[0].trainable_variables))

    return all_Q_values, next_Q_values, Q_values, target_Q_values, loss, grads, model[0].trainable_variables  ###


# ------------------------------------------------------------------
# f.close()
f = open("log_C.txt", "w")
f.write(f"episode \t epsilon \t ")
for i in range(N): f.write(f"r{i} \t ")
for i in range(N): f.write(f"k{i} \t ")
for i in range(N): f.write(f"k{i}**2 \t ")
for i in range(N): f.write(f"k{i}**3 \t ")
for i in range(N): f.write(f"r{i}**2 \t ")
for i in range(N): f.write(f"fourth{i} \t ")
for i in range(N): f.write(f"H{i} \t ")
for i in range(N): f.write(f"reward{i} \t ")
f.write(f"Hamilton \t\t")
for i in range(N): f.write(f"All Q{i} \t next Q{i} \t Q{i} \t target Q{i} \t LOSS{i} \t")  ###
f.write(f"\t")
for j in range(3): f.write(f"Grads {j} \t bias {j} \t ")  ###
for j in range(3): f.write(f"Trainable {j} \t bias {j} \t ")  ###


def log_print(env, episode, epsilon, H, reward, all_q, next_q, qq, target_q, LOSS, grad, trainable_variables):
    f.write(f"\n{episode} \t {epsilon} \t ")
    for i in range(N): f.write(f"{env.r[i]} \t ")
    for i in range(N): f.write(f"{env.k[i]} \t ")
    for i in range(N): f.write(f"{env.k[i] ** 2} \t ")
    for i in range(N): f.write(f"{env.k[i] ** 3} \t ")
    for i in range(N): f.write(f"{env.r[i] ** 2} \t ")
    for i in range(N): f.write(f"{env.Hamiltonian(i)[1]} \t ")
    for i in range(N): f.write(f"{env.Hamiltonian(i)[0]} \t ")
    for i in range(N): f.write(f"{reward[i]} \t ")
    f.write(f"{H} \t\t")

    for i in range(N):
        f.write(f"{all_q} \t ")
        f.write(f"{next_q} \t ")
        f.write(f"{qq} \t ")
        f.write(f"{target_q} \t ")
        f.write(f"{LOSS} \t ")

    f.write(f"\t")
    for j in range(6): f.write(f"{grad[j].numpy().ravel()} \t ")
    for j in range(6): f.write(f"{trainable_variables[j].numpy().ravel()} \t ")


# ------------------------------------------------------------------
N = 4  # Number of agents
L = 2  # The length of the simulation box
env = Distributed_System(N, L)

batch_size = 10
discount_rate = 0.98

best_H = 0; H = 0; LOSS = 0
Hamilton = []
replay_memory = []
best_weights = []
for i in range(N):
    replay_memory.append(deque(maxlen=40))
    best_weights.append(model[i].get_weights())

state, reward = env.step(np.zeros(N))
state = [[state[j][i] for j in range(3)] for i in range(N)]

camera = Camera(plt.figure())

for episode in range(3000):
    epsilon = max(1 - episode / 1000, 0.0)  # first is more random and than use greedy
    state, reward = play_one_step(env, state, epsilon)

    if episode > 50:
        for i in range(N):
            all_q, next_q, qq, target_q, LOSS, grad, trainable_variables = training_step(i, batch_size, discount_rate)
        log_print(env, episode, epsilon, H, reward,
                  all_q[0], next_q[0], qq[0], target_q[0], LOSS, grad, trainable_variables)

    H = 0
    for i in range(N): H += env.Hamiltonian(i)[0]  # hamiltonian of the whole system  ###
    Hamilton.append(H)

    if H <= best_H and episode > 1000:  # find the minimum of Hamiltonian
        for i in range(N):
            best_weights[i] = model[i].get_weights()  # saving model weights for the best Hamiltonian
            best_H = H

    if episode % 10 == 0:
        env.Plot(episode)

    print("\rEpisode: {}, eps: {:.3f}, Min(Hamilton): {:.3f}, H: {:.3f}".format(episode, epsilon, best_H, H), end="")

# model.set_weights(best_weights)
anim = camera.animate(interval=100, repeat=True, repeat_delay=500, blit=True)
anim.save('./animation/animation_C.gif')
f.close()


# ------------------------------------------------------------------
# ------------------------------------------------------------------
print(best_H)
plt.plot(Hamilton)
plt.text(1700, max(Hamilton)-1.65, "Min(H): %f" % (min(Hamilton)) )
plt.text(1700, max(Hamilton)-2.15, "Arg(H): %f" % (np.argmin(Hamilton)) )
plt.show()
