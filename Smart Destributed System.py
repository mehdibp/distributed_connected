import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
from collections import deque
import networkx as nx
import time

import tensorflow as tf
# from tensorflow import keras

# %matplotlib notebook


# ------------------------------------------------------------------
class Distributed_System():
    def __init__(self, N, L):
        np.random.seed(41)
        self.N = N
        self.L = L

        # Variables: x , y (positions of agents in plan)
        self.x = np.random.rand(N) * L  # Initialize xᵢ
        self.y = np.random.rand(N) * L  # Initialize yᵢ

        self.r = np.ones(N)  # Wave sending radius
        self.A = np.zeros((N, N))  # Adjacency Matrix
        self.k = np.zeros(N)  # Degree of a vertex
        for i in range(N):
            for j in range(i + 1, N):
                if ((self.x[i] - self.x[j]) ** 2 + (self.y[i] - self.y[j]) ** 2) ** 0.5 < self.r[i]:
                    self.A[i][j] = self.A[j][i] = 1
                    self.k[i] += self.A[i][j]
                    self.k[j] += self.A[i][j]

    def step(self, r):
        Hamilton_t0 = np.zeros(self.N)
        Hamilton_t1 = np.zeros(self.N)
        for i in range(self.N):
            Hamilton_t0[i], e = self.Hamiltonian(i)  ###

        self.A = np.zeros((N, N))
        self.r = r
        self.k = np.zeros(N)
        mean_r_ij = np.zeros((N))

        for i in range(self.N):
            for j in range(i + 1, self.N):
                distance = ((self.x[i] - self.x[j]) ** 2 + (self.y[i] - self.y[j]) ** 2) ** 0.5
                if distance < self.r[i] and distance < self.r[j]:
                    self.A[i][j] = self.A[j][i] = 1
                    self.k[i] += self.A[i][j]
                    self.k[j] += self.A[i][j]

                    mean_r_ij[i] += distance
                    mean_r_ij[j] += distance

        for i in range(self.N):
            Hamilton_t1[i], e = self.Hamiltonian(i)  ###
        reward = Hamilton_t0 - Hamilton_t1
        return ([self.k, self.r, mean_r_ij / (self.k + 1e-10)], reward)

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

    def Plot(self, episode):  ###
        options = {'node_size': 60, 'width': 0.3}

        G = nx.from_numpy_array(self.A)
        for i in range(self.N):
            G.add_node(i, pos=(self.x[i], self.y[i]))
        pos = nx.get_node_attributes(G, 'pos')

        nx.draw_networkx(G, pos, with_labels=False, **options)

        plt.text(self.L - 0.15 * self.L, self.L + 0.3, f'Episode {episode}', fontname='Comic Sans MS', fontsize=12)  ###
        plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.xlim(-0.1, self.L + 0.1)
        plt.ylim(-0.1, self.L + 0.1)
        plt.grid(alpha=0.3)
        camera.snap()

# ------------------------------------------------------------------
N = 3                      # Number of agents
L = 2                      # The length of the simulation box
env = Distributed_System(N,L)

# ------------------------------------------------------------------
# Create Sequential Model
tf.random.set_seed(41)
np.random.seed(41)

input_shape = [3]              # == env.observation_space.shape
n_outputs = 1                  # == env.action_space.n

model = []
for i in range(N):
    model.append(tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation="relu", input_shape=input_shape),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(n_outputs, activation="sigmoid")
    ]))


# ------------------------------------------------------------------
replay_memory = []
for i in range(N):
    replay_memory.append(deque(maxlen=800))


def play_one_step(env, state, epsilon=0.):
    action = np.zeros(N)

    for i in range(N):
        if np.random.rand() < epsilon:
            action[i] = np.random.rand(n_outputs) * L
        else:
            action[i] = model[i].predict(np.reshape(state[i], (3)).reshape(1, -1), verbose=0)[0] * L

    next_state, reward = env.step(action)
    next_state = [[next_state[j][i] for j in range(3)] for i in range(N)]

    for i in range(N):
        replay_memory[i].append((state[i], action[i], reward[i], next_state[i]))
    return next_state, reward


# ------------------------------------------------------------------
batch_size = 32
discount_rate = 0.95
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-2)
loss_fn   = tf.keras.losses.mean_squared_error


def training_step(i, batch_size):
    indices = np.random.randint(len(replay_memory[i]), size=batch_size)  # 32ta random number between[0 - len(replay)]
    batch = [replay_memory[i][index] for index in indices]  # number in replay_memory[indices]

    states, actions, rewards, next_states = [  # from replay_memory read these and save in...
        np.array([experience[field_index] for experience in batch])
        for field_index in range(4)]

    next_Q_values = model[i].predict(next_states, verbose=0) * L  # 32 predict of 2 actions
    next_Q_values = np.squeeze(next_Q_values)  # reshape to (32,) beacuse of rewards.shape
    target_Q_values = rewards + discount_rate * next_Q_values  # Equation 18-5. Q-Learning algorithm
    target_Q_values = target_Q_values.reshape(-1, 1)  # reshape to (32,1) beacuse of Q_values.shape

    with tf.GradientTape() as tape:
        Q_values = model[i](states) * L
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model[i].trainable_variables)
    optimizer.apply_gradients(zip(grads, model[i].trainable_variables))


# ------------------------------------------------------------------
state, reward = env.step(np.ones(N))
state = [[state[j][i] for j in range(3)] for i in range(N)]

f = open("log.txt", "w")
f.write(f"episode \t epsilon \t ")
for i in range(N): f.write(f"r{i} \t ")
for i in range(N): f.write(f"k{i} \t ")
for i in range(N): f.write(f"k{i}**2 \t ")
for i in range(N): f.write(f"k{i}**3 \t ")
for i in range(N): f.write(f"r{i}**2 \t ")
for i in range(N): f.write(f"fourth{i} \t ")
for i in range(N): f.write(f"H{i} \t ")
for i in range(N): f.write(f"reward{i} \t ")
f.write(f"Hamilton \t Time(one_step) \t Time(training) \n")

camera = Camera(plt.figure())
start_total = time.time()
for episode in range(4000):
    Hamilton = 0
    epsilon = max(1 - (episode) / 500, 0.01)  # first is more random and than use greedy

    ti_onestep = time.time()  ### not shown
    state, reward = play_one_step(env, state, epsilon)
    tf_onestep = time.time()  ### not shown

    ti_train = time.time()  ### not shown
    if episode > 200:
        for i in range(N):
            training_step(i, batch_size)
    tf_train = time.time()  ### not shown

    if episode % 10 == 0:
        env.Plot(episode)

    for i in range(N):
        Hamilton += env.Hamiltonian(i)[0]

    f.write(f"{episode} \t {epsilon} \t ")
    for i in range(N): f.write(f"{env.r[i]} \t ")
    for i in range(N): f.write(f"{env.k[i]} \t ")
    for i in range(N): f.write(f"{env.k[i] ** 2} \t ")
    for i in range(N): f.write(f"{env.k[i] ** 3} \t ")
    for i in range(N): f.write(f"{env.r[i] ** 2} \t ")
    for i in range(N): f.write(f"{env.Hamiltonian(i)[1]} \t ")
    for i in range(N): f.write(f"{env.Hamiltonian(i)[0]} \t ")
    for i in range(N): f.write(f"{reward[i]} \t ")
    f.write(f"{Hamilton} \t {tf_onestep - ti_onestep} \t {tf_train - ti_train} \n")

    print("\rEpisode: {}, eps: {:.3f}, Time One Step: {:.3f}".format(episode, epsilon, tf_onestep - ti_onestep), end="")

print(f"\nTotal Time: {time.time() - start_total}")

anim = camera.animate(interval=100, repeat=True, repeat_delay=500, blit=True)
anim.save('animation_.gif')
f.close()