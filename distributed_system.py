import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

class DistributedSystem():
    def __init__(self, N, L):
        np.random.seed(41)
        self.N = N
        self.L = L

        # Variables: x , y (positions of agents in plan)
        self.x = np.random.rand(N) * L  # Initialize xᵢ
        self.y = np.random.rand(N) * L  # Initialize yᵢ

        self.r = np.ones(N)  # Wave sending radius
        self.A = np.zeros((N, N))  # Adjacency Matrix
        self.k = np.zeros(N)  #
        for i in range(N):
            for j in range(i + 1, N):
                if ((self.x[i] - self.x[j]) ** 2 + (self.y[i] - self.y[j]) ** 2) ** 0.5 < self.r[i]:
                    self.A[i][j] = self.A[j][i] = 1
                    self.k[i] += self.A[i][j]
                    self.k[j] += self.A[j][i]

    def step(self, i, r):
        H_t = self.Hamiltonian(i)

        self.A[i][:] = 0
        self.r[i] = r
        self.k[i] = 0
        mean_r_ij = 0

        for j in range(self.N):
            if i != j:
                distance = ((self.x[i] - self.x[j]) ** 2 + (self.y[i] - self.y[j]) ** 2) ** 0.5
                if distance < self.r[i]:
                    self.A[i][j] = self.A[j][i] = 1

                    self.k[i] += 1
                    mean_r_ij += distance

        H_t1 = self.Hamiltonian(i)
        reward = H_t - H_t1

        return ([self.k[i], self.r[i], mean_r_ij / (self.k[i] + 1e-10)], reward)

    def Hamiltonian(self, i):
        alfa_1 = -0.5
        alfa_2 = +0.001
        alfa_3 = +0.5
        alfa_4 = -1

        fourth = np.zeros(self.N)
        for j in range(self.N):
            if i != j:
                fourth[i] += (self.A[i][j] / (((self.x[i] - self.x[j]) ** 2 + (self.y[i] - self.y[j]) ** 2) ** 0.5))

        H = alfa_1 * self.k[i] ** 2 + alfa_2 * self.k[i] ** 3 + alfa_3 * self.r[i] ** 2 + alfa_4 * fourth[i]
        #         print(H)
        return H

    def Plot(self):
        fig, ax = plt.subplots()
        options = {'node_size': 60, 'width': 0.3}

        G = nx.from_numpy_array(self.A)
        for i in range(self.N):
            G.add_node(i, pos=(self.x[i], self.y[i]))
        pos = nx.get_node_attributes(G, 'pos')

        nx.draw_networkx(G, pos, with_labels=False, **options)

        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)  # turn the axis on
        plt.xlim(-0.1, self.L + 0.1)
        plt.ylim(-0.1, self.L + 0.1)
        plt.grid(alpha=0.3)
        plt.show()
