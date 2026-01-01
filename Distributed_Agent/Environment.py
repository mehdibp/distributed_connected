import numpy as np
import networkx as nx
from celluloid import Camera

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.transforms import Bbox


# -------------------------------------------------------------------------------------------
class Environment():
    # ---------------------------------------------------------------------------------------
    def __init__(self, ENV_Parameters: list):
        """
        Args: ENV_Parameters (list): [L, buildings_type, num_buildings, num_streets]
        """

        self.L, buildings_type, num_buildings, num_streets = ENV_Parameters

        self.buildings = []
        self.build_bounds = []
        # self.border = np.array([[-L, 0, L, L], [0, -L, L, L], [ L, 0, L, L], [ 0, L, L, L]])

        if buildings_type == "random" : self.RandomBuilding (num_buildings)
        if buildings_type == "regular": self.RegularBuilding(num_streets)

    # ---------------------------------------------------------------------------------------
    def BoundaryCondition(self, agent_pos: tuple, radian: float, Rho: float):
        x, y = agent_pos
        if (x > self.L) or (x < 0): 
            radian = np.pi-radian if radian > 0 else -np.pi-radian
            x = Rho if x < 0 else self.L-Rho
        if (y > self.L) or (y < 0): 
            radian = -radian
            y = Rho if y < 0 else self.L-Rho
        
        return x, y, radian
   
    # ---------------------------------------------------------------------------------------
    def RandomBuilding(self, num_buildings: int):
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
    def RegularBuilding(self, num_streets: int):
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
    def do_intersect(self, agents: list):
        agents = Path(agents)
        for build in self.build_bounds:
            if agents.intersects_bbox(build): return True
        return False

    # ---------------------------------------------------------------------------------------
    def BuildingsBoundary(self, agent_pos: tuple, agent_previous_pos: tuple, radian: float, Rho: float):
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
    def __init__(self, environment: Environment, Agents: list):
        """
        Args:
            environment (Env) : An object of the "Environment" class that specifies the environment properties for each agent
            Agents      (list): List of all agents (objects created from the Distributed_Agent class) 
        """

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
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=None, hspace=None)
        self.camera = Camera(fig)


    # ---------------------------------------------------------------------------------------
    def Environmental_Changes(self, Agents: list):
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
    def Env_Plot(self, step: int):
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
    def Animation(self, step: int):
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
    def Calculate_Result(self, Agents: list, s):
        self.Environmental_Changes(Agents)

        # Hamiltonian of the whole system -----------------------------------------
        hamilton = 0
        for i in range(self.N): hamilton += self.Agents[i].hamiltonian()
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

