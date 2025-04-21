import random
import numpy as np
import networkx as nx
from openpyxl import Workbook
import importlib

from . import Agent
from . import Environment



# -------------------------------------------------------------------------------------------
def Initializer(ENV_Parameters: list, Parameters: list, Functions: list, model_path: str, ParticleExcel: Workbook, AveragedExcel: Workbook, sheet_name: str, seed: int=42):
    importlib.reload(Agent); importlib.reload(Environment)
    np.random.seed(seed);    random.seed(seed)

    # Build the environment and the agents ----------------------------------------------
    environment = Environment.Environment(ENV_Parameters)

    Agents = []
    for _ in range(Parameters[0]): 
        Agents.append(Agent.Distributed_Agent(environment, Parameters, Functions, model_path))
    
    plot_env = Environment.Plot_Environment(environment, Agents)


    # Creating Excel files for storage --------------------------------------------------
    ParticleExcel_ = ParticleExcel.create_sheet(title=sheet_name)
    AveragedExcel_ = AveragedExcel.create_sheet(title=sheet_name)
    ParticleExcel_.append(["step", "index", "X", "Y", "Component", "r", "k", "rho", "Hamilton", "connected to"
                           , "state", "action", "reward", "next_state", "loss"])
    AveragedExcel_.append(["step", "Hamilton", "Giant", "Edges", "Energy", "Tau", "R_avg", "loss"])

    
    return plot_env, Agents, ParticleExcel_, AveragedExcel_


# -------------------------------------------------------------------------------------------
def ExportResults(step: int, plot_env: Environment.Plot_Environment, Agents: list, AveragedExcel_, ParticleExcel_):
    plot_env.Environmental_Changes(Agents)
    G = nx.from_numpy_array(plot_env.A)
    Giant = sorted(nx.connected_components(G), key=len, reverse=True)
     
    # Show Averaged Results ---------------------------------------------------------------------
    # step	Hamilton	Giant	Edges	Energy	Tau	R_avg loss ----------------------------------
    hamilton, edge, energy, average_r, giant, tau = plot_env.Calculate_Result(Agents, step)
    loss = 0; 
    for i in range(len(Agents)): loss += Agents[i].loss.numpy()
    AveragedExcel_.append([step, hamilton, giant, edge, energy, tau, average_r, loss])


    # Show Particle Results ---------------------------------------------------------------------
    # step	index	X	Y	Component	r	k	rho	Energy	connected_to ------------------------
    # state   action  reward  next_state    loss ------------------------------------------------
    for i in range(len(Agents)):
        particle_hamilton = Agents[i].Hamiltonian()
        particle_k, particle_r, particle_rho = Agents[i].MyState()
        state, action, reward, next_state    = Agents[i].replay_memory[-1]

        for j in range(len(Giant)):
            if i in Giant[j]: component = j

        co = []
        for j in range(len(Agents)):
            if plot_env.A[i][j] == 1: co.append(j) 

        ParticleExcel_.append([step, i, Agents[i].x, Agents[i].y, component, 
                                particle_r, particle_k, particle_rho, particle_hamilton, str(co), 
                                str(state), action, reward, str(next_state), Agents[i].loss.numpy()])


# -------------------------------------------------------------------------------------------
def base_model(plot_env: Environment.Plot_Environment, Agents: list, ParticleExcel_, AveragedExcel_):

    for step in range(1001):

        for i in range(len(Agents)):
            Agents[i].N = len(Agents)
            Agents[i].OtherAgents = (Agents[:i] + Agents[i+1:])

            state  = [s+1 for s in Agents[i].MyState()]
            if Agents[i].k < 5: action = np.random.randint(2)
            else: action = 0
        
            STEP = Agents[i].step(action)
            reward = STEP[1]
            next_state = [ns+1 for ns in STEP[0]]

            Agents[i].replay_memory.append((state, action, reward, next_state))


        # -------------------------------------------------------------------------------------------
        ExportResults(step, plot_env, Agents, AveragedExcel_, ParticleExcel_)
        print("\rStep: {}".format(step), end="")


# -------------------------------------------------------------------------------------------
def AI_model(plot_env: Environment.Plot_Environment, Agents: list, ParticleExcel_, AveragedExcel_):

    for step in range(1001):
        for i in range(len(Agents)):
            Agents[i].Prediction( (Agents[:i] + Agents[i+1:]) )

        
        # -------------------------------------------------------------------------------------------
        ExportResults(step, plot_env, Agents, AveragedExcel_, ParticleExcel_)
        print("\rStep: {}".format(step), end="")


