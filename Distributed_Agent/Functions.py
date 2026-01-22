import numpy as np
import networkx as nx
from openpyxl import Workbook

from . import Agent
from . import Environment



# -------------------------------------------------------------------------------------------
def Initializer(ENV_Parameters: list, Parameters: list, Functions: list, model_path: str=None, seed: int=30):
    """
    Args:
        ENV_Parameters (list): [L, buildings_type, num_buildings, num_streets]
        Parameters     (list): [N, L, Alphas, learning_rate, discount_rate, batch_size, steps_per_train]
        Functions      (list): [requesting, moving, training]
        model_path     (str):  Address and name of the initial saved model
        seed           (int):  Fixed initial seed for easier comparison
    """
    
    np.random.seed(seed)

    # Build the environment and the agents ----------------------------------------------
    environment = Environment.Environment(ENV_Parameters)

    Agents = []
    for _ in range(Parameters[0]): 
        Agents.append(Agent.Distributed_Agent(environment, Parameters[1:], Functions, model_path))
    
    plot_env = Environment.Plot_Environment(environment, Agents)

    return plot_env, Agents


# -------------------------------------------------------------------------------------------
def ExportResults(step: int, plot_env: Environment.Plot_Environment, Agents: list[Agent.Distributed_Agent]):
    """
    Args:
        step     (int):       The time step we are in
        plot_env (Plot_Env):  An object of the Plot_Environment class to extract its parameters
        Agents   (list):      List of all agents (objects created from the Distributed_Agent class)
    """

    plot_env.Environmental_Changes(Agents)
    G = nx.from_numpy_array(plot_env.A)
    Giant = sorted(nx.connected_components(G), key=len, reverse=True)
     
    # Show Averaged Results ---------------------------------------------------------------------
    # step	Hamilton	Giant	Edges	Energy	R_avg loss --------------------------------------
    hamilton, edge, energy, average_r, giant = plot_env.Calculate_Result(Agents)
    loss = np.mean([a.loss.numpy() for a in Agents])
    AveragedResult = [step, hamilton, giant, edge, energy, average_r, loss]


    # Show Particle Results ---------------------------------------------------------------------
    # step	index	X	Y	Component	r	k	rho	Energy	connected_to ------------------------
    # state   action  reward  next_state    loss ------------------------------------------------
    ParticleResult = []

    for i in range(len(Agents)):
        particle_hamilton = Agents[i].hamiltonian()
        particle_k, particle_r, particle_rho = Agents[i].my_state()
        state, action, reward, next_state    = Agents[i].replay_memory[-1]

        for j in range(len(Giant)):
            if i in Giant[j]: component = j

        co = []
        for j in range(len(Agents)):
            if plot_env.A[i][j] == 1: co.append(j) 

        ParticleResult.append([step, i, Agents[i].x, Agents[i].y, component, 
                               particle_r, particle_k, particle_rho, particle_hamilton, str(co), 
                               str(state), action, reward, str(next_state), Agents[i].loss.numpy()])
        
    return AveragedResult, ParticleResult


# -------------------------------------------------------------------------------------------
def base_model(plot_env: Environment.Plot_Environment, Agents: list[Agent.Distributed_Agent], MAX_step: int=1001):
    """
    Args:
        plot_env (Plot_Env):  An object of the Plot_Environment class to extract its parameters
        Agents   (list):      List of all agents (objects created from the Distributed_Agent class) 
    """

    AveragedResults, ParticleResults = [], []

    for step in range(MAX_step):

        for i in range(len(Agents)):
            neighbors = Agent.Agent_Interaction(Agents[i]).update_neighbors(Agents)
            Agents[i].neighbors = neighbors
            Agents[i].N_density = len(Agents)/(plot_env.L**2)

            state  = [s+1 for s in Agents[i].my_state()]
            if Agents[i].k < 5: action = np.random.randint(2)
            else: action = 0
        
            STEP = Agents[i].step(action)
            reward = STEP[1]
            next_state = [ns+1 for ns in STEP[0]]

            Agents[i].replay_memory.append((state, action, reward, next_state))


        # -------------------------------------------------------------------------------------------
        AveragedResult, ParticleResult = ExportResults(step, plot_env, Agents)
        AveragedResults.append(AveragedResult)
        ParticleResults.append(ParticleResult)
        print("\rStep: {}".format(step), end="")

    return AveragedResults, ParticleResults


# -------------------------------------------------------------------------------------------
def AI_model(plot_env: Environment.Plot_Environment, Agents: list[Agent.Distributed_Agent], MAX_step: int=1001):
    """
    Args:
        plot_env (Plot_Env):  An object of the Plot_Environment class to extract its parameters
        Agents   (list):      List of all agents (objects created from the Distributed_Agent class) 
    """

    AveragedResults, ParticleResults = [], []

    for step in range(MAX_step):
        for i in range(len(Agents)):
            neighbors = Agent.Agent_Interaction(Agents[i]).update_neighbors(Agents)
            Agents[i].prediction( neighbors, len(Agents)/(plot_env.L**2) )

        
        # -------------------------------------------------------------------------------------------
        AveragedResult, ParticleResult = ExportResults(step, plot_env, Agents)
        AveragedResults.append(AveragedResult)
        ParticleResults.append(ParticleResult)
        print("\rStep: {}".format(step), end="")

    return AveragedResults, ParticleResults


# -------------------------------------------------------------------------------------------
def SaveExcel(ParticleExcel: Workbook, AveragedExcel: Workbook, sheet_name: str, AveragedResults: list, ParticleResults: list):
    """
    Arg:
        ParticleExcel   (Workbook):  Excel file created to store particle details at irregular times
        AveragedExcel   (Workbook):  Excel file created to store averaged parameters
        sheet_name      (str):       Sheet name in Excel file
        AveragedResults (List):      
        ParticleResults (List):      
    """
    # Creating Excel files for storage --------------------------------------------------
    ParticleExcel_ = ParticleExcel.create_sheet(title=sheet_name)
    AveragedExcel_ = AveragedExcel.create_sheet(title=sheet_name)

    ParticleExcel_.append(["step", "index", "X", "Y", "Component", "r", "k", "rho", "Hamilton", "connected to", "state", "action", "reward", "next_state", "loss"])
    AveragedExcel_.append(["step", "Hamilton", "Giant", "Edges", "Energy", "R_avg", "loss"])
    
    ParticleResults = [ particle for step in ParticleResults for particle in step ]
    for p in ParticleResults: ParticleExcel_.append(list(p))
    for a in AveragedResults: AveragedExcel_.append(a)
