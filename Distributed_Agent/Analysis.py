import ast
import numpy as np
import pandas as pd
from openpyxl import Workbook
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from . import Functions

# -------------------------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "legend.fontsize": 16,
    "axes.labelsize": 18,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})

# -------------------------------------------------------------------------------------------
def scientific_3(x, pos): return f'{x}'.replace("000.0"   , "×10$^3$")
def scientific_4(x, pos): return f'{x}'.replace("0000.0"  , "×10$^4$")
def scientific_5(x, pos): return f'{x}'.replace("00000.0" , "×10$^5$")
def scientific_6(x, pos): return f'{x}'.replace("000000.0", "×10$^6$")


# -------------------------------------------------------------------------------------------
def FileParsing(excel_file: pd.DataFrame):
    step = pd.Series.to_numpy(excel_file["Unnamed: 0"])

    columns  = ['Hamilton', 'Giant', 'Edges', 'Energy', 'Tau', 'R_avg'] # Names of saved columns
    col_read = ['', '.1', '.2']     # because the columns in the Excel have the same name, they are numbered in the renaming
    result_sets = {}                # dictionary of all values

    for n in range(3):              # (base, model, request)
        for col in columns: 
            key = f'{col}_{n+1}'    # key of dictionary
            result_sets[key] = pd.Series.to_numpy(excel_file[f"{col}{col_read[n]}"])    # Retrieving values ​​from ensemble file

            if col == 'Edges': result_sets[key] = result_sets[key]/50   # Because in Excel, the total number of edges is recorded, not average

    return step, result_sets


# -------------------------------------------------------------------------------------------
def Ploting(excel_name: str, sheet_name: str, figure_feature: list, output_name: str):

    excel = pd.read_excel(excel_name, sheet_name=sheet_name)
    step, result_sets = FileParsing(excel)

    figsize, (num_raw, num_col), labelpad, ylim, legend, bbox_to_anchor = figure_feature
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(num_raw, num_col)

    # depending on how we want to perform the ritual ---------------------
    if num_col == 1: ax = [plt.subplot(gs[i, 0]) for i in range(num_raw) ]      # Two-column article
    else:                                                                       # Single column article
        ax = []
        for i in range(num_raw):    ax.append( plt.subplot(gs[0, 3*i:3*(i+1)]) )
        for i in range(num_col//2): ax.append( plt.subplot(gs[1, 2*i:2*(i+1)]) )


    # --------------------------------------------------------------------
    title  = ["Connectivity (%)", "Hamiltonian", "Energy (sum)", "Radius (avr)", "Degrees (avr)"]
    plot_  = ["Giant", "Hamilton", "Energy", "R_avg", "Edges"]
    colors = ['#F96E2A', '#E73879', '#441752']
    shape  = [':', '--', '-']

    # Drawing graphs -----------------------------------------------------
    for i in range(len(plot_)):
        for n in range(len(shape)):
            key = f'{plot_[i]}_{n+1}'
            ax[i].plot(step, result_sets[key], shape[n] , color=colors[n])

        ax[i].set_ylabel(title[i], labelpad=labelpad[i])
        ax[i].locator_params(axis='y', nbins=4)
        ax[i].set_xlim([-5,1000])
        ax[i].set_ylim(ylim[i])
        ax[i].grid(alpha= 0.3)

    # ax[1].yaxis.set_major_formatter(FuncFormatter(scientific_5))
    ax[0].legend(legend, bbox_to_anchor=bbox_to_anchor, ncol=1)
    plt.tight_layout(w_pad=-0.8)
    plt.savefig(output_name, dpi=330, bbox_inches='tight')


# -------------------------------------------------------------------------------------------
def Ploting_multi(excels_list: list, sheet_name: str, figure_feature: list, output_name: str):

    results_sets = []
    for excel_name in excels_list:
        excel = pd.read_excel(excel_name, sheet_name=sheet_name)
        step, result_sets = FileParsing(excel)
        results_sets.append(result_sets)

    # --------------------------------------------------------------------
    figsize, (num_raw, num_col), labelpad, ylim, legend, bbox_to_anchor, ncol = figure_feature
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(num_raw, num_col)

    # depending on how we want to perform the ritual ---------------------
    if num_col == 1: ax = [plt.subplot(gs[i, 0]) for i in range(num_raw) ]      # Two-column article
    else:                                                                       # Single column article
        ax = []
        for i in range(num_raw):    ax.append( plt.subplot(gs[0, 3*i:3*(i+1)]) )
        for i in range(num_col//2): ax.append( plt.subplot(gs[1, 2*i:2*(i+1)]) )


    # --------------------------------------------------------------------
    title  = ["Connectivity (%)", "Hamiltonian", "Energy (sum)", "Radius (avr)", "Degrees (avr)"]
    plot_  = ["Giant", "Hamilton", "Energy", "R_avg", "Edges"]
    colors = ['#F96E2A', '#E73879', '#441752', '#FFD63A', '#8E1616', '#3A59D1']
    shape  = [':', '--', '-', '-.', '-', '--']

    # Drawing graphs -----------------------------------------------------
    for i in range(len(plot_)):
        for n in range(len(excels_list)):
            key = f'{plot_[i]}_{2+1}'
            if plot_[i] == 'Energy': results_sets[n][key] *= 5
            ax[i].plot(step, results_sets[n][key], shape[n], color=colors[n])

        ax[i].set_ylabel(title[i], labelpad=labelpad[i])
        ax[i].locator_params(axis='y', nbins=4)
        ax[i].set_xlim([-5,1000])
        ax[i].set_ylim(ylim[i])
        ax[i].grid(alpha= 0.3)

    # ax[1].yaxis.set_major_formatter(FuncFormatter(scientific_5))
    ax[0].legend(legend, bbox_to_anchor=bbox_to_anchor, ncol=ncol)
    plt.tight_layout(w_pad=-0.8)
    plt.savefig(output_name, dpi=330, bbox_inches='tight')


# -------------------------------------------------------------------------------------------
def Make_Animation(excel_path: str, excel_sheet: str, ENV_Parameters: list, Parameters: list, on_off: list, seed: int, output_name: str):
    result = pd.read_excel(excel_path, sheet_name=excel_sheet)

    WB = Workbook()
    model_path = f'./All Results/Different Model Training/models with alpha4 = -1000/L=45/model_H_best_weight.keras'
    plot_env, Agents, _, _ = Functions.Initializer(ENV_Parameters, Parameters, [0,0,0], model_path, WB, WB, '', seed)

    N = Parameters[0]
    ind = -N
    c = 0
    for step in range(1001):
        ind += N
        if c != len(on_off):
            if step == on_off[c][0]: N3 += on_off[c][1]; c += 1     # If any factors were removed or added during the run

        connection = result["connected to"][ind:ind+N].apply(lambda cell: ast.literal_eval(cell))        # string to list

        A = np.zeros((N, N))
        for i in range(N):
            for j in connection[ind + i]: A[i][j] = 1       # Creating an adjacency matrix from connections stored in Excel
        plot_env.A = A                                      # Casting the adjacency matrix into the Plot_Environment class object

        for i in range(N): Agents[i].x, Agents[i].y = result["X"][ind+i], result["Y"][ind+i] # Reading agent locations from an Excel

        if step%5==0: plot_env.Animation(step)
    
    anim = plot_env.camera.animate(interval= 120, repeat=True, repeat_delay= 500, blit=True)
    anim.save(output_name)


# -------------------------------------------------------------------------------------------
