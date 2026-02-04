import numpy as np
import matplotlib.pyplot as plt

from ..environments.base import Environment


# -------------------------------------------------------------------------------------------
# plt.rcParams.update({
#     "font.family": "Times New Roman",
#     "legend.fontsize": 16,
#     "axes.labelsize": 18,
#     "xtick.labelsize": 13,
#     "ytick.labelsize": 13,
# })



class ResultPloter():
    # ---------------------------------------------------------------------------------------
    def __init__(self, log_dir: str, environment: Environment):
        self.log_dir = log_dir
        self.environment = environment

        xmin, ymin, xmax, ymax = self.environment.get_bounds()
        area = max((xmax - xmin) * (ymax - ymin), 1e-6)
        self.L = np.sqrt(area)

    # ---------------------------------------------------------------------------------------
    def average_result(self, averaged_results):
        titles = [ "step", "Hamiltonian", "Connectivity (%)", "Degrees (avr)", 
                  "Transition Energy (sum)", "Reduced Radius (avr)", "Loss" ]
        header = [ "step", "Hamiltonian", "Giant (%)", "Edges", "Energy", "Average_r", "Loss" ]

        for i in range(1,len(titles)):
            result = np.array(averaged_results)
            result[:, header.index('Edges')] /= len(self.environment._entities)/2   # Average number of degrees
            result[:, header.index('Average_r')] /= self.L                # Reduced radius due to system length

            plt.figure(figsize=(10,10))
            plt.plot(result[:, 0], result[:, i])

            plt.xlabel(titles[0])
            plt.ylabel(titles[i])
            plt.ticklabel_format(axis='y', style='scientific', scilimits=(-3,3), useMathText=True)
            plt.grid(alpha= 0.3)

            plt.savefig(f"./{self.log_dir}/{titles[i]}.png", dpi=330, bbox_inches='tight')

