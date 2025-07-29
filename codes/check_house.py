import networks
import matplotlib.pyplot as plt
import plotting
import parameters as par
import time 
import training
import networkx as nx
import sys
sys.path.append(par.PACKAGE_PATH)
import ahkab  

PLOT_PATH = f'{par.PLOT_PATH}single_memristor/'

G = networks.single_memristor(save_data=True) 

fig, ax = plt.subplots()
plotting.plot_memristor_resistances(ax,G)
ax.legend(fontsize = par.legend_size)
fig.tight_layout()
fig.savefig(f"{PLOT_PATH}conductance.pdf", transparent=True)


