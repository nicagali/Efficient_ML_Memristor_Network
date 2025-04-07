import parameters as par
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import ahkab  
import networks
import training
import plotting
import matplotlib.gridspec as gridspec


graph_id = 'G00010001'
DATA_PATH = f'{par.DATA_PATH}regression{graph_id}/'
G = nx.read_graphml(f'{DATA_PATH}{graph_id}.graphml')
training_steps = 400
training_type = 'regression'
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(12, 8), constrained_layout=True)
outer = gridspec.GridSpec(2, 1, height_ratios=[1, 1], figure=fig)

# First row: 2 columns
top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0])

ax2 = fig.add_subplot(top[0, 1])
plotting.plot_mse(ax2, fig, graph_id, training_type, f'length')
plotting.plot_mse(ax2, fig, graph_id, training_type, f'radius_base')
plotting.plot_mse(ax2, fig, graph_id, training_type, f'rho')
plotting.plot_mse(ax2, fig, graph_id, training_type, f'pressure')
ax2.legend(fontsize = par.legend_size)

bottom = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1])
ax3 = fig.add_subplot(bottom[0, 0])
plotting.plot_regression(ax3, graph_id, 'length', step=0)
ax3.legend(fontsize = par.legend_size)

ax4 = fig.add_subplot(bottom[0, 1])
plotting.plot_regression(ax4, graph_id, 'length', 50)
ax4.legend(fontsize = par.legend_size)

ax5 = fig.add_subplot(bottom[0, 2])
plotting.plot_regression(ax5, graph_id, 'length', step=training_steps)
ax5.legend(fontsize = par.legend_size)


# fig.tight_layout()
fig.savefig(f"../paper/plots/regression/G00010001.pdf")
