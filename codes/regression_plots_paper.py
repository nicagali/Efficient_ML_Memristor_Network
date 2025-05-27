import parameters as par
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import ahkab  
import networks
import training
import plotting
import matplotlib.gridspec as gridspec


graph_id = 'G00010002'
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
plotting.plot_regression(ax4, graph_id, 'length', int(50))
ax4.legend(fontsize = par.legend_size)

ax5 = fig.add_subplot(bottom[0, 2])
plotting.plot_regression(ax5, graph_id, 'length', step=400)
ax5.legend(fontsize = par.legend_size)


# fig.tight_layout()
fig.savefig(f"../paper/plots/regression/{graph_id}.pdf")


fig = plt.figure(figsize=(13, 6.5), constrained_layout=True)
outer = gridspec.GridSpec(1, 2, width_ratios=[1, 2], figure=fig)

# Right column for MSE plot
ax2 = fig.add_subplot(outer[0, 1])
plotting.plot_mse(ax2, fig, graph_id, training_type, 'length')
plotting.plot_mse(ax2, fig, graph_id, training_type, 'radius_base')
plotting.plot_mse(ax2, fig, graph_id, training_type, 'rho')
plotting.plot_mse(ax2, fig, graph_id, training_type, 'pressure')
ax2.legend(fontsize=par.legend_size)

# Add inset axes inside ax2
inset_ax1 = ax2.inset_axes([0.07, 0.5, 0.2, 0.2])  # [x0, y0, width, height]
plotting.plot_regression(inset_ax1, graph_id, 'length', step=0)
# inset_ax1.set_title("Step 0", fontsize=(par.size_ticks-7))
inset_ax1.tick_params(axis='both', labelsize=(par.size_ticks-7))

inset_ax2 = ax2.inset_axes([0.55, 0.6, 0.2, 0.2])
plotting.plot_regression(inset_ax2, graph_id, 'length', step=50)
# inset_ax2.set_title("Step 50", fontsize=(par.size_ticks-7))
inset_ax2.tick_params(axis='both', labelsize=(par.size_ticks-7))

inset_ax3 = ax2.inset_axes([0.7, 0.1, 0.2, 0.2])
plotting.plot_regression(inset_ax3, graph_id, 'length', step=400)
# inset_ax3.set_title("Step 400", fontsize=(par.size_ticks-7))
inset_ax3.tick_params(axis='both', labelsize=(par.size_ticks-7))

fig.savefig(f"../paper/plots/regression/{graph_id}_newformat.pdf")
