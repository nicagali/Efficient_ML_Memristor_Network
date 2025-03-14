import networks
import training
import plotting
import parameters as par
import matplotlib.pyplot as plt
import networkx as nx
import time
import matplotlib.gridspec as gridspec
import numpy as np

start = time.time()

# --------- INITIALIZE NETWORK ---------

# -> DEFINE graph from networks module
G = networks.voltage_divider(save_data=True) 

# -> PLOT graph in /plots 
fig, ax = plt.subplots()
pos = plotting.plot_graph('voltage_divider')
fig.tight_layout()
fig.savefig(f"../plots/voltage_divider/graph.pdf")

# --------- TRAIN NETWORK WITH DIFFERENT WEIGHTS ---------

training_steps = 10
training_type = 'allostery'
weight_type_vec = ['length', 'radius_base', 'rho', 'pressure']
delta_weight_vec = [1e-3, 1, 1e-4, 1e-3]
learning_rate_vec = [3e-6, 3e-6, 4e-4, 100]

# for weight_type_index in range(len(weight_type_vec)):
for weight_type_index in range(1):

    G_ml = G.copy(as_view=False)  
    training.train(G_ml, training_type=training_type, training_steps=training_steps, weight_type=weight_type_vec[weight_type_index], delta_weight = delta_weight_vec[weight_type_index], learning_rate=learning_rate_vec[weight_type_index])

# --------- PLOT EVOLUTION OF TRAINED NW ---------

fig, ax = plt.subplots(figsize = par.figsize_1)
plotting.plot_potential_each_node(ax, G_ml)
fig.tight_layout()
fig.savefig(f"../paper/plots/voltage_divider/evolution_finalnw.pdf")

# --------- PLOT ERROR AND WEIGHTS WITH OTHER WEIGHTS ---------

# fig = plt.figure(figsize=(12, 8))
# gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

# ax1 = fig.add_subplot(gs[0, 0:2]) 
# plotting.plot_mse(ax1, fig, f'allostery_pressure', show_xlabel = False)
# plotting.plot_mse(ax1, fig, f'allostery_rho', show_xlabel = False)
# plotting.plot_mse(ax1, fig, f'allostery_length', show_xlabel = False)
# plotting.plot_mse(ax1, fig, f'allostery_radius_base', show_xlabel = False)
# ax1.legend()

# ax2 = fig.add_subplot(gs[1, 0])
# plotting.plot_weights(ax2, G, training_steps=training_steps, rule=f'allostery_rho', show_xlabel=False)

# ax3 = fig.add_subplot(gs[1, 1])
# plotting.plot_weights(ax3, G, training_steps=training_steps, rule=f'allostery_length')

# ax4 = fig.add_subplot(gs[1, 2])
# plotting.plot_weights(ax4, G, training_steps=training_steps, rule=f'allostery_radius_base', show_xlabel=False)

# ax5 = fig.add_subplot(gs[0, 2:3])
# plotting.plot_weights(ax5, G, training_steps=training_steps, rule=f'allostery_pressure', show_xlabel=False)

# # ax6 = fig.add_subplot(gs[0, 2:4])
# # plotting.plot_potential_drops_each_node(ax6, G)

# # Adjust layout and save the figure
# fig.tight_layout()
# fig.savefig(f"../paper/plots/voltage_divider/weights_others.pdf")

# --------- TRAIN NETWORK WITH DIFFERENT TARGETS ---------

# target_values = [1, 3, 4]
# training_steps = 15

# fig, ax = plt.subplots(2, 1, figsize = (8,7))

# G_target = networks.voltage_divider(save_data=True) 
# for target_index in range(len(target_values)):

#     G_target.nodes[1]['desired'] = target_values[target_index]

#     training.train(G_target, training_type=training_type, training_steps=training_steps, weight_type='pressure', delta_weight = 1e-3, learning_rate=100)

#     plotting.plot_weights(ax[1], G_target, training_steps=training_steps, rule=f'allostery_pressure', show_xlabel=True, starting_step=(target_index*training_steps))


# plotting.plot_final_potential_vd(ax[0], target_values)
# ax[0].legend(fontsize = par.legend_size)
# fig.tight_layout()
# fig.savefig(f"../paper/plots/voltage_divider/evolution_targets.pdf")

end = time.time()
print("Running time = ", end-start, "seconds")