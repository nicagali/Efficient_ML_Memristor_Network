import networks
import training
import plotting
import parameters as par
import matplotlib.pyplot as plt
import networkx as nx
import time
import matplotlib.gridspec as gridspec

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

training_steps = 20

G_ml = G.copy(as_view=False)  
# training.train(G_ml, training_steps=training_steps, weight_type='length', delta_weight = 1e-3, learning_rate=3e-6)
G_ml = G.copy(as_view=False)  
# training.train(G_ml, training_steps=training_steps, weight_type='rbrt', delta_weight = 1e-3, learning_rate=1)
G_ml = G.copy(as_view=False)  
# training.train(G_ml, training_steps=training_steps, weight_type='rho', delta_weight = 1e-4, learning_rate=4e-4)
G_pressure = G.copy(as_view=False)  
# training.train(G_pressure, training_steps=training_steps, weight_type='pressure', delta_weight = 1e-3, learning_rate=100)

# --------- PLOT EVOLUTION OF TRAINED NW ---------

# fig, ax = plt.subplots(figsize = par.figsize_1)
# plotting.plot_potential_drops_each_node(ax, G_pressure)
# fig.tight_layout()
# fig.savefig(f"../paper/plots/voltage_divider/evolution_finalnw.pdf")

# --------- PLOT ERROR AND WEIGHTS WITH OTHER WEIGHTS ---------

# fig = plt.figure(figsize=(12, 8))
# gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

# ax1 = fig.add_subplot(gs[0, 0:2]) 
# plotting.plot_mse(ax1, fig, f'allostery_pressure', show_xlabel = False)
# plotting.plot_mse(ax1, fig, f'allostery_rho', show_xlabel = False)
# plotting.plot_mse(ax1, fig, f'allostery_length', show_xlabel = False)
# plotting.plot_mse(ax1, fig, f'allostery_rbrt', show_xlabel = False)
# ax1.legend()

# ax2 = fig.add_subplot(gs[1, 0])
# plotting.plot_weights(ax2, G, training_steps=training_steps, rule=f'allostery_rho', show_xlabel=False)

# ax3 = fig.add_subplot(gs[1, 1])
# plotting.plot_weights(ax3, G, training_steps=training_steps, rule=f'allostery_length')

# ax4 = fig.add_subplot(gs[1, 2])
# plotting.plot_weights(ax4, G, training_steps=training_steps, rule=f'allostery_rbrt', show_xlabel=False)

# ax5 = fig.add_subplot(gs[0, 2:3])
# plotting.plot_weights(ax5, G, training_steps=training_steps, rule=f'allostery_pressure', show_xlabel=False)

# ax6 = fig.add_subplot(gs[0, 2:4])
# plotting.plot_potential_drops_each_node(ax6, G_pressure)

# Adjust layout and save the figure
fig.tight_layout()
fig.savefig(f"../paper/plots/voltage_divider/weights_others.pdf")

# --------- TRAIN NETWORK WITH DIFFERENT TARGETS ---------

# G_target1 = networks.voltage_divider(save_data=True, voltage_desired=[1]) 

# fig, ax = plt.subplots(figsize = par.figsize_1)

# training.train(G_target1, training_steps=training_steps, weight_type='pressure', delta_weight = 1e-3, learning_rate=100)
# final_resistances = plotting.plot_potential_drops_each_node(ax, G_target1)

# G_target1.nodes[1]['desired'] = 3
# for index, edge in enumerate(G.edges()):
#     resistance = final_resistances[-1][index]
#     G_target1.edges[edge]['conductance'] = resistance
# training.train(G_target1, training_steps=training_steps, weight_type='pressure', delta_weight = 1e-3, learning_rate=100)
# final_resistances = plotting.plot_potential_drops_each_node(ax, G_target1, factor_time=2)

# G_target1.nodes[1]['desired'] = 4
# for index, edge in enumerate(G.edges()):
#     resistance = final_resistances[-1][index]
#     G_target1.edges[edge]['conductance'] = resistance
# training.train(G_target1, training_steps=training_steps, weight_type='pressure', delta_weight = 1e-3, learning_rate=100)
# final_resistances = plotting.plot_potential_drops_each_node(ax, G_target1, factor_time=3)

# # ax.legend()
# fig.tight_layout()
# fig.savefig(f"../paper/plots/voltage_divider/evolution_targets.pdf")

fig, ax = plt.subplots(figsize = par.figsize_1)
import numpy as np
targets = [1, 3, 4]
voltage_during_training = []
for targets_number in range(3):
    
    voltage_data = np.loadtxt(f"{par.DATA_PATH}potential_drops{targets[targets_number]}.txt", unpack=True)
    for steps in range(training_steps):
        print((voltage_data[0][steps]-voltage_data[1][steps]-5)/2)
        voltage_during_training.append((voltage_data[0][steps]-voltage_data[1][steps]-5)/2)
        
print(range(0,len(voltage_during_training)))
print(len(voltage_during_training))
ax.plot(range(0,len(voltage_during_training)), voltage_during_training)

fig.tight_layout()
fig.savefig(f"../paper/plots/voltage_divider/evolution_targets.pdf")

end = time.time()
print("Running time = ", end-start, "seconds")