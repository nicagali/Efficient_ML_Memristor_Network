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
# G = networks.random_graph(save_data=True) 
G = nx.read_graphml(f'{par.DATA_PATH}random_graph.graphml')
# G = nx.read_graphml(f'{par.DATA_PATH}random_graph_working.graphml')
# G = nx.read_graphml(f'{par.DATA_PATH}random_rigthbases.graphml')

# -> PLOT graph in /plots 
fig, ax = plt.subplots()
pos = plotting.plot_graph(G)
fig.tight_layout()
fig.savefig(f"../paper/plots/regression/graph.pdf", transparent=True)

# --------- TRAIN NETWORK ---------

training_steps = 50    # choose
training_type = 'regression'    # choose

weight_type_vec = ['length', 'radius_base', 'rho', 'pressure', 'resistance']
delta_weight_vec = [1e-3, 1e-3, 1e-4, 1e-3, 1e-3]
learning_rate_vec = [2e-5, 1e-5, 8e-3, 100, 100]

weight_type_index = 0   # choose

# training.train(G, training_type=training_type, training_steps=training_steps, weight_type=weight_type_vec[weight_type_index], delta_weight = delta_weight_vec[weight_type_index], learning_rate=learning_rate_vec[weight_type_index])

# --------- PLOT ERROR, WEIGHTS & RESISTANCE ---------

fig, ax = plt.subplots()
plotting.plot_weights(ax, G, training_steps=training_steps, rule=f'{training_type}_{weight_type_vec[weight_type_index]}', show_xlabel=False, starting_step=0)
# ax.legend()
fig.tight_layout()
fig.savefig(f"../paper/plots/regression/weights.pdf", transparent=True)

fig, ax = plt.subplots()
plotting.plot_mse(ax, fig, f'{weight_type_vec[weight_type_index]}')
ax.legend()
fig.tight_layout()
fig.savefig(f"../paper/plots/regression/mse.pdf", transparent=True)

# --------- TEST REGRESSION AND PLOT RESULT ---------

# training.test_regression(G, step=0, weight_type='length')
# training.test_regression(G, step=int(training_steps/2), weight_type='length')
# training.test_regression(G, step=training_steps, weight_type='length')

# fig, ax = plt.subplots(1, 3, figsize=(15,5))
# plotting.plot_regression(ax[0], step=0)
# plotting.plot_regression(ax[1], step=int(training_steps/2))
# plotting.plot_regression(ax[2], step=training_steps)
# fig.tight_layout()
# fig.savefig(f"../paper/plots/regression/snapshots.pdf", transparent=True)

# --------- PLOT TRAINED GRAPH  ---------
# The width of the edges are indicative of their resistance for edge-training, size of dots for node-training

# fig, ax = plt.subplots()
# pos = plotting.plot_graph(G, weight_type = 'length')
# fig.tight_layout()
# fig.savefig(f"../paper/plots/regression/graph.pdf", transparent=True)

# --------- PLOT RESISTANCES OF MEMRISTORS DURING TRAINING ---------

# fig, ax = plt.subplots(figsize = par.figsize_1)
# plotting.plot_memristor_resistances(ax, G)
# fig.tight_layout()
# fig.savefig(f"../paper/plots/voltage_divider/memristors_resisatnces.pdf")

end = time.time()
print("Running time = ", end-start, "seconds")


