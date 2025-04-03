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

# Using graphs:
# regression_working: wokring nw for length not directed with 3V extra source
# regression_working_rho: wokring nw for length directed with 6V extra source, working for rho training

# -> DEFINE graph from networks module
# G = networks.random_graph(save_data=True) 
# G = nx.read_graphml(f'{par.DATA_PATH}random_graph.graphml')
# G = nx.read_graphml(f'{par.DATA_PATH}random_graph_verynice.graphml')
G = nx.read_graphml(f'{par.DATA_PATH}random_graph_working.graphml')
# G = nx.read_graphml(f'{par.DATA_PATH}random_graph_small_work.graphml')
# G = nx.read_graphml(f'{par.DATA_PATH}random_rigthbases.graphml')



G = nx.read_graphml(f'{par.DATA_PATH}regression_working.graphml')
networks.initialize_edges(G)
# G = networks.to_directed_graph(G, shuffle=False)


# nx.write_graphml(G, f"{par.DATA_PATH}random_graph_shuffled.graphml")
# G = nx.read_graphml(f'{par.DATA_PATH}random_graph_shuffled.graphml')
# G = nx.read_graphml(f'{par.DATA_PATH}regression_working_rho.graphml')
G.nodes['3']['voltage']  = 6

# data = np.loadtxt(f"{par.DATA_PATH}weights/regression/radius_base/radius_base100.txt", unpack=True)
# data = data[1]
# for edge_index, edge in enumerate(G.edges()):
#     G.edges[edge]['radius_base'] = data[edge_index]

# -> PLOT graph in /plots 
fig, ax = plt.subplots()
pos = plotting.plot_graph(G)
fig.tight_layout()
fig.savefig(f"../paper/plots/regression/graph.pdf", transparent=True)

# --------- TRAIN NETWORK ---------
training_steps = 300    # choose
training_type = 'regression'    # choose


weight_type_vec = ['length', 'radius_base', 'rho', 'pressure', 'resistance']
delta_weight_vec = [1e-3, 1e-3, 1e-4, 1e-3, 1e-3]
learning_rate_vec = [1e-6, 1e-5, 1e-3, 3e2, 1e3]

weight_type_index = 0   # choose

training.train(G, training_type=training_type, training_steps=training_steps, weight_type=weight_type_vec[weight_type_index], delta_weight = delta_weight_vec[weight_type_index], learning_rate=learning_rate_vec[weight_type_index])

# --------- PLOT ERROR, WEIGHTS & RESISTANCE ---------

fig, ax = plt.subplots()
# plotting.plot_weights(ax, G, training_steps=training_steps, rule=f'{training_type}_length', show_xlabel=False, starting_step=0)
plotting.plot_weights(ax, G, training_steps=training_steps, rule=f'{training_type}_{weight_type_vec[weight_type_index]}', show_xlabel=False, starting_step=0)
# ax.legend()
fig.tight_layout()
fig.savefig(f"../paper/plots/regression/weights.pdf", transparent=True)

fig, ax = plt.subplots()
plotting.plot_mse(ax, fig, f'length')
plotting.plot_mse(ax, fig, f'radius_base')
# plotting.plot_mse(ax, fig, f'rho')
plotting.plot_mse(ax, fig, f'pressure')
ax.legend()
fig.tight_layout()
fig.savefig(f"../paper/plots/regression/mse.pdf", transparent=True)

# --------- TEST REGRESSION AND PLOT RESULT ---------

training.test_regression(G, step=0, weight_type=f'{weight_type_vec[weight_type_index]}')
training.test_regression(G, step=int(training_steps/2), weight_type=f'{weight_type_vec[weight_type_index]}')
training.test_regression(G, step=training_steps, weight_type=f'{weight_type_vec[weight_type_index]}')

fig, ax = plt.subplots(1, 3, figsize=(15,5))
plotting.plot_regression(ax[0], step=0)
plotting.plot_regression(ax[1], step=int(training_steps/2))
plotting.plot_regression(ax[2], step=training_steps)
fig.tight_layout()
fig.savefig(f"../paper/plots/regression/snapshots.pdf", transparent=True)

# --------- PLOT TRAINED GRAPH  ---------
# The width of the edges are indicative of their resistance for edge-training, size of dots for node-training

# fig, ax = plt.subplots()
# pos = plotting.plot_graph(G, weight_type = 'length')
# fig.tight_layout()
# fig.savefig(f"../paper/plots/regression/graph.pdf", transparent=True)

# --------- PLOT RESISTANCES OF MEMRISTORS DURING TRAINING ---------

fig, ax = plt.subplots(figsize = par.figsize_1)
plotting.plot_memristor_resistances(ax, G)
fig.tight_layout()
fig.savefig(f"../paper/plots/voltage_divider/memristors_resisatnces.pdf")

end = time.time()
print("Running time = ", end-start, "seconds")


