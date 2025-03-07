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
# G = networks.three_inout(save_data=True) 
# G = networks.random_graph(save_data=True) 
# G = networks.voltage_divider(save_data=True) 
# G = nx.read_graphml(f'{par.DATA_PATH}random_graph_working.graphml')
G = nx.read_graphml(f'{par.DATA_PATH}random_rigthbases.graphml')

# a = networks.compute_regression_coefficients(G)
# print('Coefficient a:', a)

# G.nodes['3']['type']  = 'source'
# G.nodes['3']['color'] = par.color_dots[0]
# G.nodes['3']['constant_source']  = True
# G.nodes['3']['voltage']  = 1

# G.add_edge('6','2')
# G.add_edge('6','4')
# G.add_edge('4','5')
# G.add_edge('6','7')

# G.remove_node('7')
# G.remove_node('9')

# mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(G.nodes()))}

# Apply the relabeling
# G = nx.relabel_nodes(G, mapping)

# networks.initialize_edges(G)
# for node in G.nodes():
#     G.nodes[node]['rho'] = 0.5

# nx.write_graphml(G, f"{par.DATA_PATH}random_rigthbases.graphml")

# for edge in G.edges():
#     print(edge)

# G.remove_edge(1,2)
# G.remove_node('1')

# -> PLOT graph in /plots 
fig, ax = plt.subplots()
# pos = plotting.plot_graph('three_inout')
pos = plotting.plot_graph(G)
fig.tight_layout()
fig.savefig(f"../paper/plots/regression/graph.pdf", transparent=True)

training_steps = 100
training_type = 'regression'

# data = np.loadtxt(f"{par.DATA_PATH}weights/regression/rho/rho266.txt", unpack=True)
# weight_vec = data[1]
# print(weight_vec)

# for index, edge in enumerate(G.edges):
#     G.edges[edge][f'rho'] = weight_vec[index]

# for index, node in enumerate(G.nodes):
#     G.nodes[node][f'rho'] = weight_vec[index]

# G_ml = G.copy(as_view=False)  

# training.train(G, training_type=training_type, training_steps=training_steps, weight_type='resistance', delta_weight = 1e-3, learning_rate=100)
# training.train(G, training_type=training_type, training_steps=training_steps, weight_type='length', delta_weight = 1e-3, learning_rate=2e-5)
# training.train(G, training_type=training_type, training_steps=training_steps, weight_type='pressure', delta_weight = 1e-3, learning_rate=1e2)
 
training.train(G, training_type=training_type, training_steps=training_steps, weight_type='rho', delta_weight = 1e-4, learning_rate=5e-3)



fig, ax = plt.subplots()
# plotting.plot_weights(ax, G, training_steps=training_steps, rule=f'regression_length', show_xlabel=False)
# plotting.plot_weights(ax, G, training_steps=training_steps, rule=f'{training_type}_resistance', show_xlabel=False)
# plotting.plot_weights(ax, G, training_steps=training_steps, rule=f'{training_type}_pressure', show_xlabel=False)
plotting.plot_weights(ax, G, training_steps=training_steps, rule=f'{training_type}_rho', show_xlabel=False)
ax.legend()
fig.tight_layout()
fig.savefig(f"../paper/plots/regression/weights.pdf", transparent=True)

fig, ax = plt.subplots()
# plotting.plot_mse(ax, fig, f'resistance')
# plotting.plot_mse(ax, fig, f'pressure')
plotting.plot_mse(ax, fig, f'rho')
# plotting.plot_mse(ax, fig, f'allostery_length')
ax.legend()
fig.tight_layout()
fig.savefig(f"../paper/plots/regression/mse.pdf", transparent=True)


# training.test_regression(G, step=0, weight_type='length')
# training.test_regression(G, step=int(training_steps/2), weight_type='length')
# training.test_regression(G, step=training_steps, weight_type='length')

# fig, ax = plt.subplots(1, 3, figsize=(15,5))
# plotting.plot_regression(ax[0], step=0)
# plotting.plot_regression(ax[1], step=int(training_steps/2))
# plotting.plot_regression(ax[2], step=training_steps)
# fig.tight_layout()
# fig.savefig(f"../paper/plots/regression/snapshots.pdf", transparent=True)

# fig, ax = plt.subplots()
# pos = plotting.plot_graph(G, weight_type = 'length')
# fig.tight_layout()
# fig.savefig(f"../paper/plots/regression/graph.pdf", transparent=True)


end = time.time()
print("Running time = ", end-start, "seconds")


