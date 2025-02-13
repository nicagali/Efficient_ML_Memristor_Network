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
G = networks.random_graph(save_data=True) 
# G = nx.read_graphml(f'{par.DATA_PATH}random_graph.graphml')
 
# -> PLOT graph in /plots 
fig, ax = plt.subplots()
pos = plotting.plot_graph('random_graph')
fig.tight_layout()
fig.savefig(f"../paper/plots/linear_regression/graph.pdf")

training_steps = 20
training_type = 'linear_regression'

# data = np.loadtxt(f"{par.DATA_PATH}weights/linear_regression/length/length100.txt", unpack=True)
# weight_vec = data[1]

# for index, edge in enumerate(G.edges):
#         G.edges[edge][f'length'] = weight_vec[index]

G_ml = G.copy(as_view=False)  
training.train(G, training_type=training_type, training_steps=training_steps, weight_type='length', delta_weight = 1e-3, learning_rate=5e-6)
# training.train(G, training_type=training_type, training_steps=training_steps, weight_type='resistance', delta_weight = 1e-1, learning_rate=800)
# G_pressure = G.copy(as_view=False)  
# training.train(G_pressure, training_type=training_type, training_steps=training_steps, weight_type='pressure', delta_weight = 1e-3, learning_rate=1e4)


fig, ax = plt.subplots()
# plotting.plot_mse(ax, fig, f'allostery_resistance')
plotting.plot_mse(ax, fig, f'allostery_length')
ax.legend()
fig.tight_layout()
fig.savefig(f"../paper/plots/linear_regression/mse.pdf", transparent=True)


training.test_regression(G, step=0, weight_type='length')
training.test_regression(G, step=10, weight_type='length')
training.test_regression(G, step=20, weight_type='length')

fig, ax = plt.subplots(1, 3, figsize=(15,5))
plotting.plot_regression(ax[0], step=0)
plotting.plot_regression(ax[1], step=10)
plotting.plot_regression(ax[2], step=20)
fig.tight_layout()
fig.savefig(f"../paper/plots/linear_regression/snapshots.pdf")



