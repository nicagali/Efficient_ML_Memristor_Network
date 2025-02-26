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
G = nx.read_graphml(f'{par.DATA_PATH}random_graph.graphml')


# -> PLOT graph in /plots 
fig, ax = plt.subplots()
# pos = plotting.plot_graph('three_inout')
pos = plotting.plot_graph('random_graph')
fig.tight_layout()
fig.savefig(f"../paper/plots/regression/graph.pdf", transparent=True)

training_steps = 100
training_type = 'regression'

# data = np.loadtxt(f"{par.DATA_PATH}weights/regression/resistance/resistance1.txt", unpack=True)
# weight_vec = data[1]

# for index, edge in enumerate(G.edges)x:
#         G.edges[edge][f'resistance'] = weight_vec[index]

G_ml = G.copy(as_view=False)  

# training.train(G, training_type=training_type, training_steps=training_steps, weight_type='resistance', delta_weight = 1e-3, learning_rate=200)



fig, ax = plt.subplots()
# plotting.plot_weights(ax, G, training_steps=training_steps, rule=f'regression_length', show_xlabel=False)
plotting.plot_weights(ax, G, training_steps=training_steps, rule=f'{training_type}_resistance', show_xlabel=False)
ax.legend()
fig.tight_layout()
fig.savefig(f"../paper/plots/regression/weights.pdf", transparent=True)

fig, ax = plt.subplots()
plotting.plot_mse(ax, fig, f'resistance')
# plotting.plot_mse(ax, fig, f'allostery_length')
ax.legend()
fig.tight_layout()
fig.savefig(f"../paper/plots/regression/mse.pdf", transparent=True)


training.test_regression(G, step=0, weight_type='resistance')
training.test_regression(G, step=int(training_steps/2), weight_type='resistance')
training.test_regression(G, step=training_steps, weight_type='resistance')

fig, ax = plt.subplots(1, 3, figsize=(15,5))
plotting.plot_regression(ax[0], step=0)
plotting.plot_regression(ax[1], step=int(training_steps/2))
plotting.plot_regression(ax[2], step=training_steps)
fig.tight_layout()
fig.savefig(f"../paper/plots/regression/snapshots.pdf")

# fig, ax = plt.subplots()
# # pos = plotting.plot_graph('three_inout')
# pos = plotting.plot_graph('random_graph', step = training_steps)
# fig.tight_layout()
# fig.savefig(f"../paper/plots/regression/graph.pdf", transparent=True)

end = time.time()
print("Running time = ", end-start, "seconds")


