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
G = networks.random_graph(save_data=True) 
# G = nx.read_graphml(f'{par.DATA_PATH}random_graph.graphml')

# -> PLOT graph in /plots 
fig, ax = plt.subplots()
pos = plotting.plot_graph('random_graph')
fig.tight_layout()
fig.savefig(f"../paper/plots/linear_regression/graph.pdf")

training_steps = 20
training_type = 'linear_regression'

G_ml = G.copy(as_view=False)  
# training.train(G_ml, training_type=training_type, training_steps=training_steps, weight_type='length', delta_weight = 1e-3, learning_rate=5e-6)
G_pressure = G.copy(as_view=False)  
training.train(G_pressure, training_type=training_type, training_steps=training_steps, weight_type='pressure', delta_weight = 1e-3, learning_rate=2e3)


training.test_regression(G, step=0, weight_type='pressure')
training.test_regression(G, step=10, weight_type='pressure')
training.test_regression(G, step=19, weight_type='pressure')

fig, ax = plt.subplots(1, 3, figsize=(15,5))
plotting.plot_regression(ax[0], step=2)
plotting.plot_regression(ax[1], step=5)
plotting.plot_regression(ax[2], step=10)
fig.tight_layout()
fig.savefig(f"../paper/plots/linear_regression/snapshots.pdf")



