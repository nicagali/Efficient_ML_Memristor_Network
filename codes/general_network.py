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
fig.savefig(f"../plots/general_network/graph.pdf")

# --------- TRAIN NETWORK WITH DIFFERENT WEIGHTS ---------

training_steps = 100

G_ml = G.copy(as_view=False)  
# training.train(G_ml, training_steps=training_steps, weight_type='length', delta_weight = 1e-3, learning_rate=1e-6)
G_ml = G.copy(as_view=False)  
# training.train(G_ml, training_steps=training_steps, weight_type='rbrt', delta_weight = 1e-3, learning_rate=0.5)
G_ml = G.copy(as_view=False)  
# training.train(G_ml, training_steps=training_steps, weight_type='rho', delta_weight = 1e-4, learning_rate=1e-3)
G_pressure = G.copy(as_view=False)  
training.train(G_pressure, training_steps=training_steps, weight_type='pressure', delta_weight = 1e-3, learning_rate=1e2)

# --------- PLOT ERROR AND WEIGHTS ---------

fig, ax = plt.subplots()
plotting.plot_mse(ax, fig, f'allostery_pressure')
plotting.plot_mse(ax, fig, f'allostery_length')
plotting.plot_mse(ax, fig, f'allostery_rbrt')
plotting.plot_mse(ax, fig, f'allostery_rho')
ax.legend()
fig.tight_layout()
fig.savefig(f"../paper/plots/general_network/mse_random.pdf")

# fig, ax = plt.subplots()
# plotting.plot_weights(ax, G, training_steps=training_steps, rule = f'allostery_pressure')
# fig.tight_layout()
# fig.savefig(f"../paper/plots/general_network/weights_random2.pdf")

end = time.time()
print("Running time = ", end-start, "seconds")