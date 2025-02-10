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
# G = networks.random_graph(save_data=True) 
G = nx.read_graphml(f'{par.DATA_PATH}random_graph.graphml')

# -> PLOT graph in /plots 
fig, ax = plt.subplots()
pos = plotting.plot_graph('random_graph')
fig.tight_layout()
fig.savefig(f"../plots/general_network/graph.pdf")

# --------- TRAIN NETWORK WITH DIFFERENT WEIGHTS ---------

training_steps = 50
training_type = 'allostery'
weight_type_vec = ['length', 'radius_base', 'rho', 'pressure']
delta_weight_vec = [1e-3, 1, 1e-4, 1e-3]
learning_rate_vec_random2 = [1e-5, 1e-6, 1e-4, 1e2]
learning_rate_vec_random3 = [5e-6, 1e-6, 1e-3, 1e2]

G_ml = G.copy(as_view=False)  
# training.train(G_ml, training_type=training_type, training_steps=training_steps, weight_type='length', delta_weight = 1e-3, learning_rate=1e-5)
# G_ml = G.copy(as_view=False)  
# training.train(G_ml, training_type=training_type, training_steps=training_steps, weight_type='radius_base', delta_weight = 1e-3, learning_rate=1)
# G_ml = G.copy(as_view=False)  
# training.train(G_ml, training_type=training_type, training_steps=training_steps, weight_type='rho', delta_weight = 1e-4, learning_rate=1e-3)
G_ml = G.copy(as_view=False)  
# training.train(G_ml, training_type=training_type, training_steps=training_steps, weight_type='pressure', delta_weight = 1e-3, learning_rate=3e1)

# --------- PLOT ERROR AND WEIGHTS ---------

# fig, ax = plt.subplots()
# plotting.plot_mse(ax, fig, f'allostery_pressure')
# plotting.plot_mse(ax, fig, f'allostery_length')
# plotting.plot_mse(ax, fig, f'allostery_radius_base')
# plotting.plot_mse(ax, fig, f'allostery_rho')
# ax.legend()
# fig.tight_layout()
# fig.savefig(f"../paper/plots/general_network/mse_random.pdf", transparent=True)

end = time.time()
print("Running time = ", end-start, "seconds")

# --------- PRE-TRAIN NETWORK WITH LENGTH/RADIUS ---------

training_steps = 50
# G_ml = G.copy(as_view=False)  
# training.train(G_ml, training_type=training_type, training_steps=training_steps, weight_type='pressure', delta_weight = 1e-3, learning_rate=2e2)
# fig, ax = plt.subplots()
# plotting.plot_mse(ax, fig, f'allostery_pressure')
# ax.legend()
# fig.tight_layout()
# fig.savefig(f"../paper/plots/general_network/mse_press.pdf", transparent=True)

G_length = G.copy(as_view=False)  
training.train(G_length, training_type=training_type, training_steps=training_steps, weight_type='length', delta_weight = 1e-3, learning_rate=5e-6)
training.train(G_length, training_type=training_type, training_steps=training_steps, weight_type='pressure', delta_weight = 1e-3, learning_rate=2e2)
# fig, ax = plt.subplots()
# plotting.plot_mse(ax, fig, f'allostery_pressure')
# ax.legend()
# fig.tight_layout()
# fig.savefig(f"../paper/plots/general_network/mse_pretrained.pdf", transparent=True)

