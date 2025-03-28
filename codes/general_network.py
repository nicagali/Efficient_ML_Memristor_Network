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
# G = nx.read_graphml(f'{par.DATA_PATH}random_graph.graphml')
G = nx.read_graphml(f'{par.DATA_PATH}random_graph_verynice.graphml')
# G = nx.read_graphml(f'{par.DATA_PATH}random_graph_small_work.graphml')

G = nx.reverse(G, copy=True)

# voltage_input = [0, 5, 2] # node initialized here because different for differnent nw
# voltage_desired = [3, 4]

# networks.initialize_nodes(G, voltage_input, voltage_desired)
networks.initialize_edges(G)

# -> PLOT graph in /plots 
fig, ax = plt.subplots()
pos = plotting.plot_graph(G)
fig.tight_layout()
fig.savefig(f"../plots/general_network/graph.pdf")

# --------- TRAIN NETWORK WITH DIFFERENT WEIGHTS ---------

training_steps = 30
training_type = 'allostery'
weight_type_vec = ['length', 'radius_base', 'rho', 'pressure']
delta_weight_vec = [1e-3, 1, 1e-4, 1e-3]
learning_rate_vec_random2 = [1e-5, 1e-6, 1e-4, 1e2]
learning_rate_vec_random3 = [5e-6, 1e-6, 1e-3, 1e2]

# G_ml = G.copy(as_view=False)  
# training.train(G, training_type=training_type, training_steps=training_steps, weight_type='length', delta_weight = 1e-3, learning_rate=1e-5)
# G_ml = G.copy(as_view=False)  
# training.train(G_ml, training_type=training_type, training_steps=training_steps, weight_type='radius_base', delta_weight = 1e-3, learning_rate=1e-6)
# G_ml = G.copy(as_vie
# w=False)  
# training.train(G_ml, training_type=training_type, training_steps=training_steps, weight_type='rho', delta_weight = 1e-4, learning_rate=5e-3)
# G_ml = G.copy(as_view=False)  
training.train(G, training_type=training_type, training_steps=training_steps, weight_type='pressure', delta_weight = 1e-3, learning_rate=1e2)

# --------- PLOT ERROR AND WEIGHTS ---------

fig, ax = plt.subplots()
plotting.plot_mse(ax, fig, f'pressure')
plotting.plot_mse(ax, fig, f'length')
# plotting.plot_mse(ax, fig, f'radius_base')
# plotting.plot_mse(ax, fig, f'rho')
ax.legend()
fig.tight_layout()
fig.savefig(f"../paper/plots/general_network/mse_random.pdf", transparent=True)


end = time.time()
print("Running time = ", end-start, "seconds")
