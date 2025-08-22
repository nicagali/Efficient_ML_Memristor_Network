import networks
import training
import plotting
import parameters as par
import matplotlib.pyplot as plt
import networkx as nx
import time
import matplotlib.gridspec as gridspec

start = time.time()
graph_id = 'G00010001'
DATA_PATH = f'{par.DATA_PATH}iris{graph_id}/'
PLOT_PATH = f'{par.PLOT_PATH}iris{graph_id}/'

# --------- INITIALIZE NETWORK ---------

# -> DEFINE graph from networks module

# G = networks.random_graph(number_nodes=12, number_edges=22 ,save_data=True) 
G = nx.read_graphml(f'{DATA_PATH}{graph_id}.graphml')
# G = networks.to_directed_graph(G, shuffle=True)

# G.graph['name'] = f'{graph_id}'
# G.remove_edge('9', '12')
# G.remove_edge('6', '7')
# G.add_edge('6','10')
# G.add_edge('7','8')
# networks.initialize_edges(G)
# nx.write_graphml(G, f'{DATA_PATH}{graph_id}.graphml')

# -> PLOT graph in /plots 
fig, ax = plt.subplots()
pos = plotting.plot_graph(G)
fig.tight_layout()
fig.savefig(f"{PLOT_PATH}graph.pdf", transparent=True)

# --------- TRAIN NETWORK WITH DIFFERENT WEIGHTS ---------

training_steps = 1000
training_type = 'iris'
weight_type_vec = ['length', 'radius_base', 'rho', 'pressure', 'length_radius_base', 'length_pressure', 'best_choice']
delta_weight_vec = [1e-3, 1, 1e-4, 1e-3, [1e-3, 1], [1e-3, 1e-3], [1e-3, 1, 1e-4, 1e-3]] 
learning_rate_vec = [5e-6, 8e-7, 1e-4, 20, [1e-6, 8e-7], [1e-6, 20], [1e-6, 8e-7, 1e-4, 20]]

for weight_type_index in [0]:
    G_train = G.copy(as_view=False)

    training.train(G_train, training_type=training_type, training_steps=training_steps, weight_type=weight_type_vec[weight_type_index], delta_weight = delta_weight_vec[weight_type_index], learning_rate=learning_rate_vec[weight_type_index], save_final_graph=True, write_weights=True)

# --------- PLOT ERROR AND WEIGHTS ---------

fig, ax = plt.subplots()
# plotting.plot_mse(ax, fig, graph_id, training_type, f'best_choice')
plotting.plot_mse(ax, fig, graph_id, training_type, f'length')
# plotting.plot_mse(ax, fig, graph_id, training_type, f'radius_base')
# plotting.plot_mse(ax, fig, graph_id, training_type, f'length_radius_base')
# plotting.plot_mse(ax, fig, graph_id, training_type, f'rho')
# plotting.plot_mse(ax, fig, graph_id, training_type, f'pressure')
# plotting.plot_mse(ax, fig, graph_id, training_type, f'length_pressure')
ax.legend(fontsize = par.legend_size)
fig.tight_layout()
fig.savefig(f"{PLOT_PATH}mse.pdf", transparent=True)



end = time.time()
print("Running time = ", end-start, "seconds")
