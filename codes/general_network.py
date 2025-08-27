import networks
import training
import plotting
import parameters as par
import matplotlib.pyplot as plt
import networkx as nx
import time
import matplotlib.gridspec as gridspec

start = time.time()
graph_id = 'G00030001'
DATA_PATH = f'{par.DATA_PATH}allostery{graph_id}/'
PLOT_PATH = f'{par.PLOT_PATH}allostery{graph_id}/'

# --------- INITIALIZE NETWORK ---------

# -> DEFINE graph from networks module
# G = networks.random_graph(save_data=True) 
# G = nx.read_graphml(f'{par.DATA_PATH}random_graph_working.graphml')
# nx.write_graphml(G, f'{DATA_PATH}random_graph.graphml')

# G = nx.DiGraph(G)

# G = nx.read_graphml(f'{DATA_PATH}{graph_id}_original.graphml')
G = nx.read_graphml(f'{DATA_PATH}{graph_id}shuffled.graphml')
# G.graph['name'] = f'{graph_id}'

G.remove_edge('5', '4')
G.add_edge('3','0')
networks.initialize_edges(G)
# print(G.edges())
# nx.write_graphml(G, f'{DATA_PATH}{graph_id}.graphml')
# G = networks.to_directed_graph(G, shuffle=True)
G.graph['name'] = f'{graph_id}'

# nx.write_graphml(G, f'{DATA_PATH}{graph_id}.graphml')

# -> PLOT graph in /plots 
fig, ax = plt.subplots()
pos = plotting.plot_graph(G)
fig.tight_layout()
fig.savefig(f"{PLOT_PATH}graph.pdf", transparent=True)

# --------- TRAIN NETWORK WITH DIFFERENT WEIGHTS ---------

training_steps = 400
training_type = 'allostery'
weight_type_vec = ['length', 'radius_base', 'rho', 'pressure', 'length_radius_base', 'length_pressure', 'best_choice']
delta_weight_vec = [1e-3, 1, 1e-4, 1e-3, [1e-3, 1], [1e-3, 1e-3], [1e-3, 1, 1e-4, 1e-3]] 
learning_rate_vec = [1e-6, 8e-7, 1e-4, 20, [1e-6, 8e-7], [1e-6, 20], [1e-6, 8e-7, 1e-4, 20]]

for weight_type_index in [0]:
# weight_type_index = 4
    G_train = G.copy(as_view=False)
    # print(G_train.nodes)

    # training.train(G_train, training_type=training_type, training_steps=training_steps, weight_type=weight_type_vec[weight_type_index], delta_weight = delta_weight_vec[weight_type_index], learning_rate=learning_rate_vec[weight_type_index], save_final_graph=True, write_weights=True)

# --------- PLOT ERROR AND WEIGHTS ---------

fig, ax = plt.subplots()
plotting.plot_mse(ax, fig, graph_id, training_type, f'length')
plotting.plot_mse(ax, fig, graph_id, training_type, f'radius_base')
# plotting.plot_mse(ax, fig, graph_id, training_type, f'length_radius_base')
plotting.plot_mse(ax, fig, graph_id, training_type, f'rho')
plotting.plot_mse(ax, fig, graph_id, training_type, f'pressure')
plotting.plot_mse(ax, fig, graph_id, training_type, f'best_choice')
# plotting.plot_mse(ax, fig, graph_id, training_type, f'length_pressure')
ax.legend(fontsize = par.legend_size)
fig.tight_layout()
fig.savefig(f"{PLOT_PATH}mse.pdf", transparent=True)



end = time.time()
print("Running time = ", end-start, "seconds")
