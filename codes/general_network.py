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

# --------- INITIALIZE NETWORK ---------

# -> DEFINE graph from networks module
# G = networks.random_graph(save_data=True) 
# G = nx.read_graphml(f'{par.DATA_PATH}random_graph.graphml')

G = nx.read_graphml(f'{DATA_PATH}{graph_id}_original.graphml')

# attributes = {"type" : "target", 'color' : par.color_dots[1]}
# G.add_node(7, **attributes)
# targets = [x for x in G.nodes() if G.nodes[x]['type']=='target']
# sources = [x for x in G.nodes() if G.nodes[x]['type']=='source']
# voltage_input = [0, 5] # node initialized here because different for differnent nw
# voltage_desired = [3, 4]
# networks.initialize_nodes(G, sources, targets, voltage_input, voltage_desired)

# G.add_edge('2', '7')
# G.add_edge('6', '7')
# networks.initialize_edges(G)

# graph_id = 'G00030002'

# G = networks.to_directed_graph(G, shuffle=True)
# G.graph['name'] = f'{graph_id}'
# nx.write_graphml(G, f'{DATA_PATH}{graph_id}.graphml')
G = nx.read_graphml(f'{DATA_PATH}{graph_id}.graphml')

# -> PLOT graph in /plots 
fig, ax = plt.subplots()
pos = plotting.plot_graph(G)
fig.tight_layout()
fig.savefig(f"{DATA_PATH}graph.pdf", transparent=True)

# --------- TRAIN NETWORK WITH DIFFERENT WEIGHTS ---------

training_steps = 200
training_type = 'allostery'
weight_type_vec = ['length', 'radius_base', 'rho', 'pressure']
delta_weight_vec = [1e-3, 1, 1e-4, 1e-3]
learning_rate_vec = [5e-6, 5e-6, 1e-3, 3e2]

for weight_type_index in [0,1,2,3]:
    
    G_train = G.copy(as_view=False)

    training.train(G_train, training_type=training_type, training_steps=training_steps, weight_type=weight_type_vec[weight_type_index], delta_weight = delta_weight_vec[weight_type_index], learning_rate=learning_rate_vec[weight_type_index], save_final_graph=True, write_weights=True)

# --------- PLOT ERROR AND WEIGHTS ---------

fig, ax = plt.subplots()
plotting.plot_mse(ax, fig, graph_id, training_type, f'length')
plotting.plot_mse(ax, fig, graph_id, training_type, f'radius_base')
plotting.plot_mse(ax, fig, graph_id, training_type, f'rho')
plotting.plot_mse(ax, fig, graph_id, training_type, f'pressure')
ax.legend()
fig.tight_layout()
fig.savefig(f"{DATA_PATH}mse.pdf", transparent=True)



end = time.time()
print("Running time = ", end-start, "seconds")
