import networks
import training
import plotting
import parameters as par
import matplotlib.pyplot as plt
import networkx as nx
import time
import matplotlib.gridspec as gridspec

start = time.time()
graph_id = 'G00040001'
DATA_PATH = f'{par.DATA_PATH}allostery{graph_id}/'

# --------- INITIALIZE NETWORK ---------

# -> DEFINE graph from networks module
# G = networks.random_graph(save_data=True) 
# G = nx.read_graphml(f'{par.DATA_PATH}random_graph.graphml')

G = nx.read_graphml(f'{DATA_PATH}{graph_id}.graphml')

G.add_edge('2', '')

nx.write_graphml(G, f'{DATA_PATH}{graph_id}.graph_ml')

# -> PLOT graph in /plots 
fig, ax = plt.subplots()
pos = plotting.plot_graph(G)
fig.tight_layout()
fig.savefig(f"{DATA_PATH}graph.pdf", transparent=True)

# --------- TRAIN NETWORK WITH DIFFERENT WEIGHTS ---------

training_steps = 50
training_type = 'allostery'
weight_type_vec = ['length', 'radius_base', 'rho', 'pressure']
delta_weight_vec = [1e-3, 1, 1e-4, 1e-3]
learning_rate_vec = [1e-5, 1e-6, 1e-4, 1e2]

for weight_type_index in [0,1,2,3]:
    
    G = nx.read_graphml(f'{DATA_PATH}{graph_id}.graphml')

    # training.train(G, training_type=training_type, training_steps=training_steps, weight_type=weight_type_vec[weight_type_index], delta_weight = delta_weight_vec[weight_type_index], learning_rate=learning_rate_vec[weight_type_index], save_final_graph=True, write_weights=True)

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
