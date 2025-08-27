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

graph_id = 'G00010001'
DATA_PATH = f'{par.DATA_PATH}regression{graph_id}/'

# --------- INITIALIZE NETWORK ---------

# -> DEFINE graph from networks module
# G = networks.random_graph(number_edges=20, number_nodes=10, save_data=True) 
# G = nx.read_graphml(f'{par.DATA_PATH}random_graph.graphml')
G = nx.read_graphml(f'{DATA_PATH}{graph_id}.graphml')

# Using graphs:
# regression_working: wokring nw for length not directed with 3V extra source
# regression_working_rho: wokring nw for length directed with 6V extra source, working for rho training

# print(G.edges())
# G.add_edge('1','7')
# networks.initialize_edges(G)
# G = networks.to_directed_graph(G, shuffle=True)
G.graph['name'] = graph_id
# print(G.graph['name'])
nx.write_graphml(G, f'{DATA_PATH}{graph_id}.graphml')
# print(G.edges())

# print(G.nodes['3']['voltage'])

# G.nodes['3']['voltage'] = 3
# --------- PLOT GRAPH ---------

# -> PLOT graph in /plots 
fig, ax = plt.subplots()
pos = plotting.plot_graph(G)
fig.tight_layout()
fig.savefig(f"{DATA_PATH}graph.pdf", transparent=True)

# --------- TRAIN NETWORK ---------
training_steps = 30   # choose
training_type = 'regression'    # choose

weight_type_vec = ['length', 'radius_base', 'rho', 'pressure', 'resistance', 'best_choice']
delta_weight_vec = [1e-3, 1e-3, 1e-4, 1e-3, 1e-3, [1e-3, 1e-3, 1e-4, 1e-3]]
# learning_rate_vec = [2e-7, 1e-6, 1e-4, 2e2, 1e3] #1
# learning_rate_vec = [8e-7, 1e-6, 9e-4, 2e2, 1e4] #2
learning_rate_vec = [5e-7, 3e-6, 2e-4, 5e1, 1e4, [2e-9, 1e-6, 1e-4, 2e2]]
constant_source = [11, 4, 4, 11, 4, [11, 4, 4, 11]]

weight_type_index = 0   # choose

for weight_type_index in [5]:
    
    G = nx.read_graphml(f'{DATA_PATH}{graph_id}.graphml')
    # G.nodes['3']['voltage'] = constant_source[weight_type_index]
    G.nodes['3']['voltage'] = constant_source[0]

    training.train(G, training_type=training_type, training_steps=training_steps, weight_type=weight_type_vec[weight_type_index], delta_weight = delta_weight_vec[weight_type_index], learning_rate=learning_rate_vec[weight_type_index], save_final_graph=True, write_weights=True, constant_source=constant_source[weight_type_index])

#     # --------- PLOT ERROR, WEIGHTS & RESISTANCE ---------

    # fig, ax = plt.subplots()
    # plotting.plot_weights(ax, G, training_steps=training_steps, training_type=training_type, weight_type = weight_type_vec[weight_type_index], show_xlabel=False, starting_step=0)
    # fig.tight_layout()
    # fig.savefig(f"{DATA_PATH}weights.pdf", transparent=True)

    fig, ax = plt.subplots()
    plotting.plot_mse(ax, fig, graph_id, training_type, f'length')
    plotting.plot_mse(ax, fig, graph_id, training_type, f'radius_base')
    plotting.plot_mse(ax, fig, graph_id, training_type, f'rho')
    plotting.plot_mse(ax, fig, graph_id, training_type, f'pressure')
    plotting.plot_mse(ax, fig, graph_id, training_type, f'best_choice')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{DATA_PATH}mse.pdf", transparent=True)

#     # --------- TEST REGRESSION AND PLOT RESULT ---------
    training_steps = 391
    training.test_regression(G, step=0, weight_type=f'{weight_type_vec[weight_type_index]}')
    training.test_regression(G, step=int(training_steps/2), weight_type=f'{weight_type_vec[weight_type_index]}')
    training.test_regression(G, step=training_steps, weight_type=f'{weight_type_vec[weight_type_index]}')

    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    plotting.plot_regression(ax[0], graph_id, weight_type_vec[weight_type_index], step=0)
    plotting.plot_regression(ax[1], graph_id, weight_type_vec[weight_type_index], step=int(training_steps/2))
    # plotting.plot_regression(ax[1], graph_id, weight_type_vec[weight_type_index], step=50)
    plotting.plot_regression(ax[2], graph_id, weight_type_vec[weight_type_index], step=training_steps)
    fig.tight_layout()
    fig.savefig(f"{DATA_PATH}snapshots_{weight_type_vec[weight_type_index]}.pdf", transparent=True)

#     # --------- PLOT RESISTANCES OF MEMRISTORS DURING TRAINING ---------

#     G_trained = nx.read_graphml(f'{DATA_PATH}trained_graph_{weight_type_vec[weight_type_index]}.graphml')

#     fig, ax = plt.subplots(figsize = par.figsize_1)
#     plotting.plot_memristor_resistances(ax, G)
#     fig.tight_layout()
#     fig.savefig(f"{DATA_PATH}memristors_resistances_{weight_type_vec[weight_type_index]}.pdf")

end = time.time()
print("Running time = ", end-start, "seconds")
 

