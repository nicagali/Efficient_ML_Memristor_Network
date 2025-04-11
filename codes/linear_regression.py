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

graph_id = 'G00010002'
DATA_PATH = f'{par.DATA_PATH}regression{graph_id}/'

# --------- INITIALIZE NETWORK ---------

# -> DEFINE graph from networks module
# G = networks.random_graph(save_data=True) 
# G = nx.read_graphml(f'{par.DATA_PATH}random_graph.graphml')

# Using graphs:
# regression_working: wokring nw for length not directed with 3V extra source
# regression_working_rho: wokring nw for length directed with 6V extra source, working for rho training

G = nx.read_graphml(f'{DATA_PATH}{graph_id}.graphml')
print(G.edges())
# G.add_edge('1','7')
# networks.initialize_edges(G)
# G = networks.to_directed_graph(G, shuffle=True)
# G.graph['name'] = graph_id
# nx.write_graphml(G, f'{DATA_PATH}G00010003.graphml')

# print(G.nodes['3']['voltage'])

# G.nodes['3']['voltage'] = 3
# --------- PLOT GRAPH ---------

# -> PLOT graph in /plots 
fig, ax = plt.subplots()
pos = plotting.plot_graph(G)
fig.tight_layout()
fig.savefig(f"{DATA_PATH}graph.pdf", transparent=True)

# --------- TRAIN NETWORK ---------
training_steps = 400   # choose
training_type = 'regression'    # choose

weight_type_vec = ['length', 'radius_base', 'rho', 'pressure', 'resistance']
delta_weight_vec = [1e-3, 1e-3, 1e-4, 1e-3, 1e-3]
learning_rate_vec = [5e-7, 1e-6, 7e-4, 1e2, 1e3]
constant_source = [11, 3, 3, 3]

weight_type_index = 0   # choose

for weight_type_index in [0]:
    # print(f'{DATA_PATH}')
    
    # G = nx.read_graphml(f'{DATA_PATH}{graph_id}.graphml')
    G = nx.read_graphml(f'{DATA_PATH}G00010002.graphml')
    # G = nx.read_graphml(f'{DATA_PATH}G00010001.graphml')
    # G = nx.read_graphml(f'{DATA_PATH}G00010003.graphml')
    G.nodes['3']['voltage'] = constant_source[weight_type_index]

    training.train(G, training_type=training_type, training_steps=training_steps, weight_type=weight_type_vec[weight_type_index], delta_weight = delta_weight_vec[weight_type_index], learning_rate=learning_rate_vec[weight_type_index], save_final_graph=True, write_weights=True)



    # --------- PLOT ERROR, WEIGHTS & RESISTANCE ---------

    fig, ax = plt.subplots()
    plotting.plot_weights(ax, G, training_steps=training_steps, training_type=training_type, weight_type = weight_type_vec[weight_type_index], show_xlabel=False, starting_step=0)
    fig.tight_layout()
    fig.savefig(f"{DATA_PATH}weights.pdf", transparent=True)

    fig, ax = plt.subplots()
    plotting.plot_mse(ax, fig, graph_id, training_type, f'length')
    plotting.plot_mse(ax, fig, graph_id, training_type, f'radius_base')
    plotting.plot_mse(ax, fig, graph_id, training_type, f'rho')
    plotting.plot_mse(ax, fig, graph_id, training_type, f'pressure')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{DATA_PATH}mse.pdf", transparent=True)

    # --------- TEST REGRESSION AND PLOT RESULT ---------

    training.test_regression(G, step=0, weight_type=f'{weight_type_vec[weight_type_index]}')
    training.test_regression(G, step=int(training_steps/2), weight_type=f'{weight_type_vec[weight_type_index]}')
    # training.test_regression(G, step=50, weight_type=f'{weight_type_vec[weight_type_index]}')
    training.test_regression(G, step=training_steps, weight_type=f'{weight_type_vec[weight_type_index]}')

    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    plotting.plot_regression(ax[0], graph_id, weight_type_vec[weight_type_index], step=0)
    plotting.plot_regression(ax[1], graph_id, weight_type_vec[weight_type_index], step=int(training_steps/2))
    # plotting.plot_regression(ax[1], graph_id, weight_type_vec[weight_type_index], step=50)
    plotting.plot_regression(ax[2], graph_id, weight_type_vec[weight_type_index], step=training_steps)
    fig.tight_layout()
    fig.savefig(f"{DATA_PATH}snapshots_{weight_type_vec[weight_type_index]}.pdf", transparent=True)

    # --------- PLOT TRAINED GRAPH  ---------
    # The width of the edges are indicative of their resistance for edge-training, size of dots for node-training

    # fig, ax = plt.subplots()
    # pos = plotting.plot_graph(G, weight_type = 'length')
    # fig.tight_layout()
    # fig.savefig(f"../paper/plots/regression/graph.pdf", transparent=True)

    # --------- PLOT RESISTANCES OF MEMRISTORS DURING TRAINING ---------

    G_trained = nx.read_graphml(f'{DATA_PATH}trained_graph_{weight_type_vec[weight_type_index]}.graphml')

    fig, ax = plt.subplots(figsize = par.figsize_1)
    plotting.plot_memristor_resistances(ax, G)
    fig.tight_layout()
    fig.savefig(f"{DATA_PATH}memristors_resistances_{weight_type_vec[weight_type_index]}.pdf")

end = time.time()
print("Running time = ", end-start, "seconds")
 

