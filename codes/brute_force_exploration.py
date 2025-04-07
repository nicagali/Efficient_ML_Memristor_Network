import networks
import training
import plotting
import parameters as par
import matplotlib.pyplot as plt
import networkx as nx
import time
import matplotlib.gridspec as gridspec
import numpy as np
import sys

start = time.time()

new = int(sys.argv[1])
graph_id = str(sys.argv[2])

training_steps = 300   # choose
training_type = 'regression'

weight_type_vec = ['length', 'radius_base', 'rho', 'pressure']
weight_symbol_vec = ['L', 'Rb', 'rho', 'P']
delta_weight_vec = [1e-3, 1e-3, 1e-4, 1e-3]
learning_rate_vec = [1e-6, 5e-6, 1e-3, 3e2]

if new == 1:
    print("New graph generated")
    # -> DEFINE graph from networks module
    G = networks.random_graph(number_nodes=8, number_edges=16, save_data=True) 
else:
    print(f"Reusing graph {graph_id}")
    G = nx.read_graphml(f'{par.DATA_PATH_RAW}G{graph_id}.graphml')


G_prior = nx.read_graphml(f'{par.DATA_PATH_RAW}G{graph_id}.graphml')
def run_mutations(G_prior, graph_id, n_mut = 100):
    for i in range(n_mut):
        G_mutated = networks.to_directed_graph(G_prior, shuffle=True)
        try:
            for weight_index in range(len(weight_type_vec)):
                training.train_buffer(G_mutated, training_type=training_type, training_steps=training_steps, weight_type=weight_type_vec[weight_index], delta_weight = delta_weight_vec[weight_type_index], learning_rate= learning_rate_vec[weight_type_index])
                
        except:
            print("Mutated graph not valid")
            continue





