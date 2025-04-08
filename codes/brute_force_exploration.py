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
import pandas as pd

start = time.time()

# new = int(sys.argv[1])  # 1 -> generate a new graph, 0 -> use graph with graph_id sys.argv[2]
# graph_id = str(sys.argv[2]) # Graph ideantification
# n_mut = int(sys.argv[3])    # Number of mutations (change in memr directions) of a graph
new = 0  # 1 -> generate a new graph, 0 -> use graph with graph_id sys.argv[2]
graph_id = '0001' # Graph ideantification
mut_id = '0001' 
n_mut = 20   # Number of mutations (change in memr directions) of a graph

training_steps = 300   # choose
training_type = 'regression'

weight_type_vec = ['length', 'radius_base', 'rho', 'pressure']
weight_symbol_vec = ['L', 'Rb', 'rho', 'P']
delta_weight_vec = [1e-3, 1e-3, 1e-4, 1e-3]
learning_rate_vec = [3e-6, 5e-6, 1e-3, 2e2]

def run_mutations(G_prior, graph_id, n_mut):
    df = pd.DataFrame({})
    for i in range(3,n_mut):
        print(f"Performing mutation {i}")
        G_mutated = networks.to_directed_graph(G_prior, shuffle=True)
        mut_id = str(i).zfill(4)
        graph_id_mut = f'{graph_id}{mut_id}'
        data = {"graph_id" : [f'{graph_id_mut}']}
        # try:
        for weight_index in range(len(weight_type_vec)):
            print(f"----> Training {weight_type_vec[weight_index]}")
            # result -> training result, returns: minimum value error, avrg last third, stdv last third
            result = training.train_buffer(G_mutated, training_type=training_type, training_steps=training_steps, weight_type=weight_type_vec[weight_index], delta_weight = delta_weight_vec[weight_index], learning_rate= learning_rate_vec[weight_index])
            data.update({f'{weight_symbol_vec[weight_index]}errmin' : [result[0]]})
            data.update({f'{weight_symbol_vec[weight_index]}avg' : [result[1]]})
            data.update({f'{weight_symbol_vec[weight_index]}stdv' : [result[2]]})
                
        #except:
        #    print("error: Mutated graph not valid")
        #    continue
        
        df = pd.concat([df, pd.DataFrame(data)])
        nx.write_graphml(G_mutated, f'{par.DATA_PATH_RAW}G{graph_id_mut}.graphml')
    df.to_csv(f'{par.DATA_PATH_RAW}T{graph_id}.csv', index=False)

    return

if __name__ == "__main__":    
    if new == 1:
        print("New graph generated")
        # -> DEFINE graph from networks module
        G = networks.random_graph(number_nodes=8, number_edges=16, save_data=True) 
    else:
        print(f"Reusing graph {graph_id}")
        G = nx.read_graphml(f'{par.DATA_PATH_RAW}G{graph_id}{mut_id}.graphml')
    run_mutations(G_prior=G,graph_id=graph_id,n_mut=n_mut)


