import networks
import training
import plotting
import parameters as par
import matplotlib.pyplot as plt
import networkx as nx
import time
import matplotlib.gridspec as gridspec
import ahkab

start = time.time()
graph_id = 'G00080001'
DATA_PATH = f'{par.DATA_PATH}allostery{graph_id}/'
PLOT_PATH = f'{par.PLOT_PATH}allostery{graph_id}/'

# --------- INITIALIZE NETWORK ---------

# -> DEFINE graph from networks module
# G = networks.random_graph(number_nodes=9, number_edges=15, save_data=True) 
G = nx.read_graphml(f'{par.DATA_PATH}random_graph.graphml')
nx.write_graphml(G, f'{DATA_PATH}random_graph.graphml')
G.graph['name'] = graph_id

# -> PLOT graph in /plots 
fig, ax = plt.subplots()
pos = plotting.plot_graph(G)
fig.tight_layout()
fig.savefig(f"{PLOT_PATH}graph.pdf", transparent=True)


# circuit = networks.circuit_from_graph(G, type='memristors') 
# analysis = ahkab.new_tran(tstart=0, tstop=0.1, tstep=1e-3, x0=None, varying_len=True)

# # DEFINE a transient analysis (analysis of the circuit over time)
# result = ahkab.run(circuit, an_list=analysis) #returns two arrays: resistance over time of the memristors, voltages over time in the nodes

# resistance_vec = result[1]
# result = result[0]
# print(result])

# --------- TRAIN NETWORK WITH DIFFERENT WEIGHTS ---------

training_steps = 200
training_type = 'allostery'

G_train = G.copy(as_view=False)
training.train(G_train, training_type=training_type, training_steps=training_steps, weight_type='pressure', delta_weight = 1e-3, learning_rate=100, save_final_graph=True, write_weights=True, varying_len=False)

# --------- PLOT ERROR AND WEIGHTS ---------

fig, ax = plt.subplots()
plotting.plot_mse(ax, fig, graph_id, training_type, f'pressure')

G_train = G.copy(as_view=False)
training.train(G_train, training_type=training_type, training_steps=training_steps, weight_type='pressure', delta_weight = 1e-3, learning_rate=100, save_final_graph=True, write_weights=True, varying_len=True)
plotting.plot_mse(ax, fig, graph_id, training_type, f'pressure')


ax.legend(fontsize = par.legend_size)
fig.tight_layout()
fig.savefig(f"{PLOT_PATH}mse.pdf", transparent=True)

end = time.time()
print("Running time = ", end-start, "seconds")
