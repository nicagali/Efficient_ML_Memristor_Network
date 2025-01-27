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
G = networks.random_graph(save_data=True) 

# -> PLOT graph in /plots 
fig, ax = plt.subplots()
pos = plotting.plot_graph('random_graph')
fig.tight_layout()
fig.savefig(f"../plots/general_network/graph.pdf")

# --------- TRAIN NETWORK WITH DIFFERENT WEIGHTS ---------

training_steps = 10

G_ml = G.copy(as_view=False)  
training.train(G_ml, training_steps=training_steps, weight_type='length', delta_weight = 1e-3, learning_rate=3e-6)
G_ml = G.copy(as_view=False)  
# training.train(G_ml, training_steps=training_steps, weight_type='rbrt', delta_weight = 1e-3, learning_rate=1)
G_ml = G.copy(as_view=False)  
# training.train(G_ml, training_steps=training_steps, weight_type='rho', delta_weight = 1e-4, learning_rate=4e-4)
G_pressure = G.copy(as_view=False)  
# training.train(G_pressure, training_steps=training_steps, weight_type='pressure', delta_weight = 1e-3, learning_rate=100)