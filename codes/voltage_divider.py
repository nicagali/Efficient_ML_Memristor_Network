import networks
import training
import plotting
import parameters as par
import matplotlib.pyplot as plt
import networkx as nx
import time

start = time.time()

# --------- INITIALIZE NETWORK ---------

# -> DEFINE graph from networks module
G = networks.voltage_divider(save_data=True) 

# -> PLOT graph in /plots 
fig, ax = plt.subplots()
pos = plotting.plot_graph('voltage_divider')
fig.tight_layout()
fig.savefig(f"../plots/voltage_divider/graph.pdf")

# --------- TRAIN NETWORK WITH PRESSURE ---------

training_steps = 20

G_ml = G.copy(as_view=False)  
training.train(G_ml, training_steps=training_steps, weight_type='pressure', delta_weight = 1e-5, learning_rate=100)

# --------- PLOT ERROR AND WEIGHTS WITH PRESSURE ---------

fig, ax = plt.subplots(1, 2, figsize = par.figsize_2horizontal)
plotting.plot_mse(ax[0], fig, f'allostery_pressure')
plotting.plot_weights(ax[1], G, training_steps=training_steps, training_job='allostery', weight_type='pressure')
fig.tight_layout()
fig.savefig(f"../paper/plots/voltage_divider/mse_weights.pdf")

# --------- PLOT EVOLUTION OF TRAINED NW ---------

fig, ax = plt.subplots(figsize = par.figsize_1)
plotting.plot_potential_drops_each_node(ax, G_ml)
fig.tight_layout()
fig.savefig(f"../paper/plots/voltage_divider/evolution_finalnw.pdf")

end = time.time()

print("Running time = ", end-start, "seconds")