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

training_steps = 2

fig, ax = plt.subplots()

G_ml = G.copy(as_view=False)
training.train(G_ml, training_steps=training_steps, weight_type='pressure', delta_weight=1e-2, learning_rate=100)
plotting.plot_mse(ax, fig, 'allostery_pressure') 
ax.plot([], [], ' ', label=r'$\Delta w = 1 \times 10^{-2}$')

# G_ml = G.copy(as_view=False)
# training.train(G_ml, training_steps=training_steps, weight_type='pressure', delta_weight=1e-3, learning_rate=100)
# plotting.plot_mse(ax, fig, 'allostery_pressure_1e-3')

# G_ml = G.copy(as_view=False)
# training.train(G_ml, training_steps=training_steps, weight_type='pressure', delta_weight=1e-4, learning_rate=100)
# plotting.plot_mse(ax, fig, 'allostery_pressure_1e-4')


ax.legend()
fig.tight_layout()
fig.savefig(f"../paper/plots/voltage_divider/mse_weights.pdf")


end = time.time()

print("Running time = ", end-start, "seconds")