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
# G = networks.random_graph(save_data=True) 
G = nx.read_graphml(f'{par.DATA_PATH}random_graph.graphml')

# -> PLOT graph in /plots 
fig, ax = plt.subplots()
pos = plotting.plot_graph('random_graph')
fig.tight_layout()
fig.savefig(f"../plots/general_network/graph.pdf")