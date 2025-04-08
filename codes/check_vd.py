import networks
import training
import plotting
import parameters as par
import matplotlib.pyplot as plt
import networkx as nx
import time
import matplotlib.gridspec as gridspec
import numpy as np
import ahkab

start = time.time()

def plot_ginfty(ax, g_0):

    potential = np.linspace(-10,10)
    g_infty = sigmoid(potential, g_0) * g0_standard
    ax.plot(potential, g_infty, label=f'{g_0:.5}')
    print(g_infty[0])
    ax.legend()

fig, ax = plt.subplots()
plot_ginfty(ax, g_0=4.20155902)
plot_ginfty(ax, g_0=2*4.20155902)
plt.savefig("../plots/g_infity.pdf")

# -> DEFINE graph from networks module
G = networks.voltage_divider(save_data=True) 

circuit = networks.circuit_from_graph(G, type='memristors') 
tran_analysis = ahkab.new_tran(tstart=0, tstop=0.1, tstep=1e-3, x0=None)
result = ahkab.run(circuit, an_list=tran_analysis) 
resistances_vec = result[1]

