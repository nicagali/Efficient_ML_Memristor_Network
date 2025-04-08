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

def plot_ginfty(ax, circuit, double=False):

    potential_vec = np.linspace(-10, 10)
    mysistor = ahkab.Circuit.get_elem_by_name(circuit, part_id='R1')
    g_infinity = []
    for potential in potential_vec:
        if potential<3 and double==True:
            # print(potential)
            g_infinity.append(ahkab.transient.g_infinity_func(potential, 0, 0, mysistor)*mysistor.g_0)
        else:
            g_infinity.append(ahkab.transient.g_infinity_func(potential, 0, 0, mysistor)*mysistor.g_0)

    ax.plot(potential_vec, g_infinity, label = 'M1')
    mysistor = ahkab.Circuit.get_elem_by_name(circuit, part_id='R2')
    g_infinity = []
    for potential in potential_vec:
        g_infinity.append(ahkab.transient.g_infinity_func(potential, 0, 0, mysistor)*mysistor.g_0)

    ax.plot(potential_vec, g_infinity)
    ax.plot(potential_vec, g_infinity, label = 'M2')


# -> DEFINE graph from networks module
G = networks.voltage_divider(save_data=True) 

circuit = networks.circuit_from_graph(G, type='memristors') 

fig, ax = plt.subplots()
plot_ginfty(ax, circuit)
# plot_ginfty(ax, g_0=2*4.20155902)
plt.savefig("../plots/show/g_infity.pdf")

# tran_analysis = ahkab.new_tran(tstart=0, tstop=0.1, tstep=1e-3, x0=None)
# result = ahkab.run(circuit, an_list=tran_analysis) 
# resistances_vec = result[1]

# result = result[0]
# print(result['tran'].keys())

# potential_drop1 = result['tran']['VN0'] - result['tran']['VN1']
# potential_drop2 = result['tran']['VN1'] - result['tran']['VN2']
# fig, ax = plt.subplots()
# ax.plot(result['tran']['T'], result['tran']['VN0'], label= r'$V_1$')
# ax.plot(result['tran']['T'], potential_drop1, label= r'$\Delta V_1$')
# ax.plot(result['tran']['T'], potential_drop2, label= r'$\Delta V_2$')
# ax.plot(result['tran']['T'], result['tran']['VN2'], label= r'$V_2$')
# ax.legend()
# plt.savefig("../plots/show/potential_drops.pdf")

fig, ax = plt.subplots()
plotting.plot_memristor_resistances(ax, G)
ax.legend()
plt.savefig("../plots/show/conductances.pdf")


# G_trained = networks.voltage_divider(save_data=True) 
# training.train(G_trained, training_type='allostery', training_steps=20, weight_type='length', delta_weight = 1e-3, learning_rate=3e-6)
# nx.write_graphml(G_trained, '../plots/show/Gtrained.graphml')
G_trained = nx.read_graphml('../plots/show/Gtrained.graphml')


fig, ax = plt.subplots()
plotting.plot_memristor_resistances(ax, G_trained)
ax.legend()
plt.savefig("../plots/show/conductances_trained.pdf")

circuit2 = networks.circuit_from_graph(G_trained, type='memristors') 

fig, ax = plt.subplots()
plot_ginfty(ax, circuit)
plot_ginfty(ax, circuit2, double=True)
# plot_ginfty(ax, g_0=2*4.20155902)
ax.legend()
plt.savefig("../plots/show/g_infity_trained.pdf")

tran_analysis = ahkab.new_tran(tstart=0, tstop=0.1, tstep=1e-3, x0=None)
result = ahkab.run(circuit2, an_list=tran_analysis) 
resistances_vec = result[1]

result = result[0]
print(result['tran'].keys())

potential_drop1 = result['tran']['VN0'] - result['tran']['VN1']
potential_drop2 = result['tran']['VN1'] - result['tran']['VN2']
fig, ax = plt.subplots()
ax.plot(result['tran']['T'], result['tran']['VN0'], label= r'$V_1$')
ax.plot(result['tran']['T'], potential_drop1, label= r'$\Delta V_1$')
ax.plot(result['tran']['T'], potential_drop2, label= r'$\Delta V_2$')
ax.plot(result['tran']['T'], result['tran']['VN2'], label= r'$V_2$')
ax.legend()
plt.savefig("../plots/show/potential_drops_trained.pdf")


