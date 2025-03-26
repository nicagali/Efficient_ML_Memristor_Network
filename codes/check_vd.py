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

# --------- INITIALIZE NETWORK ---------

# -> DEFINE graph from networks module
G = networks.voltage_divider(save_data=True) 

circuit = networks.circuit_from_graph(G, type='memristors') 
tran_analysis = ahkab.new_tran(tstart=0, tstop=0.1, tstep=1e-3, x0=None)
result = ahkab.run(circuit, an_list=tran_analysis) 
resistances_vec = result[1]
# print(resistances_vec)
result = result[0]
print(result['tran'].keys())
print(result['tran']['VN0'][-1])
print(result['tran']['VN1'][-1])
print(result['tran']['VN2'][-1])

pressure_vec = np.linspace(-1e6, 1e6)
# pressure_vec = np.linspace(-10, 10)
mysistor = ahkab.Circuit.get_elem_by_name(circuit, part_id='R1')
g_infinity = []
for pressure in pressure_vec:
    g_infinity.append(ahkab.transient.g_infinity_func(2.5, pressure, 0, mysistor)*mysistor.g_0)

plt.plot(pressure_vec, g_infinity)

g_infinity = []
for pressure in pressure_vec:
    g_infinity.append(ahkab.transient.g_infinity_func(-2.5, pressure, 0, mysistor)*mysistor.g_0)

plt.plot(pressure_vec, g_infinity)
plt.savefig('g_infty_pressure.pdf')
