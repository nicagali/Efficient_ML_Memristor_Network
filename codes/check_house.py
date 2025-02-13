import networks
import matplotlib.pyplot as plt
import plotting
import parameters as par
import time 
import training
import networkx as nx
import sys
sys.path.append(par.PACKAGE_PATH)
import ahkab  

G = networks.random_graph(save_data=True) 

tstop = 0.05
tstop = 0.1

training.update_input_output_volt(G, -2, -2)

# Feedforward -> reach steady state and get ouputs
circuit = networks.circuit_from_graph(G, type='memristors') 
tran_analysis = ahkab.new_tran(tstart=0, tstop=tstop, tstep=1e-3, x0=None)
result = ahkab.run(circuit, an_list=tran_analysis) 
resistances_vec = result[1]
# print(resistances_vec)
result = result[0]
print(result['tran'].keys())
print(result['tran']['VN2'][-1])
print(result['tran']['VN3'][-1])
print(result['tran']['VN4'][-1])
print(result['tran']['VN0'][-1])
print(result['tran']['VN1'][-1])
