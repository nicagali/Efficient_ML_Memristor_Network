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

G = nx.read_graphml(f'{par.DATA_PATH}random_graph_working.graphml')

# a = networks.compute_regression_coefficients(G)
# print('Coefficient a:', a)

G.nodes['3']['type']  = 'source'
G.nodes['3']['color'] = par.color_dots[0]
G.nodes['3']['constant_source']  = True
G.nodes['3']['voltage']  = 3

G.add_edge('6','2')
G.add_edge('6','4')
G.add_edge('4','5')
G.add_edge('6','7')

tstop = 0.05
tstop = 0.1

networks.initialize_edges(G)


# Feedforward -> reach steady state and get ouputs
circuit = networks.circuit_from_graph(G, type='memristors') 
tran_analysis = ahkab.new_tran(tstart=0, tstop=tstop, tstep=1e-3, x0=None)
result = ahkab.run(circuit, an_list=tran_analysis) 
resistances_vec = result[1]
# print(resistances_vec)
result = result[0]
print(result['tran'].keys())
print(result['tran']['VN0'][-1])
print(result['tran']['VN1'][-1])
print(result['tran']['VN2'][-1])
print(result['tran']['VN3'][-1])
print(result['tran']['VN4'][-1])
print(result['tran']['VN5'][-1])
print(result['tran']['VN6'][-1])
print(result['tran']['VN7'][-1])
print(result['tran']['VN8'][-1])

print(G.nodes['9']['rho '])
G.nodes['9']['rho'] += 100
print(G.nodes['9']['rho '])


# Feedforward -> reach steady state and get ouputs
circuit = networks.circuit_from_graph(G, type='memristors') 
tran_analysis = ahkab.new_tran(tstart=0, tstop=tstop, tstep=1e-3, x0=None)
result = ahkab.run(circuit, an_list=tran_analysis) 
resistances_vec = result[1]
# print(resistances_vec)
result = result[0]
print(result['tran'].keys())
print(result['tran']['VN0'][-1])
print(result['tran']['VN1'][-1])
print(result['tran']['VN2'][-1])
print(result['tran']['VN3'][-1])
print(result['tran']['VN4'][-1])
print(result['tran']['VN5'][-1])
print(result['tran']['VN6'][-1])
print(result['tran']['VN7'][-1])
print(result['tran']['VN8'][-1])
