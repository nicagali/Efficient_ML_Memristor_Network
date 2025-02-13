import networkx as nx
import parameters as par
import sys
sys.path.append(par.PACKAGE_PATH)
import ahkab    
import random

# This module contains functions to:
# 1 - Initialize graph parameters
# 2 - Define different graphs
# 3 - Transform a graph in a circuit

# 1 --------- INITIALIZE GRAPH PARAMETERS ---------

def initialize_nodes(G, voltage_input, voltage_desired):

    # Initialize every node with the same values of pressure = initial_pressure and rho = initial_rho
    initial_rho = 0.2
    initial_pressure = 1

    index_sources=0
    index_desired=0
    for node in G.nodes():

        G.nodes[node]['rho'] = initial_rho 
        G.nodes[node]['pressure'] = initial_pressure

    # Initialize source and desired voltages
        if G.nodes[node]['type'] == 'source':
            G.nodes[node]['voltage'] = voltage_input[index_sources]
            index_sources+=1
        if G.nodes[node]['type'] == 'target':
            # print(G.nodes[node]['desired'])
            G.nodes[node]['desired'] = voltage_desired[index_desired]
            index_desired+=1


def initialize_edges(G):

    initial_length = 10 # [mu m]
    initial_radius_base = 200 # [nm]
    initial_value_resistance = 50 #do I still need this??
    initial_value_conductance = 1/initial_value_resistance

    for edge in G.edges():
        G.edges[edge]['resistance'] = initial_value_resistance
        G.edges[edge]['conductance'] = initial_value_conductance
        G.edges[edge]['length'] = initial_length
        G.edges[edge]['radius_base'] = initial_radius_base
        G.edges[edge]['pressure'] = 0
        G.edges[edge]['delta_rho'] = 0

# 2 --------- DEFINE DIFFERENT GRAPHS ---------

# VOLTAGE DIVIDER: 0 --- 1 --- 2
def voltage_divider(save_data=False, voltage_desired = [4]):

    G = nx.DiGraph()    # I am using directed graphs to keep trak of sign when def circuit
    G.name = 'voltage_divider'

    # ADD nodes
    attributes = {"type" : "source", 'color' : par.color_dots[0]}
    G.add_node(0, **attributes)
    attributes = {"type" : "target", 'color' : par.color_dots[1]}
    G.add_node(1, **attributes)
    attributes = {"type" : "source", 'color' : par.color_dots[0]}
    G.add_node(2, **attributes)

    # ADD edges
    G.add_edge(0,1)
    G.add_edge(1,2)
    
    # INITIALIZE nodes and edges
    voltage_input = [5, 0] # node initialized here because different for differnent nw
    # voltage_desired = [4]

    initialize_nodes(G, voltage_input, voltage_desired)
    initialize_edges(G)

    # SAVE to data folder
    if save_data:
        nx.write_graphml(G, f"{par.DATA_PATH}voltage_divider.graphml")

    return G

# RANDOM NETWORK

 
def random_graph(save_data=False, res_change=False):

    # CREATE random graph with number_nodes conected by number_edges
    number_nodes = 5
    number_edges = 8
    # G = nx.gnm_random_graph(number_nodes, number_edges)
    G = nx.house_graph()
    G = nx.grid_2d_graph(2, 2)
    G = nx.convert_node_labels_to_integers(G)

    # DEFINE number sources and targets, then randomly select sources and targets nodes between number_nodes : sources containg source index and targets contains target indeces
    number_sources = 2
    number_targets = 1
    sources = random.sample(G.nodes(), number_sources)
    sources = [2,3,4]
    target_sampling_list = [x for x in G.nodes() if x not in sources]
    targets = random.sample(target_sampling_list, number_targets)
    targets = [0]

    volt_input = []
    # Assign attributes nodes
    volt_index=0
    for node in range(len(G.nodes)):
        if node in sources:
            G.nodes[node]['type'] = 'source'
            G.nodes[node]['color'] = par.color_dots[0]
            G.nodes[node]['constant_source'] = False

        elif node in targets:
            G.nodes[node]['type'] = 'target'
            G.nodes[node]['color'] = par.color_dots[1]
        else:
            G.nodes[node]['type'] = 'hidden'
            G.nodes[node]['color'] = par.color_dots[2]

    G.nodes[3]['constant_source'] = True

    # INITIALIZE nodes and edges
    voltage_input = [0, 2, 0] # node initialized here because different for differnent nw
    voltage_desired = [3, 4]

    initialize_nodes(G, voltage_input, voltage_desired)
    initialize_edges(G)

    if save_data:

        nx.write_graphml(G, f"{par.DATA_PATH}random_graph.graphml")

    return G


# 3 --------- GRAPH -> CIRCUIT ---------
# Create a the class 'Circuit' used in the package ahkab from the desired graph.
def circuit_from_graph(G, type):

    circuit = ahkab.circuit.Circuit('Circuit')
    
    # ADD voltage sources 
    for node in G.nodes():

        # Adding the voltage sources from ground to input nodes
        if G.nodes[node]['type'] == 'source':
            circuit.add_vsource(f"VN{node}", n1=f'n{node}', n2=circuit.gnd, dc_value=G.nodes[node]['voltage'])
            
    # ADD elements on links
    for index, edge in enumerate(G.edges()):
        
        # An edge = (u,v), the nodes are then called 'n u' and 'n v', u = edge[0], ...

        if type == 'memristors':

            circuit.add_mysistor(f'R{index+1}', f'n{edge[0]}', f'n{edge[1]}', value = G.edges[edge]["conductance"], rho_b=G.nodes[edge[0]]['rho'], length_channel = G.edges[edge]['length']*1e-6, radius_base = G.edges[edge]['radius_base']*1e-9, pressure=(G.nodes[edge[0]]['pressure']-G.nodes[edge[1]]['pressure'])*1e5, delta_rho = (G.nodes[edge[0]]['rho']-G.nodes[edge[1]]['rho']))

        else:

            circuit.add_resistor(f'R{index}', f'n{edge[0]}', f'n{edge[1]}', value = G.edges[edge]['resistance'])
        
    return circuit
