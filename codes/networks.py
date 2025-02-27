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
    G.nodes[0]['constant_source'] = False
    attributes = {"type" : "target", 'color' : par.color_dots[1]}
    G.add_node(1, **attributes)
    attributes = {"type" : "source", 'color' : par.color_dots[0]}
    G.add_node(2, **attributes)
    G.nodes[2]['constant_source'] = True


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
    number_nodes = 10
    number_edges = 16
    G = nx.gnm_random_graph(number_nodes, number_edges)
    # G = nx.house_graph()
    # G = nx.grid_2d_graph(3, 3)
    # G = nx.convert_node_labels_to_integers(G)
    # G = nx.complete_graph(5)

    # DEFINE number sources and targets, then randomly select sources and targets nodes between number_nodes : sources containg source index and targets contains target indeces
    number_sources = 2
    number_targets = 1
    sources = random.sample(G.nodes(), number_sources)
    # sources = [2,3,4]
    target_sampling_list = [x for x in G.nodes() if x not in sources]
    targets = random.sample(target_sampling_list, number_targets)
    # targets = [0]

    volt_input = []
    # Assign attributes nodes
    constantsource_index=0
    for node in range(len(G.nodes)):
        if node in sources:
            G.nodes[node]['type'] = 'source'
            G.nodes[node]['color'] = par.color_dots[0]
            if constantsource_index < (number_sources-1):
                G.nodes[node]['constant_source'] = True
                constantsource_index += 1
            else:
                G.nodes[node]['constant_source'] = False
            print(node, G.nodes[node]['constant_source'])

        elif node in targets:
            G.nodes[node]['type'] = 'target'
            G.nodes[node]['color'] = par.color_dots[1]
        else:
            G.nodes[node]['type'] = 'hidden'
            G.nodes[node]['color'] = par.color_dots[2]

    # G.nodes[3]['constant_source'] = True

    # INITIALIZE nodes and edges
    voltage_input = [0, 2, 2] # node initialized here because different for differnent nw
    voltage_desired = [3, 4]

    initialize_nodes(G, voltage_input, voltage_desired)
    initialize_edges(G)

    if save_data:

        nx.write_graphml(G, f"{par.DATA_PATH}random_graph.graphml")

    return G

def three_inout(save_data=False):

    G = nx.Graph()

    attributes = {"type" : "source", 'color' : par.color_dots[0]}
    G.add_node(1, **attributes)
    G.nodes[1]['constant_source'] = False
    G.add_node(2, **attributes)
    G.nodes[2]['constant_source'] = True
    G.add_node(7, **attributes)
    G.nodes[7]['constant_source'] = True
    attributes = {"type" : "target", 'color' : par.color_dots[1]}
    G.add_node(4, **attributes)
    attributes = {"type" : "hidden", 'color' : par.color_dots[2]}
    G.add_node(3, **attributes)
    G.add_node(0, **attributes)
    G.add_node(5, **attributes)
    G.add_node(6, **attributes)
    G.add_node(8, **attributes)

    G.add_edges_from([(0,5), (0,6), (0,7), (3,6), (1,3), (5,7), (0,4), (4,7)])    #   for graphical reprer without one edge
    G.add_edges_from([(1,6), (1,8), (6,8), (3,5), (2,3), (2,5), (4,5), (0,3)])

    voltage_input = [0, 0, 2] # node initialized here because different for differnent nw
    voltage_desired = [3, 4]

    initialize_nodes(G, voltage_input, voltage_desired)
    initialize_edges(G)

    if save_data:

        nx.write_graphml(G, f"{par.DATA_PATH}three_inout.graphml")

    return G

# 3 --------- GRAPH -> CIRCUIT ---------
# Create a the class 'Circuit' used in the package ahkab from the desired graph.
def circuit_from_graph(G, type):

    circuit = ahkab.circuit.Circuit('Circuit')
    
    # ADD voltage sources 
    for node in G.nodes():

        # Adding the voltage sources from ground to input nodes
        if G.nodes[node]['type'] == 'source':
            # print(node, G.nodes[node]['type'], G.nodes[node]['voltage'])
            circuit.add_vsource(f"VN{node}", n1=f'n{node}', n2=circuit.gnd, dc_value=G.nodes[node]['voltage'])
            
    # ADD elements on links
    for index, edge in enumerate(G.edges()):
        
        # An edge = (u,v), the nodes are then called 'n u' and 'n v', u = edge[0], ...

        if type == 'memristors':

            circuit.add_mysistor(f'R{index+1}', f'n{edge[0]}', f'n{edge[1]}', value = G.edges[edge]["conductance"], rho_b=G.nodes[edge[0]]['rho'], length_channel = G.edges[edge]['length']*1e-6, radius_base = G.edges[edge]['radius_base']*1e-9, pressure=(G.nodes[edge[0]]['pressure']-G.nodes[edge[1]]['pressure'])*1e5, delta_rho = (G.nodes[edge[0]]['rho']-G.nodes[edge[1]]['rho']))

        else:

            circuit.add_resistor(f'R{index}', f'n{edge[0]}', f'n{edge[1]}', value = G.edges[edge]['resistance'])
        
    return circuit


def compute_regression_coefficients(G):

    input_index = 0
    output_index = 0

    for node in G.nodes():
        if G.nodes[node]['type'] == 'source':
            # print(G.nodes[node]['constant_source']) 
            if G.nodes[node]['constant_source'] == False:
                input_index = int(node)
        if G.nodes[node]['type'] == 'target':
            output_index = int(node)
    



    # for node in G.nodes():

    #     if G.edges[input_index][node] or G.edges[node][input_index]:

    #     Gmn -= G.edges[input_index][node]

    G_matrix = nx.laplacian_matrix(G).toarray()
    
    Gmn = G_matrix[output_index][input_index]
    # print(Gmn)
    Gmm = G_matrix[output_index][output_index]

    a = - Gmn/Gmm

    return a