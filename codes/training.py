import parameters as par
import networks

import sys
sys.path.append(par.PACKAGE_PATH)
import ahkab 

# --------- WRITE RESULTS TO FILES ---------

def write_weights_to_file(G, step, weight_type, path_patch):

    file = open(f"{par.DATA_PATH}weights/{path_patch}/{weight_type}/{weight_type}{step}.txt", "w")
    
    for index, edge in enumerate(G.edges()): 
        if weight_type=='length' or weight_type=='rbrt':
            file.write(f"{index}\t{G.edges[edge][weight_type]}\n")
    for node in G.nodes():
        if weight_type=='pressure' or weight_type=='rho':  
            file.write(f"{int(node)}\t{G.nodes[node][weight_type]}\n")

    file.close()
    
    return

# --------- UTILITIES FUNCTIONS ---------

def voltage_drop_element(circuit, result, element):

    if isinstance(element, str):
        element = ahkab.circuit.Circuit.get_elem_by_name(circuit, element)

    node1 = element.n1 
    node2 = element.n2 
    
    if node1==0:
        voltage_node1 = 0
    else:
        ext_name = circuit.nodes_dict[node1]
        voltage_node1 = result['tran'][f'VN{ext_name[1:]}']
        
    if node2==0:
        voltage_node2 = 0
    else:
        ext_name = circuit.nodes_dict[node2]
        voltage_node2 = result['tran'][f'VN{ext_name[1:]}']

    voltage_drop = voltage_node1 - voltage_node2

    return voltage_drop

# --------- ALGORITHM FUNCTIONS ---------

def cost_function(G, write_potential_drops_to_file=None, update_initial_res = False):
    
    # TRANSFORM graph into circuit
    circuit = networks.circuit_from_graph(G, type='memristors') 

    # DEFINE a transient analysis (analysis of the circuit over time)
    tran_analysis = ahkab.new_tran(tstart=0, tstop=0.1, tstep=1e-3, x0=None)
    result = ahkab.run(circuit, an_list=tran_analysis) #returns two arrays: conductance over time of the memristors, voltages over time in the nodes

    conductance_vec = result[1]
    result = result[0]

    # UPDATE the value of the initial conductance to the last in the tran simulation to speed up (in a voltage divider 5 steps speeds of 0.01s giving same results)
    if update_initial_res:
        for index, edge in enumerate(G.edges()):
            conductance = conductance_vec[index][-1]
            G.edges[edge]['conductance'] = conductance

    # COMPUTE error     
    error=0
    for node in G.nodes():
        target_index=0
        if G.nodes[node]['type']=='target':  
            error += (G.nodes[node]['desired'] - result['tran'][f'VN{node}'][-1])**2
            target_index+=1
            
    # WRITE last element potential drop each edge (useful in the voltage divider case, otherwise too many)
    if write_potential_drops_to_file is not None:

        for index, edge in enumerate(G.edges()):

            voltage_drop_vec = voltage_drop_element(circuit, result, f"R{index+1}")

            write_potential_drops_to_file.write(f"{voltage_drop_vec[-1]}\t")
        write_potential_drops_to_file.write(f"\n")

    return error    

def update_weights(G,base_error, weight_type, delta_weight, learning_rate):

    gradients = []
    
    if weight_type=='pressure' or weight_type == 'rho':

        for node in G.nodes():

            G_increment = G.copy(as_view=False)
            
            G_increment.nodes[node][f'{weight_type}'] += delta_weight

            new_error = cost_function(G_increment)

            if weight_type == 'pressure':
                denominator = delta_weight*1e5
            else:
                denominator = delta_weight
                
            gradients.append((new_error - base_error)/denominator)

        for node in G.nodes():

            G.nodes[node][f'{weight_type}'] -= learning_rate*gradients[int(node)]
    else:

        for index, edge in enumerate(G.edges()):

            G_increment = G.copy(as_view=False)
  
            G_increment.edges[edge][f'{weight_type}'] += delta_weight
            new_error = cost_function(G_increment)

            if weight_type == 'length':
                denominator = delta_weight*1e-6
            else:
                denominator = delta_weight
            
            gradients.append((new_error - base_error)/delta_weight)
            
        for index, edge in enumerate(G.edges()):    #Different loop cause you don't want to change edges yet

            G.edges[edge][f'{weight_type}'] -= learning_rate*gradients[index]

        
    return G

# --------- TRAINING FUNCTIONS ---------

def train(G, training_steps, weight_type, delta_weight, learning_rate):

    # OPEN files to write results
    mse_file = open(f"{par.DATA_PATH}mse/mse_allostery_{weight_type}.txt", "w") #write error 
    potential_drops_file = open(f"{par.DATA_PATH}potential_drops.txt", "w") #write potential drops across links       

    path_patch = 'allostery' 

    # WRITE initial condition: intialized wieghts and intial error
    write_weights_to_file(G, step=0, weight_type=weight_type, path_patch=path_patch)
    
    error = cost_function(G, potential_drops_file, update_initial_res=False)    #compute initial error
    error_normalization = error #define it as normalization error

    mse_file.write(f"{error/error_normalization}\n")

    print('Step:', 0, error)
    
    # LOOP over training steps
    for step in range(training_steps): 

        update_weights(G, error, weight_type, delta_weight, learning_rate)

        # update_weights_parallel(G, error, weight_type, delta_weight, learning_rate)
            
        write_weights_to_file(G, step+1, weight_type, path_patch)

        error = cost_function(G, potential_drops_file, update_initial_res=False)

        print('Step:', step+1, error)
        mse_file.write(f"{error/error_normalization}\n")

    mse_file.close()
    potential_drops_file.close()