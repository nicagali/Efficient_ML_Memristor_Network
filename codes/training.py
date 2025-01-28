import parameters as par
import networks
import multiprocessing as mp
import numpy as np
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

def cost_function(G, write_potential_target_to_file=None, update_initial_res = False):
    
    # TRANSFORM graph into circuit
    circuit = networks.circuit_from_graph(G, type='memristors') 

    # DEFINE a transient analysis (analysis of the circuit over time)
    tran_analysis = ahkab.new_tran(tstart=0, tstop=0.1, tstep=1e-3, x0=None)
    result = ahkab.run(circuit, an_list=tran_analysis) #returns two arrays: resistance over time of the memristors, voltages over time in the nodes

    resistance_vec = result[1]
    result = result[0]

    # UPDATE the value of the initial conductance to the last in the tran simulation to speed up (in a voltage divider 5 steps speeds of 0.01s giving same results)
    if update_initial_res:
        for index, edge in enumerate(G.edges()):
            resistance = resistance_vec[-1][index]
            G.edges[edge]['conductance'] = resistance

    # COMPUTE error     
    error=0
    for node in G.nodes():
        target_index=0
        if G.nodes[node]['type']=='target':  
            # print(G.nodes[node]['desired'])
            error += (G.nodes[node]['desired'] - result['tran'][f'VN{node}'][-1])**2
            target_index+=1
            
    # WRITE last element potential drop each edge (useful in the voltage divider case, otherwise too many)
    if G.number_of_nodes() == 3 and write_potential_target_to_file is not None:

        write_potential_target_to_file.write(f"{result['tran'][f'VN{1}'][-1]}\n")

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

            # print(new_error- base_error)

            gradients.append((new_error - base_error)/denominator)
            
        for index, edge in enumerate(G.edges()):    #Different loop cause you don't want to change edges yet

            G.edges[edge][f'{weight_type}'] -= learning_rate*gradients[index]

        
    return G

# FUNC----------- PARALLEL

def compute_single_gradient_parallel_helper(G, weight_index, base_error, weight_type, delta_weight, stored_gradient, lock):
    
    gradient = 0
    G_increment = G.copy(as_view=False)
    if weight_type=='pressure' or weight_type == 'rho':

        # TRANSFORM weight_index into integer. If the graph is read, weight_index is a string, if the graph is generated, weight_index is an integer 
        # if isinstance(weight_index, str):
        #     weight_index = int(weight_index)

        G_increment.nodes[f'{weight_index}'][f'{weight_type}'] += delta_weight

        if weight_type == 'pressure':
                denominator = delta_weight*1e5
        else:
                denominator = delta_weight

        gradient = (cost_function(G_increment) - base_error) / denominator

    else:
        
        edge = list(G.edges)[weight_index]
        G_increment.edges[edge][f'{weight_type}'] += delta_weight

        if weight_type == 'length':
            denominator = delta_weight*1e-6
        else:
            denominator = delta_weight

        gradient = (cost_function(G_increment) - base_error) / denominator

    with lock:
        stored_gradient[weight_index] = gradient

    return stored_gradient
     
def to_shared_array(array):
    shared_array = mp.Array('d', array.size, lock=False)
    temp = np.frombuffer(shared_array, dtype=array.dtype)
    temp[:] = array.flatten(order = "C")
    return shared_array

def to_numpy_array(shared_array, shape):
    array = np.ctypeslib.as_array(shared_array)
    return array.reshape(shape)

def  update_weights_parallel(G, base_error, weight_type, delta_weight, learning_rate):

    if weight_type=='pressure' or weight_type=='rho':
        batch_size = G.number_of_nodes()
        number_of_weights = G.number_of_nodes()
    else:
        # batch_size = int(G.number_of_edges()/4)
        batch_size = G.number_of_edges()
        number_of_weights = G.number_of_edges()

    # Check if numb_nodes is a multiple of batch_size
    assert(number_of_weights%batch_size == 0)

    # Initialize gradient vector
    init_gradient = np.zeros((number_of_weights), dtype = np.float64)  

    # Create multiprocessing array to which the different processes can access to. 
    # Thanks to temp, we can write a numpy array to a mp array and initialize it in this case.
    shared_gradient = to_shared_array(init_gradient)    #Returns initialized shared array

    # Create bridge to a nupy vector 
    stored_gradient = to_numpy_array(shared_gradient, init_gradient.shape)

    lock = mp.Lock()
    # execute in batches
    for i in range(0, number_of_weights, batch_size):
        # execute all tasks in a batch
        processes = [mp.Process(target = compute_single_gradient_parallel_helper, 
                                args=(G, p, base_error, weight_type, delta_weight, stored_gradient, lock)) for p in range(i, i + batch_size)]

        # start all processes
        for process in processes:
            process.start()
        # wait for all processes to complete
        for process in processes:
            process.join()

    if weight_type == "pressure" or weight_type=='rho':

        for node in G.nodes():
            G.nodes[node][f'{weight_type}'] -= learning_rate*stored_gradient[int(node)]

    else:

        for index, edge in enumerate(G.edges()):  

            G.edges[edge][f'{weight_type}'] -= learning_rate*stored_gradient[index]
            # print(learning_rate, gradients[index],learning_rate*gradients[index])


    return G


# --------- TRAINING FUNCTIONS ---------

def train(G, training_steps, weight_type, delta_weight, learning_rate):

    # OPEN files to write results
    mse_file = open(f"{par.DATA_PATH}mse/mse_allostery_{weight_type}.txt", "w") #write error 
    potential_target_file = open(f"{par.DATA_PATH}potential_targets/potential_tagets{G.nodes[1]['desired']}.txt", "w") #write potential target during training (for voltage divider)       

    path_patch = 'allostery' 

    # WRITE initial condition: intialized wieghts and intial error
    write_weights_to_file(G, step=0, weight_type=weight_type, path_patch=path_patch)
    
    error = cost_function(G, potential_target_file, update_initial_res=False)    #compute initial error
    error_normalization = error #define it as normalization error

    mse_file.write(f"{error/error_normalization}\n")

    print('Step:', 0, error)
    
    # LOOP over training steps
    for step in range(training_steps): 

        update_weights(G, error, weight_type, delta_weight, learning_rate)

        # update_weights_parallel(G, error, weight_type, delta_weight, learning_rate)
            
        write_weights_to_file(G, step+1, weight_type, path_patch)

        error = cost_function(G, potential_target_file, update_initial_res=False)

        print('Step:', step+1, error)
        mse_file.write(f"{error/error_normalization}\n")

    # 

    mse_file.close()
    potential_target_file.close()