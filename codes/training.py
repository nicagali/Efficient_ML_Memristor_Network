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
        if weight_type=='length' or weight_type=='radius_base' or weight_type=='resistance':
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

def regression_function1(input1, input2):
    # print(input1)
    return 0.15*input1 +0.2*input2
def regression_function2(input1, inpu2):
    return 0.25*input1 + 0.1*inpu2 
def regression_function(input):
    return 0.2*input +0.2

# --------- ALGORITHM FUNCTIONS ---------

def cost_function(G, weight_type, write_potential_target_to_file=None, update_initial_res = False):
    
    # TRANSFORM graph into circuit
    if weight_type == 'resistance':
        circuit = networks.circuit_from_graph(G, type='resistors') 
        analysis = ahkab.new_op()
        # analysis = ahkab.new_tran(tstart=0, tstop=0.1, tstep=1e-3, x0=None)
    else:
        circuit = networks.circuit_from_graph(G, type='memristors') 
        analysis = ahkab.new_tran(tstart=0, tstop=0.1, tstep=1e-3, x0=None)

    # DEFINE a transient analysis (analysis of the circuit over time)
    result = ahkab.run(circuit, an_list=analysis) #returns two arrays: resistance over time of the memristors, voltages over time in the nodes

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
            if weight_type =='resistance':
                error += (G.nodes[node]['desired'] - result['op'][f'VN{node}'][0][0])**2
            else: 
                error += (G.nodes[node]['desired'] - result['tran'][f'VN{node}'][-1])**2

            target_index+=1
            # print(node, G.nodes[node]['desired'], result['op'][f'VN{node}'][0][0])
            
    # WRITE last element potential drop each edge (useful in the voltage divider case, otherwise too many)
    if G.name == 'voltage_divider' and write_potential_target_to_file is not None:

        if weight_type == 'resistance':
            write_potential_target_to_file.write(f"{result['op'][f'VN{1}'][0][0]}\n")
        else:
            write_potential_target_to_file.write(f"{result['tran'][f'VN{1}'][-1]}\n")


    return error    

def update_input_output_volt(G, input_voltage, desired_voltage, datastep=None):

    input_index = 0
    output_index = 0
    
    for node in G.nodes():
        
        if G.nodes[node]['type'] == 'source' and G.nodes[node]['constant_source']==False:

            if datastep == None:
                G.nodes[node]['voltage'] = input_voltage[input_index]
                # print(input_voltage[input_index])
            else:
                G.nodes[node]['voltage'] = input_voltage[input_index][datastep]


            # if datastep == 0:
                # print(input_voltage)
                # print(input_index)
                # print('input voltage', node, input_voltage[input_index][datastep])

            input_index += 1


        if G.nodes[node]['type'] == 'target':

            if datastep == None:
                G.nodes[node]['desired'] = desired_voltage[output_index]
            else:
                G.nodes[node]['desired'] = desired_voltage[output_index][datastep]


            # if datastep == 0:
                # print('output voltage', node, desired_voltage[output_index][datastep])

            output_index += 1

def cost_function_regression(G, weight_type, dataset_input_voltage, dataset_output_voltage, datastep, error_type):

    if error_type == 'training_error':
        update_input_output_volt(G, dataset_input_voltage, dataset_output_voltage)
        error = cost_function(G, weight_type)

    else:

        error = 0
        for datastep in range(len(dataset_input_voltage[0])):
            update_input_output_volt(G, dataset_input_voltage, dataset_output_voltage, datastep)
            error += cost_function(G, weight_type) 
        error = error/len(dataset_input_voltage[0])

    return error

def update_weights(G, training_type, base_error, weight_type, delta_weight, learning_rate, dataset_input_voltage, dataset_output_voltage, datastep):

    gradients = []
    
    if weight_type=='pressure' or weight_type == 'rho':

        for node in G.nodes():

            G_increment = G.copy(as_view=False)
            
            G_increment.nodes[node][f'{weight_type}'] += delta_weight

            if training_type == 'allostery':
                error = cost_function(G_increment)  
            else:
                error = cost_function_regression(G_increment, weight_type, dataset_input_voltage, dataset_output_voltage)

            if weight_type == 'pressure':
                denominator = delta_weight*1e5
            else:
                denominator = delta_weight
                
            gradients.append((error - base_error)/denominator)

        for node in G.nodes():

            G.nodes[node][f'{weight_type}'] -= learning_rate*gradients[int(node)]

    elif weight_type == 'resistance':

        for index, edge in enumerate(G.edges()):

            G_increment = G.copy(as_view=False)
  
            # print('weight before', G_increment.edges[edge][f'{weight_type}'])
            G_increment.edges[edge][f'{weight_type}'] += delta_weight
            # print('weight after', G_increment.edges[edge][f'{weight_type}'])
            
            if training_type == 'allostery':
                error = cost_function(G_increment, weight_type)  
            else:
                error = cost_function_regression(G_increment, weight_type, dataset_input_voltage, dataset_output_voltage, datastep, error_type='training')

            # print(index, base_error, error, (error - base_error), (error - base_error)/delta_weight)

            gradients.append((error - base_error)/delta_weight)
            
        for index, edge in enumerate(G.edges()):    #Different loop cause you don't want to change edges yet

            
            G.edges[edge][f'{weight_type}'] -= learning_rate*gradients[index]
            # print(G.edges[edge][f'{weight_type}'])
            if G.edges[edge][f'{weight_type}'] < 0:
                G.edges[edge][f'{weight_type}'] += learning_rate*gradients[index]
                print(f"Error: Negative weight detected for edge {edge} with weight type '{weight_type}'.")

    else:
        # print('updating')

        for index, edge in enumerate(G.edges()):

            G_increment = G.copy(as_view=False)
  
            G_increment.edges[edge][f'{weight_type}'] += delta_weight
            
            if training_type == 'allostery':
                error = cost_function(G_increment, weight_type)  
            else:
                error = cost_function_regression(G_increment, weight_type, dataset_input_voltage, dataset_output_voltage, datastep, error_type='training')

            if weight_type == 'length':
                denominator = delta_weight*1e-6
            else:
                denominator = delta_weight*1e-9

            # print(index, base_error, G_increment.edges[edge][f'{weight_type}'], error)

            gradients.append((error - base_error)/denominator)
            
        for index, edge in enumerate(G.edges()):    #Different loop cause you don't want to change edges yet

            G.edges[edge][f'{weight_type}'] -= learning_rate*gradients[index]
            # print(G.edges[edge][f'{weight_type}'])

            if G.edges[edge][f'{weight_type}'] < 0:
                sys.exit(f"Error: Negative weight detected for edge {edge} with weight type '{weight_type}'.")
        
    return G

def update_output(G, voltages):

    index_do = 0
    eta = 0.1

    for node in G.nodes():

        if G.nodes[node]['type'] == 'target':

            # print('upadting clasmped', node, voltages[int(node)], (G.nodes[node]['desired']), eta*(G.nodes[node]['desired']) + (1 - eta) * (voltages[node]))

            G.nodes[node]['type'] = 'source'

            G.nodes[node]['voltage'] = eta*(G.nodes[node]['desired']) + (1 - eta) * (voltages[int(node)])
            # G.nodes[node]['voltage'] = 1


    return G

def update_resistances(G_free, training_type, dataset_input_voltage, dataset_output_voltage):

    eta = 0.1
    alpha = 100000
    gamma = alpha/(2*eta)

    if training_type == 'regression':
        update_input_output_volt(G_free, dataset_input_voltage, dataset_output_voltage)

    # solve graph free

    circuit = networks.circuit_from_graph(G_free, type='resistors') 
    analysis = ahkab.new_op()
    result = ahkab.run(circuit, an_list=analysis) #returns two arrays: resistance over time of the memristors, voltages over time in the nodes
    result = result[0]
    voltages_free = np.array([result['op'][f'VN{node}'][0][0] for node in range(len(G_free.nodes()))])
    # print(result['op'])
    # print(voltages_free)

    G_clamped = G_free.copy(as_view=False)

    G_clamped = update_output(G_clamped, voltages_free)

    circuit = networks.circuit_from_graph(G_clamped, type='resistors') 
    analysis = ahkab.new_op()
    result = ahkab.run(circuit, an_list=analysis) #returns two arrays: resistance over time of the memristors, voltages over time in the nodes
    result = result[0]
    voltages_clamped = np.array([result['op'][f'VN{node}'][0][0] for node in G_clamped.nodes()])
    # print(voltages_clamped)


    for edge in G_free.edges():

        u, v = edge

        # print(u,v, voltages_free[u], voltages_free[v])

        diff_free = np.abs(voltages_free[int(u)] - voltages_free[int(v)])
        diff_clamped = np.abs(voltages_clamped[int(u)] - voltages_clamped[int(v)])

        delta_v = diff_clamped-diff_free
        delta_v_sq = diff_clamped**2 - diff_free**2
        

        prefac = gamma * (1 / (G_free.edges[edge]['resistance'])**2)

        delta_R_cont = prefac * (delta_v_sq)

        G_free.edges[edge]['resistance'] += delta_R_cont

        if G_free.edges[edge]['resistance'] < 0 :

            print(f"Error: Negative weight detected for edge {edge} with weight type resistance.")
            
            G_free.edges[edge]['resistance'] -= delta_R_cont
        
    return G_free 

# Returns two arrays with length 15: input voltage and corresponding desired output following the linear relationship

def generate_dataset_single(training_steps, random = True):

    training_steps += 1

    if random:
        input_voltage = np.random.uniform(-3,3, training_steps)
    else:
        input_voltage = np.linspace(-3,3, training_steps)

    desired_output = regression_function(input_voltage)

    input_voltage = [input_voltage]
    desired_output = [desired_output]

    # print(input_voltage)

    return input_voltage, desired_output


def generate_dataset(training_steps):

    training_steps +=1

    # input_voltage1 = np.random.uniform(1,5,19)
    # input_voltage2 = np.random.uniform(1,5,19)
    # input_voltage1 = np.linspace(1,5,19)
    # input_voltage2 = np.linspace(1,5,19)

    input_voltage1 = np.random.uniform(1,5, training_steps)
    input_voltage2 = np.random.uniform(1,5, training_steps)

    desired_output1 = regression_function1(input_voltage1, input_voltage2)
    desired_output2 = regression_function2(input_voltage1, input_voltage2)

    input_voltage = [input_voltage1, input_voltage2]
    desired_output = [desired_output1, desired_output2]

    return input_voltage, desired_output


# FUNC----------- PARALLEL

def compute_single_gradient_parallel_helper(G, weight_index, training_type, base_error, weight_type, delta_weight, dataset_input_voltage, dataset_output_voltage, datastep,  stored_gradient, lock):
    
    gradient = 0
    G_increment = G.copy(as_view=False)
    if weight_type=='pressure' or weight_type == 'rho':

        node = list(G.nodes)[weight_index]
        G_increment.nodes[node][f'{weight_type}'] += delta_weight

        if weight_type == 'pressure':
                denominator = delta_weight*1e5
        else:
                denominator = delta_weight

        if training_type == 'allostery':
            error = cost_function(G_increment, weight_type)  
        else:
            error = cost_function_regression(G_increment, weight_type, dataset_input_voltage, dataset_output_voltage, datastep, error_type='training')
        
        gradient = (error - base_error) / denominator

    else:
        
        edge = list(G.edges)[weight_index]
        G_increment.edges[edge][f'{weight_type}'] += delta_weight

        denominator = delta_weight
        if weight_type == 'length':
            denominator = delta_weight*1e-6
        if weight_type == 'radius_base':
            denominator = delta_weight*1e-9
        

        if training_type == 'allostery':
            error = cost_function(G_increment, weight_type)  
        else:
            error = cost_function_regression(G_increment, weight_type, dataset_input_voltage, dataset_output_voltage, datastep, error_type='training')


        gradient = (error - base_error) / denominator
        # print(error - base_error)

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

def  update_weights_parallel(G, training_type, base_error, weight_type, delta_weight, learning_rate, dataset_input_voltage, dataset_output_voltage, step):

    if weight_type=='pressure' or weight_type=='rho':
        batch_size = G.number_of_nodes()
        # batch_size = 1
        number_of_weights = G.number_of_nodes()
    else:
        batch_size = int(G.number_of_edges()/4)
        # batch_size = G.number_of_edges()
        number_of_weights = G.number_of_edges()

    # Check if numb_nodes is a multiple of batch_size
    if number_of_weights % batch_size != 0:
        raise ValueError(f"number_of_weights ({number_of_weights}) must be a multiple of batch_size ({batch_size}).")

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
                                args=(G, p, training_type, base_error, weight_type, delta_weight, dataset_input_voltage, dataset_output_voltage, step,  stored_gradient, lock)) for p in range(i, i + batch_size)]

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
            # G.edges[edge][f'{weight_type}'] -= learning_rate*stored_gradient[index] + np.random.normal(0,0.1)
            
            # print(learning_rate*stored_gradient[index])
            if G.edges[edge][f'{weight_type}'] < 10:
                G.edges[edge][f'{weight_type}'] += learning_rate*stored_gradient[index]
                print(f"Error: Negative weight detected for edge {edge} with weight type '{weight_type}'.")



    return G

# --------- TRAINING FUNCTIONS ---------

def train(G, training_type, training_steps, weight_type, delta_weight, learning_rate):

    # OPEN files to write results
    mse_file = open(f"{par.DATA_PATH}mse/mse_{weight_type}.txt", "w") #write error 
    if G.name == 'voltage_divider': 
        potential_target_file = open(f"{par.DATA_PATH}potential_targets/potential_targets{G.nodes[1]['desired']}.txt", "w") #write potential target during training (for voltage divider)  
    else:
        potential_target_file = None     

    # WRITE initial condition: intialized wieghts and intial error
    write_weights_to_file(G, step=0, weight_type=weight_type, path_patch=training_type)

    dataset_input_voltage=[]
    dataset_output_voltage=[]
    if training_type == 'regression':
        counter_inputs = 0
        for node in G.nodes():
            if G.nodes[node]['type'] == 'source' and G.nodes[node]['constant_source']==False:
                counter_inputs+=1
            # print(node, G.nodes[node]['constant_source'])
        if counter_inputs == 1:
            dataset_input_voltage, dataset_output_voltage = generate_dataset_single(0)
            testset_input_voltage, testset_output_voltage = generate_dataset_single(20, random = False)
            # print(dataset_input_voltage)
        else:
            dataset_input_voltage, dataset_output_voltage = generate_dataset(training_steps)

    # COMPUTE initial error
    if training_type == 'allostery':
        error = cost_function(G, weight_type, potential_target_file, update_initial_res=False) 
        print('Step:', 0, error)
        error_normalization = error #define it as normalization error
        mse_file.write(f"{error/error_normalization}\n")

    else:
        error = cost_function_regression(G, weight_type, dataset_input_voltage, dataset_output_voltage, 0, error_type='training')
        test_error = cost_function_regression(G, weight_type, testset_input_voltage, testset_output_voltage, 0, error_type='test')
        print('Step:', 0, test_error)
        error_normalization = test_error #define it as normalization error
        mse_file.write(f"{test_error/error_normalization}\n")


    # mse_file.write(f"{error}\n")

    
    # LOOP over training steps
    for step in range(training_steps): 

        # update_weights(G, training_type, error, weight_type, delta_weight, learning_rate, dataset_input_voltage, dataset_output_voltage, step)

        update_weights_parallel(G, training_type, error, weight_type, delta_weight, learning_rate, dataset_input_voltage, dataset_output_voltage, step)

        # update_resistances(G, training_type, dataset_input_voltage, dataset_output_voltage)
            
        write_weights_to_file(G, step+1, weight_type, training_type)

        # COMPUTE error
        if training_type == 'allostery':
            error = cost_function(G, weight_type, potential_target_file, update_initial_res=False)  
            print('Step:', step+1, error)
            mse_file.write(f"{error/error_normalization}\n")
        else:
            dataset_input_voltage, dataset_output_voltage = generate_dataset_single(0)
            error = cost_function_regression(G, weight_type, dataset_input_voltage, dataset_output_voltage, step +1, error_type='training')
            test_error = cost_function_regression(G, weight_type, testset_input_voltage, testset_output_voltage, 0, error_type='test')
            print('Step:', step+1, test_error)
            mse_file.write(f"{test_error/error_normalization}\n")

                         
        # mse_file.write(f"{error}\n")


    mse_file.close()
    if G.name == 'voltage_divider':
        potential_target_file.close()


def train_reproduce(G, training_type, training_steps, weight_type, delta_weight, learning_rate):

    # OPEN files to write results
    mse_file = open(f"{par.DATA_PATH}mse/mse_allostery_{weight_type}.txt", "w") #write error 

    # WRITE initial condition: intialized wieghts and intial error
    write_weights_to_file(G, step=0, weight_type=weight_type, path_patch=training_type)

    dataset_input_voltage, dataset_output_voltage = generate_dataset()

    # print(dataset_input_voltage, dataset_output_voltage)

    # COMPUTE initial error

    error = cost_function_regression(G, weight_type, dataset_input_voltage, dataset_output_voltage)

    error_normalization = error #define it as normalization error

    mse_file.write(f"{error/error_normalization}\n")

    print('Step:', 0, error)
    
    # LOOP over training steps
    for step in range(training_steps): 

        update_weights(G, training_type, error, weight_type, delta_weight, learning_rate, dataset_input_voltage, dataset_output_voltage)

        update_resistances(G, G_clamped)

        # update_weights_parallel(G, training_type, error, weight_type, delta_weight, learning_rate, dataset_input_voltage, dataset_output_voltage)
            
        write_weights_to_file(G, step+1, weight_type, training_type)

        # COMPUTE error

        dataset_input_voltage, dataset_output_voltage = generate_dataset()

        error = cost_function_regression(G, weight_type, dataset_input_voltage, dataset_output_voltage)

        print('Step:', step+1, error)
        mse_file.write(f"{error/error_normalization}\n")

    mse_file.close()


def test_regression(G, step, weight_type):
        
    data = np.loadtxt(f"{par.DATA_PATH}weights/regression/{weight_type}/{weight_type}{step}.txt", unpack=True)
    weight_vec = data[1]

    if weight_type == 'pressure':

        for node in G.nodes():

            G.nodes[node][f'{weight_type}'] = weight_vec[int(node)]
    else:

        for index, edge in enumerate(G.edges):
            G.edges[edge][f'{weight_type}'] = weight_vec[index]
            
    reg_file = open(f"{par.DATA_PATH}relations_regression/relations_regression{step}.txt", "w") 

    counter_inputs = 0
    for node in G.nodes():
        if G.nodes[node]['type'] == 'source' and G.nodes[node]['constant_source']==False:
            counter_inputs+=1
    if counter_inputs == 1:
        testset_input_voltage, testset_output_voltage = generate_dataset_single(20, random = False)
    else:
        testset_input_voltage, testset_output_voltage = generate_dataset_single(20, random = False)


    error = 0
    for datastep in range(len(testset_input_voltage[0])):

        update_input_output_volt(G, testset_input_voltage, testset_output_voltage, datastep)

        if weight_type == 'resistance':
            circuit = networks.circuit_from_graph(G, type='resistors') 
            analysis = ahkab.new_op()
            # analysis = ahkab.new_tran(tstart=0, tstop=0.1, tstep=1e-3, x0=None)
        else:
            circuit = networks.circuit_from_graph(G, type='memristors') 
            analysis = ahkab.new_tran(tstart=0, tstop=0.1, tstep=1e-3, x0=None)

        # DEFINE a transient analysis (analysis of the circuit over time)
        result = ahkab.run(circuit, an_list=analysis) #returns two arrays: resistance over time of the memristors, voltages over time in the nodes

        resistance_vec = result[1]
        result = result[0]

        
        output_voltage_target = []
        for node in G.nodes():
            
            if G.nodes[node]['type']=='target':  
                if weight_type =='resistance':
                    output_voltage_target.append(result['op'][f'VN{node}'][0][0])
                else: 
                    output_voltage_target.append(result['tran'][f'VN{node}'][-1])

        reg_file.write(f"{testset_input_voltage[0][datastep]}\t{output_voltage_target[0]}")
        reg_file.write("\n")


    test_error = cost_function_regression(G, weight_type, testset_input_voltage, testset_output_voltage, 0, error_type='test')
    print(f"Error step {step}", test_error)
    reg_file.close()
    
    return
