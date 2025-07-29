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

def plot_ginfty(ax, circuit, mysistor_id, potential_vec=False, result=None, steady_state_point = None, change_func=False, type=None):

    if type=='const_l':
        style = 'M1'
    else:
        style = 'M2'

    if potential_vec == True:
        
        potential_vec = np.linspace(-5, 5)
        mysistor = ahkab.Circuit.get_elem_by_name(circuit, part_id=mysistor_id)
        print(mysistor.length_channel)
        g_infinity = []
        for potential in potential_vec:
            if change_func and mysistor_id=='M1':
                g_infinity_value = ahkab.transient.g_infinity_func(potential, 0, 0, mysistor, change_func=True)*mysistor.g_0
            else:
                g_infinity_value = ahkab.transient.g_infinity_func(potential, 0, 0, mysistor)*mysistor.g_0
            g_infinity.append(g_infinity_value)
        ax.plot(potential_vec, g_infinity, **par.g_infinity_style[f'{style}'], zorder=0)
    
            
        ax.set_ylabel(r'$g_{\infty}[pS]$', fontsize = par.axis_fontsize)
        ax.set_xlabel(r'V[V]', fontsize = par.axis_fontsize)
        ax.tick_params(axis='both', labelsize=par.size_ticks)
        ax.legend(fontsize = par.legend_size, loc = 'upper right')
        
    else:
        deltav = result['tran']['VN1'] 
            
        mysistor = ahkab.Circuit.get_elem_by_name(circuit, part_id =mysistor_id)
        g_infinity = []
        for potential in deltav:
            if change_func and mysistor_id=='M1':
                g_infinity_value = ahkab.transient.g_infinity_func(potential, 0, 0, mysistor, change_func=True)*mysistor.g_0
            else:
                g_infinity_value = ahkab.transient.g_infinity_func(potential, 0, 0, mysistor)*mysistor.g_0
            g_infinity.append(g_infinity_value)
        ax.plot(result['tran']['T'], g_infinity, **par.g_infinity_style[f'{style}'], zorder = 0)
        ax.legend(fontsize = par.legend_size)   

def plot_conductances(ax, resistances_vec, result, type):
    
    if type=='const_l':
        style = 'M1'
    else:
        style = 'M2'

    for edge_index in range(len(resistances_vec[0])):
        y = [1/(resistances_vec[time_index][edge_index]) for time_index in range(len(resistances_vec))]
        ax.plot(result['tran']['T'], y, **par.conductances_style[f'{style}'], zorder = 0) 
        
    ax.set_ylabel(r'$g[pS]$', fontsize = par.axis_fontsize)
    ax.set_xlabel(r't[s]', fontsize = par.axis_fontsize)
    ax.tick_params(axis='both', labelsize=par.size_ticks)
    ax.legend(fontsize = par.legend_size)
    
    return 

def plot_length(ax, len_vec, result, type):
    
    if type=='const_len':
        style = 'M1'
    else:
        style = 'M2'

    ax.plot(result['tran']['T'], len_vec, **par.lengths_style[f'{style}'])
        
    ax.set_ylabel(r'$g[pS]$', fontsize = par.axis_fontsize)
    ax.set_xlabel(r't[s]', fontsize = par.axis_fontsize)
    ax.tick_params(axis='both', labelsize=par.size_ticks)
    ax.legend(fontsize = par.legend_size)
    
    return 

def plot_analysis_relaxation(circuit, circuit2, result, result2):
    
    potentials = result[0]
    resistances = result[1]
    lengths = result[2]

    potentials2 = result2[0]
    resistances2 = result2[1]
    lengths2 = result2[2]

    deltav = potentials['tran']['VN1'] 
    
    # Plot potential
    fig, ax = plt.subplots()
    ax.plot(potentials['tran']['T'], potentials['tran']['VN1'], **par.potential_drops_style['v1'])
    fig.tight_layout()
    plt.savefig(f"../plots/single_memristor/potentials.pdf")

    # Plot conductances
    fig, ax = plt.subplots()
    plot_ginfty(ax, circuit, 'M1', result = potentials, type = 'const_l')
    plot_conductances(ax, resistances, result = potentials, type = 'const_l')
    plot_ginfty(ax, circuit2, 'M1', result = potentials2, type = 'var_l')
    plot_conductances(ax, resistances2, result = potentials2, type = 'var_l')
    fig.tight_layout()
    plt.savefig(f"../plots/single_memristor/conductances.pdf")
    
    # Plot lengths
    fig, ax = plt.subplots()
    plot_length(ax, len_vec = lengths, result = potentials, type = 'const_l')
    plot_length(ax, len_vec = lengths2, result = potentials2, type = 'var_l')
    fig.tight_layout()
    plt.savefig(f"../plots/single_memristor/lengths.pdf")

    # Plot g_infty as func of potential
    fig, ax = plt.subplots()
    plot_ginfty(ax, circuit, 'M1', potential_vec=True, type='const_l')
    plot_ginfty(ax, circuit2, 'M1', potential_vec=True, type='var_l')
    # ax.legend(fontsize = par.legend_size)
    fig.tight_layout()
    plt.savefig(f"../plots/single_memristor/g_infities.pdf")
   
def plot_length_function(circuit):

    potential = np.linspace(-5, 5)
    mysistor = ahkab.Circuit.get_elem_by_name(circuit, part_id = 'M1')


    function = ahkab.transient.length_sigmoid(potential) 
    function = function/(function[0])
    print(function)

    fig, ax = plt.subplots()
    ax.plot(potential, function, c = 'green', lw=4, label = rf'$\sigma(\Delta V)$')
    ax.set_ylabel(r'$L[\mu m]/L_0$', fontsize = par.axis_fontsize)
    ax.set_xlabel(r'$\Delta V$[$\Omega$]', fontsize = par.axis_fontsize)
    ax.tick_params(axis='both', labelsize=par.size_ticks)
    ax.legend(fontsize = par.legend_size)
    fig.tight_layout()
    plt.savefig(f"../plots/single_memristor/length_function.pdf")

def plot_length_analysis(len_vec, result):

    fig, ax = plt.subplots()
    ax.plot(result['tran']['T'], len_vec, c = 'mediumaquamarine', lw=4, label = rf'$\sigma(\Delta V)$')
    ax.set_ylabel(r'$L[\mu m]$', fontsize = par.axis_fontsize)
    ax.set_xlabel(r'$\Delta V$[$\Omega$]', fontsize = par.axis_fontsize)
    ax.tick_params(axis='both', labelsize=par.size_ticks)
    ax.legend(fontsize = par.legend_size)
    fig.tight_layout()
    plt.savefig(f"../plots/single_memristor/length_time.pdf")

# -> DEFINE graph from networks module
# G = networks.single_memristor(save_data=True) 
circuit = ahkab.circuit.Circuit('Single Memristor')
circuit.add_mysistor('M1', 'n1', circuit.gnd, value=1/4, rho_b=0.1, length_channel=10e-6, radius_base=200e-9, pressure=0, delta_rho=0)

circuit2 = ahkab.circuit.Circuit('Single Memristor')
circuit2.add_mysistor('M1', 'n1', circuit.gnd, value=1/4, rho_b=0.1, length_channel=10e-6, radius_base=200e-9, pressure=0, delta_rho=0)


period = 0.01
impulse_period = 10*period
# ANALYSIS IN INITIAL CONDITION
# constant 
# voltage_step = ahkab.time_functions.sin(phi = 90, vo=0, va=2, freq=1/period)
# circuit.add_vsource("VN0", 'n1', circuit.gnd, dc_value=3, function= voltage_step)
circuit.add_vsource("VN0", 'n1', circuit.gnd, dc_value=0.2)
analysis = ahkab.new_tran(tstart=0, tstop=impulse_period, tstep=1e-4, x0=None, varying_len = False)
result = ahkab.run(circuit, an_list=analysis)

# voltage_step = ahkab.time_functions.sin(phi = 90, vo=0, va=1, freq=1/period)
# circuit2.add_vsource("VN0", 'n1', circuit.gnd, dc_value=3, function=voltage_step)
circuit2.add_vsource("VN0", 'n1', circuit.gnd, dc_value=0.2)
analysis = ahkab.new_tran(tstart=0, tstop=impulse_period, tstep=1e-4, x0=None, varying_len = True)
result2 = ahkab.run(circuit2, an_list=analysis)

plot_analysis_relaxation(circuit, circuit2, result, result2)
plot_length_function(circuit)

