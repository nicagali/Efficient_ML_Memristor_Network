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

def plot_ginfty(ax, circuit, mysistor_id, potential_vec=False, result=None):

    if potential_vec == True:
        
        potential_vec = np.linspace(-5, 5)
        mysistor = ahkab.Circuit.get_elem_by_name(circuit, part_id=mysistor_id)
        g_infinity = []
        for potential in potential_vec:
            g_infinity.append(ahkab.transient.g_infinity_func(potential, 0, 0, mysistor)*mysistor.g_0)
        ax.plot(potential_vec, g_infinity, **par.g_infinity_style[f'{mysistor_id}'])
        
        ax.set_ylabel(r'$g_{\infty}[pS]$', fontsize = par.axis_fontsize)
        ax.set_xlabel(r'V[V]', fontsize = par.axis_fontsize)
        ax.tick_params(axis='both', labelsize=par.size_ticks)
        ax.legend(fontsize = par.legend_size)
        
    else:
        deltav = 0 
        if mysistor_id == 'M1':
            deltav = result['tran']['VN0'] - result['tran']['VN1']
        else:
            deltav = result['tran']['VN1'] - result['tran']['VN2']
            
        mysistor = ahkab.Circuit.get_elem_by_name(circuit, part_id =mysistor_id)
        g_infinity = []
        for potential in deltav:
            g_infinity.append(ahkab.transient.g_infinity_func(potential, 0, 0, mysistor)*mysistor.g_0)
        ax.plot(result['tran']['T'], g_infinity, **par.g_infinity_style[f'{mysistor_id}'])
        ax.legend(fontsize = par.legend_size)   

def plot_potentialdiff(ax, result):
    
    deltav1 = result['tran']['VN0'] - result['tran']['VN1']
    delltav2 = result['tran']['VN1'] - result['tran']['VN2']
    
    ax.plot(result['tran']['T'], result['tran']['VN0'], **par.potential_drops_style['v1'])
    ax.plot(result['tran']['T'], deltav1, **par.potential_drops_style['deltav1'])
    ax.plot(result['tran']['T'], delltav2, **par.potential_drops_style['deltav2'])
    ax.plot(result['tran']['T'], result['tran']['VN2'], **par.potential_drops_style['v3'])
    
    ax.set_ylabel(r'$V[V]$', fontsize = par.axis_fontsize)
    ax.set_xlabel(r't[s]', fontsize = par.axis_fontsize)
    ax.tick_params(axis='both', labelsize=par.size_ticks)
    ax.legend(fontsize = par.legend_size)
    
def plot_conductances(ax, resistances_vec, result):
    
    x = np.array(range(len(resistances_vec)))

    for edge_index in range(len(resistances_vec[0])):
        y = [1/(resistances_vec[time_index][edge_index]) for time_index in range(len(resistances_vec))]
        ax.plot(result['tran']['T'], y, **par.conductances_style[f'M{edge_index+1}']) 
        
    ax.set_ylabel(r'$g[pS]$', fontsize = par.axis_fontsize)
    ax.set_xlabel(r't[s]', fontsize = par.axis_fontsize)
    ax.tick_params(axis='both', labelsize=par.size_ticks)
    ax.legend(fontsize = par.legend_size)

def plot_analysis_relaxation(circuit, potentials_vec, resistances_vec, phase):
        
    fig, ax = plt.subplots()
    plot_ginfty(ax, circuit, 'M1', potential_vec=True)
    plot_ginfty(ax, circuit, 'M2', potential_vec=True)
    ax.legend(fontsize = par.legend_size)
    fig.tight_layout()
    plt.savefig(f"../plots/importance_ginfinity_dynamics/plots/g_infities_{phase}.pdf")

    fig, ax = plt.subplots()
    plot_potentialdiff(ax, potentials_vec)
    fig.tight_layout()
    plt.savefig(f"../plots/importance_ginfinity_dynamics/plots/potential_drops_{phase}.pdf")

    fig, ax = plt.subplots()
    plot_ginfty(ax, circuit, 'M1', result = potentials_vec)
    plot_ginfty(ax, circuit, 'M2', result = potentials_vec)
    plot_conductances(ax, resistances_vec, potentials_vec)
    fig.tight_layout()
    plt.savefig(f"../plots/importance_ginfinity_dynamics/plots/conductances_{phase}.pdf")

# -> DEFINE graph from networks module
G = networks.voltage_divider(save_data=True) 

circuit = networks.circuit_from_graph(G, type='memristors') 


# ANALYSIS IN INITIAL CONDITION
tran_analysis = ahkab.new_tran(tstart=0, tstop=0.1, tstep=1e-3, x0=None, use_step_control=False)
result = ahkab.run(circuit, an_list=tran_analysis) 
potentials_vec = result[0]
resistances_vec = result[1]

plot_analysis_relaxation(circuit, potentials_vec, resistances_vec, 'initial_condition')

# ANALYSIS WHEN TRAINED FOR LENGTH

# G_trained = networks.voltage_divider(save_data=True) 
# training.train(G_trained, training_type='allostery', training_steps=20, weight_type='length', delta_weight = 1e-3, learning_rate=3e-6)
# nx.write_graphml(G_trained, '../plots/importance_ginfinity_dynamics/Gtrained.graphml')
G_trained = nx.read_graphml('../plots/importance_ginfinity_dynamics/Gtrained.graphml')

circuit_trained = networks.circuit_from_graph(G_trained, type='memristors') 

tran_analysis = ahkab.new_tran(tstart=0, tstop=0.02, tstep=1e-3, x0=None)
result = ahkab.run(circuit_trained, an_list=tran_analysis) 
potentials_vec = result[0]
resistances_vec = result[1]

plot_analysis_relaxation(circuit_trained, potentials_vec, resistances_vec, 'trained')


