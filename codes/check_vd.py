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

def plot_ginfty(ax, circuit, mysistor_id, potential_vec=False, result=None, steady_state_point = None, change_func=False):

    if potential_vec == True:
        
        potential_vec = np.linspace(0, 5)
        mysistor = ahkab.Circuit.get_elem_by_name(circuit, part_id=mysistor_id)
        g_infinity = []
        for potential in potential_vec:
            if change_func and mysistor_id=='M1':
                g_infinity_value = ahkab.transient.g_infinity_func(potential, 0, 0, mysistor, change_func=True)*mysistor.g_0
            else:
                g_infinity_value = ahkab.transient.g_infinity_func(potential, 0, 0, mysistor)*mysistor.g_0
            g_infinity.append(g_infinity_value)
        ax.plot(potential_vec, g_infinity, **par.g_infinity_style[f'{mysistor_id}'], zorder=0)
        
        mysistor_index = int(mysistor_id[-1])-1
        ax.scatter(steady_state_point[mysistor_index][0], steady_state_point[mysistor_index][1], **par.steady_state_point[f'{mysistor_id}'], zorder=1)

        if change_func and mysistor_id=='M1':
            ax.scatter(1, 12.394654281291412, marker='x', color = 'red', s = 150, label = 'desired')
            
        ax.set_ylabel(r'$g_{\infty}[pS]$', fontsize = par.axis_fontsize)
        ax.set_xlabel(r'V[V]', fontsize = par.axis_fontsize)
        ax.tick_params(axis='both', labelsize=par.size_ticks)
        ax.legend(fontsize = par.legend_size, loc = 'upper right')
        
    else:
        deltav = 0 
        if mysistor_id == 'M1':
            deltav = result['tran']['VN0'] - result['tran']['VN1']
        else:
            deltav = result['tran']['VN1'] - result['tran']['VN2']
            
        mysistor = ahkab.Circuit.get_elem_by_name(circuit, part_id =mysistor_id)
        g_infinity = []
        for potential in deltav:
            if change_func and mysistor_id=='M1':
                g_infinity_value = ahkab.transient.g_infinity_func(potential, 0, 0, mysistor, change_func=True)*mysistor.g_0
            else:
                g_infinity_value = ahkab.transient.g_infinity_func(potential, 0, 0, mysistor)*mysistor.g_0
            g_infinity.append(g_infinity_value)
        ax.plot(result['tran']['T'], g_infinity, **par.g_infinity_style[f'{mysistor_id}'], zorder = 0)
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
    
    for edge_index in range(len(resistances_vec[0])):
        y = [1/(resistances_vec[time_index][edge_index]) for time_index in range(len(resistances_vec))]
        ax.plot(result['tran']['T'], y, **par.conductances_style[f'M{edge_index+1}'], zorder = 0) 
        
        ax.scatter(result['tran']['T'][-1], y[-1], **par.steady_state_point[f'M{edge_index+1}'], zorder = 1)
        
    ax.set_ylabel(r'$g[pS]$', fontsize = par.axis_fontsize)
    ax.set_xlabel(r't[s]', fontsize = par.axis_fontsize)
    ax.tick_params(axis='both', labelsize=par.size_ticks)
    ax.legend(fontsize = par.legend_size)
    
    return 

def plot_analysis_relaxation(circuit, potentials_vec, resistances_vec, phase):
    
    change_func = False
    if phase == 'trained_change_func':
        change_func=True
    
    deltav1 = potentials_vec['tran']['VN0'] - potentials_vec['tran']['VN1']
    delltav2 = potentials_vec['tran']['VN1'] - potentials_vec['tran']['VN2']
    
    steady_state_point = [ [deltav1[-1], 1/resistances_vec[-1][0]], [delltav2[-1], 1/resistances_vec[-1][1]] ]
    
    # Plot conductances
    fig, ax = plt.subplots()
    plot_ginfty(ax, circuit, 'M1', result = potentials_vec)
    plot_ginfty(ax, circuit, 'M2', result = potentials_vec)
    plot_conductances(ax, resistances_vec, potentials_vec)
    fig.tight_layout()
    plt.savefig(f"../plots/importance_ginfinity_dynamics/plots/conductances_{phase}.pdf")
    
    # Plot potential  
    fig, ax = plt.subplots()
    plot_potentialdiff(ax, potentials_vec)
    fig.tight_layout()
    plt.savefig(f"../plots/importance_ginfinity_dynamics/plots/potential_drops_{phase}.pdf")
    
    # Plot g_infty as func of potential
    fig, ax = plt.subplots()
    plot_ginfty(ax, circuit, 'M1', potential_vec=True, steady_state_point=steady_state_point, change_func=change_func)
    plot_ginfty(ax, circuit, 'M2', potential_vec=True, steady_state_point=steady_state_point)
    # ax.legend(fontsize = par.legend_size)
    fig.tight_layout()
    plt.savefig(f"../plots/importance_ginfinity_dynamics/plots/g_infities_{phase}.pdf")
   
def plot_analysis_relaxation_ingrid(circuit, potentials_vec, resistances_vec, phase):
    
    change_func = False
    if phase == 'trained_change_func':
        change_func=True
    
    deltav1 = potentials_vec['tran']['VN0'] - potentials_vec['tran']['VN1']
    delltav2 = potentials_vec['tran']['VN1'] - potentials_vec['tran']['VN2']
    
    steady_state_point = [ [deltav1[-1], 1/resistances_vec[-1][0]], [delltav2[-1], 1/resistances_vec[-1][1]] ]
    
    ##### PLOT GRID FIGURE ####
    
    fig = plt.figure(figsize=(10, 7))
    gs = gridspec.GridSpec(4, 4)

    ax_left = fig.add_subplot(gs[1:3, 0:2])   
    ax_top = fig.add_subplot(gs[0:2, 2:4])    
    ax_bottom = fig.add_subplot(gs[2:4, 2:4]) 
    
    # Plot conductances
    plot_ginfty(ax_top, circuit, 'M1', result = potentials_vec, change_func=change_func)
    plot_ginfty(ax_top, circuit, 'M2', result = potentials_vec)
    plot_conductances(ax_top, resistances_vec, potentials_vec)
    
    # Plot potential  
    plot_potentialdiff(ax_bottom, potentials_vec)
    
    # Plot g_infty as func of potential
    plot_ginfty(ax_left, circuit, 'M1', potential_vec=True, steady_state_point=steady_state_point, change_func=change_func)
    plot_ginfty(ax_left, circuit, 'M2', potential_vec=True, steady_state_point=steady_state_point)
    
    handles_top, legend_top = ax_top.get_legend_handles_labels()
    handles_top, legend_top = [handles_top[2], handles_top[4]], [legend_top[2], legend_top[4]]
    ax_top.get_legend().remove()
    
    handles_left, legend_left = ax_left.get_legend_handles_labels()
    ax_left.get_legend().remove()
    
    handles_bottom, legend_bottom = ax_bottom.get_legend_handles_labels()
    ax_bottom.get_legend().remove()
    
    fig.legend(handles_left, legend_left, bbox_to_anchor = (0.46,0.93), ncol = 2)
    #fig.legend(handles_top, legend_top, bbox_to_anchor = (0.95,0.8), ncol = 1)
    ax_top.legend(handles_top, legend_top, loc = "best")
    
    fig.legend(handles_bottom, legend_bottom, bbox_to_anchor = (0.46,0.20), ncol = 4)
    
    fig.tight_layout()
    plt.savefig(f"../plots/importance_ginfinity_dynamics/plots/grid_{phase}.pdf")
   
   
# -> DEFINE graph from networks module
G = networks.voltage_divider(save_data=True) 

circuit = networks.circuit_from_graph(G, type='memristors') 

# ANALYSIS IN INITIAL CONDITION
tran_analysis = ahkab.new_tran(tstart=0, tstop=0.1, tstep=1e-3, x0=None, use_step_control=False)
result = ahkab.run(circuit, an_list=tran_analysis) 
potentials_vec = result[0]
resistances_vec = result[1]

plot_analysis_relaxation(circuit, potentials_vec, resistances_vec, 'initial_condition')

plot_analysis_relaxation_ingrid(circuit, potentials_vec, resistances_vec, 'initial_condition')

# ANALYSIS WHEN TRAINED FOR LENGTH

G_trained = nx.read_graphml('../plots/importance_ginfinity_dynamics/Gtrained.graphml')

circuit_trained = networks.circuit_from_graph(G_trained, type='memristors') 

tran_analysis = ahkab.new_tran(tstart=0, tstop=0.02, tstep=1e-3, x0=None, use_step_control=False)
result = ahkab.run(circuit_trained, an_list=tran_analysis) 
potentials_vec = result[0]
resistances_vec = result[1]
resistances_vec_trained = resistances_vec

plot_analysis_relaxation(circuit_trained, potentials_vec, resistances_vec, 'trained')

plot_analysis_relaxation_ingrid(circuit_trained, potentials_vec, resistances_vec, 'trained')

# ANALYSIS WHEN TRAINED FOR LENGTH WITH DIFFERENT GINFINITY

circuit_trained = networks.circuit_from_graph(G_trained, type='memristors') 

tran_analysis = ahkab.new_tran(tstart=0, tstop=0.02, tstep=1e-4, x0=None, change_func = True, use_step_control=False)
result = ahkab.run(circuit_trained, an_list=tran_analysis) 
potentials_vec = result[0]
resistances_vec = result[1]

plot_analysis_relaxation(circuit_trained, potentials_vec, resistances_vec, 'trained_change_func')

plot_analysis_relaxation_ingrid(circuit_trained, potentials_vec, resistances_vec, 'trained_change_func')

# ANALYSIS WHEN TRAINED ginfinity INITIAL CONDITION FOR LENGTH 

G_updated_initial_cond = G.copy(as_view=False)
for edge_index, edge in enumerate(G_updated_initial_cond.edges()):
    G_updated_initial_cond.edges[edge]["resistance"] = resistances_vec_trained[-1][edge_index]
    print(resistances_vec[-1][edge_index])

circuit_updated_initial_cond = networks.circuit_from_graph(G_updated_initial_cond, type='memristors') 

tran_analysis = ahkab.new_tran(tstart=0, tstop=0.02, tstep=1e-4, x0=None, change_func = True, use_step_control=False)
result = ahkab.run(circuit_updated_initial_cond, an_list=tran_analysis) 
potentials_vec = result[0]
resistances_vec = result[1]

plot_analysis_relaxation(circuit_updated_initial_cond, potentials_vec, resistances_vec, 'updated_initial_cond_func')

# PLOTTING ginfinity(delta P) for prositive and negative potential 

# pressure_vec = np.linspace(-1, 1)
# mysistor = ahkab.Circuit.get_elem_by_name(circuit, part_id='M1')
# g_infinity_positive = []
# g_infinity_negative = []
# for pressure in pressure_vec:
#     g_infinity_value = ahkab.transient.g_infinity_func(0.4, pressure, 0, mysistor)*mysistor.g_0
#     g_infinity_positive.append(g_infinity_value)
#     g_infinity_value = ahkab.transient.g_infinity_func(-0.4, pressure, 0, mysistor)*mysistor.g_0
#     g_infinity_positive.append(g_infinity_value)

# fig, ax = plt.subplots()
# ax.plot(pressure_vec, g_infinity)

# fig.tight_layout()
# plt.savefig(f"../plots/importance_ginfinity_dynamics/plots/gininifty_pressure.pdf")
    