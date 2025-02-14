import parameters as par
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(par.PACKAGE_PATH)
import ahkab  
import networks
import training

from matplotlib.colors import to_rgb, to_hex
def lighten_color(base_color, factor=0.5):
    """
    Lightens a given color by blending it with a lighter shade of itself.
    
    Parameters:
    - base_color: A HEX color or an (R, G, B) tuple in normalized [0, 1] range.
    - factor: Float, how much to lighten the color (0 = no change, 1 = fully white).
    
    Returns:
    - Lightened color as a HEX string.
    """
    base = to_rgb(base_color)  # Convert to RGB if it's a HEX string
    lightened = [(1 - factor) * c + factor for c in base]  # Blend toward lighter version of itself
    return to_hex(lightened)

def plot_graph(name_graph):

    # GET graph data
    G = nx.read_graphml(f'{par.DATA_PATH}{name_graph}.graphml')
    
    # GET color of node depending on which type of node it is
    color_attributes = [G.nodes[node]['color'] for node in G.nodes()]
    
    # LABEL nodes with potential, pressure and densty indeces
    labels = {node: fr'$V_{int(node)}$, $P_{int(node)}$, $\rho_{int(node)}$' for node in G.nodes()}
    
    # LABEL edges M_{i,j} : memristor that connect node i to node j (i = BASE, j = TIP)
    edge_labels = {(u, v): fr'$M_{{ {int(u)}, {int(v)} }}$' for u, v in G.edges()}  
    
    # CREATE a box on the edge that represents the memristor
    bbox = {'facecolor': 'lightblue', 'edgecolor': par.color_edges, 'alpha': 1, 'boxstyle': 'round,pad=0.5'}
    
    # GENERATE determined positions - Check out better with general nw
    # pos = nx.kamada_kawai_layout(G, scale=2)
    pos = nx.spring_layout(G)

    # DRAW the network
    nx.draw(G, with_labels=True, node_color=color_attributes,  
            pos = pos,
            node_size=par.nodes_size,
            labels = labels,
            width = par.width,
            font_size=par.font_size_nw,
            edge_color=par.color_edges)
    # adding the memristors on the edges
    nx.draw_networkx_edge_labels(G, bbox=bbox, pos=pos, edge_labels=edge_labels, font_size=par.font_size_nw, font_color=par.color_font_edges)

    return  

def plot_mse(ax, fig, rule, show_xlabel=True, saved_graph=None):
    
    if saved_graph==None:
        y = np.loadtxt(f"{par.DATA_PATH}mse/mse_{rule}.txt", unpack=True)
    else:
        y = np.loadtxt(f"{par.DATA_PATH}mse/{saved_graph}/mse_{rule}.txt", unpack=True)

    x = range(0,len(y))

    style = par.mse_styles[f'{rule}']
    ax.plot(x, y, color = style['c'], marker = style['marker'], lw = style['lw'], label = style['label'])
    ax.set_yscale('log')   
    
    # ax.legend(fontsize = par.legend_size)
    ax.set_ylabel(r'$C(w)$', fontsize = par.axis_fontsize)
    if show_xlabel:
        ax.set_xlabel(r'Training steps', fontsize = par.axis_fontsize)
    ax.tick_params(axis='both', labelsize=par.size_ticks)

import colormaps
from pypalettes import load_cmap

def plot_weights(ax, G, training_steps, rule, show_xlabel=True, starting_step = 0):

    x =list(range(starting_step, starting_step + training_steps+1))
    
    text = rule.split('_')
    training_job = text[0]
    weight_type = text[1]
    if weight_type == 'radius':
        weight_type = 'radius_base'

    if weight_type == 'pressure' or weight_type == 'rho':
        number_weights = G.number_of_nodes()
    else:
        number_weights = G.number_of_edges()
        
    # CREATE color palettes with lighter shades of base color (the color of the mse)
    if number_weights == 2:
        print(weight_type)
        color_factor = 0.6
    elif number_weights == 3:
        color_factor = 0.4
    else:
        color_factor = 1 / (number_weights - 1)
    
    style = par.mse_styles[f'{rule}']
    base_color = style['c']
    palette = [lighten_color(base_color, factor = i * color_factor) for i in range(number_weights)]
      
    # GET data: data/training_job/weight_type contains files weight_type{step} with the list of weights per step
    for weight_indx in range(number_weights):

        weight = []
        for step in range(training_steps+1):
            data = np.loadtxt(f"{par.DATA_PATH}weights/{training_job}/{weight_type}/{weight_type}{step}.txt", unpack=True)
            y = data[1]
            weight.append(y[weight_indx])

        label_without_weightindex = style['label'][1:-1]
        ax.plot(x, weight, color=palette[weight_indx], marker = style['marker'], lw = style['lw'], label = rf'${{{label_without_weightindex}}}_{weight_indx+1}$')
        
    label = style['ylabel_weights']
    ax.set_ylabel(f'{label}', fontsize = par.axis_fontsize)
    if show_xlabel:
        ax.set_xlabel(r'Training steps', fontsize = par.axis_fontsize)
    ax.tick_params(axis='both', labelsize=par.size_ticks)
    if weight_type == 'pressure' and starting_step==0:
        ax.legend(fontsize = par.legend_size, loc='lower left')
    elif starting_step ==0 :
        ax.legend(fontsize = par.legend_size)
        # ax.legend(bbox_to_anchor=(1, 1))

def plot_potential_each_node(ax, G, factor_time=1):

    circuit = networks.circuit_from_graph(G, type='memristors')

    tran_analysis = ahkab.new_tran(tstart=0, tstop=0.1, tstep=1e-3, x0=None)
    result = ahkab.run(circuit, an_list=tran_analysis)  
    result = result[0]
    print(result['tran'].keys())

    time = result['tran']['T']  

    index_input = 0
    index_target = 0
    index_hidden = 0
    for node in G.nodes():

        potential = result['tran'][f'VN{node}']

        if G.nodes[node]['type'] == 'source':
            ax.plot(time, potential, lw=3, color = par.color_dots[0], label = fr"$V_{int(node)+1}$")
            index_input += 1
            
        if G.nodes[node]['type'] == 'target':
            desired_value = G.nodes[node]['desired']
            ax.plot(time, potential, lw=1, marker = 'o', color = par.color_dots[1], label = fr"$V_{int(node)+1}$")
            index_target += 1
            print(f'Final potential node {int(node)+1} = ', potential[-1])
            
        if G.nodes[node]['type'] == 'hidden':
            ax.plot(time, potential, lw=3, color = par.color_dots[2], label = fr"$V_{int(node)+1}$")
            index_hidden += 1


    ax.plot((time[0], time[-1]), (desired_value, desired_value), color='mediumpurple', lw=1.5, ls='--', label=r'$V_D$')

    ax.margins(x=0)
    ax.legend()
    ax.set_ylabel(r'$V[V]$', fontsize = par.axis_fontsize)
    ax.set_xlabel(r't[s]', fontsize = par.axis_fontsize)
    ax.tick_params(axis='both', labelsize=par.size_ticks)

def plot_final_potential_vd(ax, target_values):
    
    for target_index, target_value in enumerate(target_values):

        y = np.loadtxt(f"{par.DATA_PATH}potential_targets/potential_targets{target_value}.txt", unpack=True)
        x = np.array(range(len(y))) + target_index*(len(y)-1)

        # PLOT 

        # Target
        ax.plot(x, y, lw=1, marker = 'o', color = par.color_dots[1], label="$V_2$" if target_index == 0 else "_nolegend_", zorder = 1)        
        # Inputs
        ax.plot(x, [5 for _ in range(len(x))], lw=3, color = par.color_dots[0], label="$V_1$" if target_index == 0 else "_nolegend_")
        ax.plot(x, [0 for _ in range(len(x))], lw=3, color = par.color_dots[0], label = "$V_3$" if target_index == 0 else "_nolegend_")
        # Desired
        ax.plot(x, [target_value for _ in range(len(x))],lw=1.5, ls='--', color = 'mediumpurple', label = "$V_D$" if target_index == 0 else "_nolegend_", zorder = 0)

    ax.set_ylabel(r'$V[V]$', fontsize = par.axis_fontsize)
    # ax.set_xlabel(r'Time steps', fontsize = par.axis_fontsize)
    ax.tick_params(axis='both', labelsize=par.size_ticks)

def plot_regression(ax, step):
    
    data = np.loadtxt(f"{par.DATA_PATH}relations_regression/relations_regression{step}.txt", unpack=True)
    
    x = data[0]
    y1 = data[1]
    
    x_interval = np.linspace(np.min(x),np.max(x))

    y1_desired = training.linear_function(x_interval)

    
    # if inp.weight_type=='rhob':
    ax.plot(x_interval, y1_desired, **par.reg_desired, zorder=0)
    # ax.plot(x_interval, y2_desired, **par.reg_desired2, zorder=0)
    
    ax.scatter(x, y1, **par.reg_output)
    # ax.scatter(x, y2, **par.reg_output2)
    
    # ax.legend()
    ax.set_ylabel(r'$V_{out}$', fontsize = par.axis_fontsize)
    ax.set_xlabel(r'$V_{in}$', fontsize = par.axis_fontsize)
    # ax.set_title(f"Step {step}")
    
    # ax.set_xlim([-0.2, 5.2])
    # ax.set_ylim([-0.2, 2])
    ax.grid(ls=":")
    ax.tick_params(axis='both', labelsize=par.size_ticks)
    
    return


    

