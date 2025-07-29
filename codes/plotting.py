import parameters as par
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import ahkab  
import networks
import training
import re


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

def plot_graph(G, weight_type = None):

    # GET graph data
    # G = nx.read_graphml(f'{par.DATA_PATH}{name_graph}.graphml')
    
    # GET color of node depending on which type of node it is
    color_attributes = [G.nodes[node]['color'] for node in G.nodes()]
    
    # LABEL nodes with potential, pressure and densty indeces
    labels = {node: fr'$V_{{{int(node)+1}}}$' for node in G.nodes()}
    
    # LABEL edges M_{i,j} : memristor that connect node i to node j (i = BASE, j = TIP)
    # edge_labels = {(u, v): fr'$M_{{ {int(u)}, {int(v)} }}$' for u, v in G.edges()}  
    edge_labels = {(u, v): fr'$M_{{{index+1}}}$' for index, (u, v) in enumerate(G.edges())}
    
    # CREATE a box on the edge that represents the memristor
    bbox = {'facecolor': 'lightblue', 'edgecolor': par.color_edges, 'alpha': 1, 'boxstyle': 'round,pad=0.5'}
    
    # GENERATE determined positions - Check out better with general nw
    # pos = nx.kamada_kawai_layout(G, scale=2)
    pos = nx.spring_layout(G)

    edge_weights = [G[u][v].get(f'{weight_type}', 1) for u, v in G.edges()]  # Default weight = 1 if missing
    max_weight = max(edge_weights) if edge_weights else 1
    widths = [2 + 10 * (w / max_weight) for w in edge_weights]  # Normalize and scale widths

    # DRAW the network
    nx.draw(G, with_labels=True, node_color=color_attributes,  
            pos = pos,
            node_size=par.nodes_size,
            labels = labels,
            width = widths,
            font_size=par.font_size_nw,
            edge_color=par.color_edges)
    # adding the memristors on the edges
    nx.draw_networkx_edge_labels(G, bbox=bbox, pos=pos, edge_labels=edge_labels, font_size=par.font_size_nw, font_color=par.color_font_edges)

    return  

def plot_mse(ax, fig, graph_id, training_type, weight_type, show_xlabel=True):
    
    data = np.loadtxt(f"{par.DATA_PATH}{training_type}{graph_id}/mse/mse_{weight_type}.txt", unpack=True)
    choosen_weight = np.loadtxt(f"{par.DATA_PATH}{training_type}{graph_id}/weights/best_choice/choosen_weights.txt", unpack=True)

    x = data[0]
    x = x[::5]
    y = data[1]
    y = y[::5]
    choosen_weight = choosen_weight[1]

    if weight_type == 'best_choice':
        color_vec = ['#F2CB05FF']
        possible_weights = ['length', 'radius_base', 'rho', 'pressure']
        
        for point in choosen_weight:
            point = int(point)
            # weight_type_par = weight_type[choosen_weight]
            style = par.weight_styles[f'{possible_weights[point]}']
            color_vec.append(style['c'])
        # print(color_vec)
        color_vec = color_vec[::5]
        ax.scatter(x, y, color = color_vec, marker = '^', s = 50, lw = style['lw'], label = 'best choice', zorder=2)
        
    else:
        style = par.weight_styles[f'{weight_type}']
        ax.plot(x, y, color = style['c'], marker = style['marker'], lw = style['lw'], label = style['label'], zorder=1)
    ax.set_yscale('log')   
    
    # ax.legend(fontsize = par.legend_size)
    if training_type=='regression':
        ax.set_ylabel(r'$C_{test}(w)$', fontsize = par.axis_fontsize)
    else:
        ax.set_ylabel(r'$C(w)$', fontsize = par.axis_fontsize)

    if show_xlabel:
        ax.set_xlabel(r'Training steps', fontsize = par.axis_fontsize)
    ax.tick_params(axis='both', labelsize=par.size_ticks)

import colormaps
from pypalettes import load_cmap

def plot_weights(ax, G, training_steps, training_type, weight_type, show_xlabel=True, starting_step = 0):

    x =list(range(starting_step, starting_step + training_steps+1))
    
    if weight_type == 'pressure' or weight_type == 'rho':
        number_weights = G.number_of_nodes()
    else:
        number_weights = G.number_of_edges()
        
    # CREATE color palettes with lighter shades of base color (the color of the mse)
    if number_weights == 2:
        color_factor = 0.6
    elif number_weights == 3:
        color_factor = 0.4
    else:
        color_factor = 1 / (number_weights - 1)
    
    style = par.weight_styles[f'{weight_type}']
    base_color = style['c']
    palette = [lighten_color(base_color, factor = i * color_factor) for i in range(number_weights)]
    # palette = plt.get_cmap('tab20')
      
    # GET data: data/training_job/weight_type contains files weight_type{step} with the list of weights per step
    for weight_indx in range(number_weights):

        weight = []
        for step in range(training_steps+1):
            data = np.loadtxt(f"{par.DATA_PATH}{training_type}{G.graph['name']}/weights/{weight_type}/{weight_type}{step}.txt", unpack=True)
            y = data[1]
            weight.append(y[weight_indx])

        label_without_weightindex = style['label'][1:-1]
        ax.plot(x, weight, color=palette[weight_indx], marker = style['marker'], lw = style['lw'], label = rf'${{{label_without_weightindex}}}_{{{weight_indx+1}}}$')
        # ax.plot(x, weight, color=palette(weight_indx), marker = style['marker'], lw = style['lw'], label = rf'${{{label_without_weightindex}}}_{{{weight_indx+1}}}$')
        # ax.plot(x, weight, marker = style['marker'], lw = style['lw'], label = rf'${{{label_without_weightindex}}}_{{{weight_indx+1}}}$')
        
    label = style['ylabel_weights']
    ax.set_ylabel(f'{label}', fontsize = par.axis_fontsize)
    if show_xlabel:
        ax.set_xlabel(r'Training steps', fontsize = par.axis_fontsize)
    ax.tick_params(axis='both', labelsize=par.size_ticks)
    # if weight_type == 'pressure' and starting_step==0:
    #     ax.legend(fontsize = par.legend_size, loc='lower left')
    # elif starting_step ==0 :
    ax.legend(fontsize = par.legend_size)
        # ax.legend(bbox_to_anchor=(1, 1))

def plot_memristor_resistances(ax, G):

    circuit = networks.circuit_from_graph(G, type='memristors') 
    analysis = ahkab.new_tran(tstart=0, tstop=0.1, tstep=1e-3, x0=None)

    # DEFINE a transient analysis (analysis of the circuit over time)
    result = ahkab.run(circuit, an_list=analysis) #returns two arrays: resistance over time of the memristors, voltages over time in the nodes
    resistances = result[1]
    
    # print('Resistanes', len(resistances), len(resistances[0]))
    x = np.array(range(len(resistances)))

    for edge_index in range(len(G.edges())):
        y = [1/(resistances[time_index][edge_index]) for time_index in range(len(resistances))]
        ax.plot(x, y, **par.memr_resistances_style, label = rf'$M_{{{edge_index+1}}}$') 
    # ax.legend()

def plot_potential_each_node(ax, G, factor_time=1):

    circuit = networks.circuit_from_graph(G, type='memristors')

    tran_analysis = ahkab.new_tran(tstart=0, tstop=0.1, tstep=1e-3, x0=None)
    result = ahkab.run(circuit, an_list=tran_analysis)  
    resistances = result[1][-1]
    # print(1/resistances[0], 1/resistances[1])

    result = result[0]
    # print(result['tran'].keys())

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

def plot_regression(ax, graph_id, weight_type, step):
    
    data = np.loadtxt(f"{par.DATA_PATH}regression{graph_id}/relations_regression/relations_regression{weight_type}{step}.txt", unpack=True)
    x = data[0]
    y1 = data[1]
    
    x_interval = np.linspace(np.min(x),np.max(x))
    y1_desired = training.regression_function(x_interval)

    style_des = par.regression_styles[f'{weight_type}_des']
    style = par.regression_styles[f'{weight_type}']

    
    ax.plot(x_interval, y1_desired, **style_des, zorder=0)
    
    ax.scatter(x, y1, **style)
    
    ax.set_ylabel(r'$V_{out}$', fontsize = (par.axis_fontsize-7))
    ax.set_xlabel(r'$V_{in}$', fontsize = (par.axis_fontsize-7))
    
    ax.grid(ls=":")
    ax.tick_params(axis='both', labelsize=par.size_ticks)
    
    return


# PER MOSTRARE COMPORTAMENTO
# 
# 
voltage_style = [dict(c = 'teal', lw=2, label=r'$V_{1}$'),
                 dict(c = 'mediumaquamarine', lw=2, label=r'$V_{2}$'),
                 dict(c = 'limegreen', lw=2, label=r'$V_{3}$'),
                 dict(c = 'darkolivegreen', lw=2, label=r'$V_{4}$')]
conductance_style = [dict(c = 'navy', lw=2.3, label=r'$g_1$'),
                     dict(c = 'slateblue', lw=2.3, label=r'$g_2$')]
voltage_drop_style = [dict(c = 'darkorange', lw=2, label=r'$\Delta V_{C1}$'),
                 dict(c = 'mediumorchid', lw=2, label=r'$\Delta V_{M1}$'),
                 dict(c = 'forestgreen', lw=2, label=r'$\Delta V_{C2}$'),
                 dict(c = 'blueviolet', lw=2, label=r'$\Delta V_{M2}$')]
g_infinity_style = [dict(c = 'skyblue', lw=3, label=r'$g_{\infty,1}$'),
                    dict(c = 'plum', lw=3, label=r'$g_{\infty,2}$')]

def plot_vd(ax, circuit, result, type, voltage_dep=False, 
         element_vdrop=False, mc_constant=False, g_infty_plot=False):
    ax.grid(ls=':')
    ax.tick_params('y')
    ax.tick_params('x')


    # Plot voltage at given node "VN.."

    if type[0]=='V':

        style_index = int(re.findall(r'\d+', type)[0]) - 1
        ax.plot(result['tran']['T'], result['tran'][f"{type}"], **voltage_style[style_index])
        ax.legend()
        ax.set_xlabel(r't[s]')
        ax.set_ylabel(r'V[V]')

        return
    
    elem = ahkab.circuit.Circuit.get_elem_by_name(circuit, type)
    n1 = elem.n1
    n2 = elem.n2

    current = result['tran']['I(VN0)']
    voltage_in_node1 = 'VN'+f'{n1}'
    voltage_in_node2 = 'VN'+f'{n2}'
    if all(voltage_in_node1 != strings for strings in result['tran'].keys()): #coul just be n!=0
        voltage_in_node1 = np.array([0]*len(result['tran']['T']))
    else:
        voltage_in_node1 = result['tran'][voltage_in_node1]
    if all(voltage_in_node2 != strings for strings in result['tran'].keys()):
        voltage_in_node2 = np.array([0]*len(result['tran']['T']))
    else:
        voltage_in_node2 = result['tran'][voltage_in_node2]

    voltage_diff = voltage_in_node1-voltage_in_node2

    # Plot voltage drops across element    
    if element_vdrop:
        style_index=0
        if type=="C1":
            style_index=0
        if type=="M1" or type=='R1':
            style_index=1
        if type=="C2":
            style_index=2
        if type=="M2" or type=='R2':
            style_index=3

        ax.plot(result['tran']['T'], voltage_diff, **voltage_drop_style[style_index])

        ax.legend()
        ax.set_xlabel(r't[s]')
        ax.set_ylabel(r'V[V]')
        return

    #  Plot conductance in memristors
    
    conductance = (-1)*current/voltage_diff

    if mc_constant:

        elem = ahkab.circuit.Circuit.get_elem_by_name(circuit, 'C1')
        capacitance = elem.value

        ax.plot(result['tran']['T'], conductance*capacitance, **conductance_style[0])

        return

    style_index = int(re.findall(r'\d+', type)[0]) - 1
    if voltage_dep==True:

        ax.plot(voltage_diff, conductance, **conductance_style[style_index])

    else:
        
        start = 0
        stop = -10

        time = result['tran']['T']
        time = time[start:stop]
        conductance = 1/conductance[start:stop]
        print(conductance[-1])
        ax.set_xlabel(r't[s]')
        ax.set_ylabel(r'g[pS]')
        if g_infty_plot:
            g_infty = elem.g_0*ahkab.transient.sigmoid(voltage_diff)
            g_infty = []
            for index in range(len(voltage_diff)):
                g_infty.append( 1/(elem.g_0*ahkab.transient.g_infinity_func(voltage_diff[index], 0, 0, elem)))
            g_infty=g_infty[start:stop]
            ax.plot(time, g_infty, **g_infinity_style[style_index])

        ax.plot(time, conductance, **conductance_style[style_index])

            # ax.plot(result['tran']['T'], elem.g_0*ahkab.transient.g_infinity_func(voltage_diff, elem), **g_infinity_style)

    ax.legend()

    return 
   

