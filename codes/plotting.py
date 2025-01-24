import parameters as par
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(par.PACKAGE_PATH)
import ahkab  
import networks

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
    pos = nx.kamada_kawai_layout(G, scale=2)
    # pos = nx.arf_layout(G)

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

def plot_mse(ax, fig, rule):
    
    y = np.loadtxt(f"{par.DATA_PATH}mse/mse_{rule}.txt", unpack=True)
    precision_check = y[1]
    y = y[0]
    print(y)
    x = range(0,len(y))

    for index in x:
        if precision_check[index]==3:
            xprecision_check = index
            yprecision_check = y[index]
            break

    ax.semilogy(x, y, **par.mse_styles[f'{rule}'])   
    ax.scatter(xprecision_check, yprecision_check)  
    
    # ax.legend(fontsize = par.legend_size)
    ax.set_ylabel(r'Error', fontsize = par.axis_fontsize)
    ax.set_xlabel(r'Training steps', fontsize = par.axis_fontsize)
    ax.tick_params(axis='both', labelsize=par.size_ticks)

import colormaps
from pypalettes import load_cmap

def plot_weights(ax, G, training_steps, training_job, weight_type):

    x =list(range(training_steps+1))

    if weight_type=='rho':
        color_map = colormaps.rdpu
        number_weights = G.number_of_nodes()
        label = r'$\rho$[mM]'
        label_legend = r'\rho'

    if weight_type=='length':
        color_map =load_cmap('green_material')
        number_weights = G.number_of_edges()
        label = r'$L$[$\mu$ m]'
        label_legend = r'L'


    if weight_type=='rbrt':
        color_map = colormaps.greys
        number_weights = G.number_of_edges()
        label = r'$R$'
        label_legend = r'R'


    if weight_type=='pressure':
        color_map = load_cmap('Purp')
        number_weights = G.number_of_nodes()
        label = r'$P$[bar]'
        label_legend = r'P'
        
    start_index = int((1/3)*len(color_map.colors))
    color_map_colors = color_map.colors[start_index:]
    color_map = plt.cm.colors.ListedColormap(color_map_colors)
      
    for weight_indx in range(number_weights):

        weight = []
        for step in range(training_steps+1):
            data = np.loadtxt(f"{par.DATA_PATH}weights/{training_job}/{weight_type}/{weight_type}{step}.txt", unpack=True)
            y = data[1]
            weight.append(y[weight_indx])

        norm = plt.Normalize(0, y.shape[0] - 1)
        color_index = norm(weight_indx)

        ax.plot(x, weight, color=color_map(color_index), lw=3, label = f'${label_legend}_{weight_indx+1}$')
        ax.set_ylabel(f'{label}', fontsize = par.axis_fontsize)
        ax.set_xlabel(r'Training steps', fontsize = par.axis_fontsize)
        ax.tick_params(axis='both', labelsize=par.size_ticks)
        # ax.legend()
        # ax.legend(bbox_to_anchor=(1, 1))

def plot_potential_drops_each_node(ax, G):

    circuit = networks.circuit_from_graph(G, type='memristors')

    tran_analysis = ahkab.new_tran(tstart=0, tstop=0.1, tstep=1e-3, x0=None)
    result = ahkab.run(circuit, an_list=tran_analysis)  
    result = result[0]
    print(result['tran'].keys())


    index_input = 0
    index_target = 0
    index_hidden = 0
    for node in G.nodes():

        potential = result['tran'][f'VN{node}']

        # print(node)

        if G.nodes[node]['type'] == 'source':
            ax.plot(result['tran']['T'], potential, lw=3, color = 'forestgreen', label = fr"$V_{int(node)+1}$")
            index_input += 1
            
        if G.nodes[node]['type'] == 'target':
            ax.plot(result['tran']['T'], potential, lw=3, color = 'darkslateblue', label = fr"$V_{int(node)+1}$")
            index_target += 1
            print(f'Final potential node {int(node)+1} = ', potential[-1])
            
        if G.nodes[node]['type'] == 'hidden':
            ax.plot(result['tran']['T'], potential, lw=3, color = 'darkgray', label = fr"$V_{int(node)+1}$")
            index_hidden += 1


    # plot desired voltages lines

    # for desired_nodes_index in range(inp.numb_target_nodes):

    # ax.axhline(4, color='mediumpurple', xmin=0, xmax=0.1, lw=1.5, ls='--', label=r'$V_D$')
    ax.plot((0, 0.1), (4, 4), color='mediumpurple', lw=1.5, ls='--', label=r'$V_D$')

    ax.margins(x=0)
    ax.legend()
    ax.set_ylabel(r'$V[V]$', fontsize = par.axis_fontsize)
    ax.set_xlabel(r't[s]', fontsize = par.axis_fontsize)
    ax.tick_params(axis='both', labelsize=par.size_ticks)
