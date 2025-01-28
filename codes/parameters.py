# --------- PATHS ---------

DATA_PATH = '../data/'
PLOT_PATH = '../plots/'
PACKAGE_PATH = '/home/monicaconte/nica/phd/Projects/Learning_with_memristors/Learning_Neural_Networks/codes/ahkab'
# PACKAGE_PATH = '/Users/monicaconte/PhD/Projects/Contrastive_Learning_Memristors/Learning_Neural_Networks_true/codes/ahkab'

# --------- NETWORKS PLOTS STYLES ---------

color_dots = ['forestgreen','darkslateblue','silver'] # color of the three types of nodes: input, output, hidden
font_size_nw = 12 #size font on nodes
nodes_size = 3000 #size nodes
width = 5 #width connecting edges

color_edges = 'lightblue'
color_font_edges = 'white'

# --------- PLOTS STYLES ---------

figsize_2horizontal = (11,5)
figsize_3horizontal = (13,5)
figsize_4horizontal = (13,5)
figsize_1 = (4.4,4)

legend_size = 15
axis_fontsize = 19
size_ticks = 20
marker_size = 10

# -> MSE style

from pypalettes import load_cmap
cmap = load_cmap("Callanthias_australis")

mse_styles = {
    'allostery_length': dict(marker = 'o', c = cmap.colors[0], lw=1, label = rf'$L$', ylabel_weights = r'$L$[$\mu$ m]'),
    'allostery_rho': dict(marker = 'o', c = cmap.colors[1], lw=1, label = rf'$\rho$', ylabel_weights = r'$\rho$[mM]'),
    'allostery_rbrt': dict(marker = 'o', c = cmap.colors[2], lw=1, label = rf'$R$', ylabel_weights = 'R'),
    'allostery_pressure': dict(marker = 'o', c = cmap.colors[3], lw=1, label = rf'$P$', ylabel_weights = r'$P$[bar]'),
}


