# --------- PATHS ---------

DATA_PATH = '../data/'
PLOT_PATH = '../plots/'
PACKAGE_PATH = '/home/monicaconte/nica/phd/Projects/Learning_with_memristors/Learning_Neural_Networks/codes/ahkab'
# PACKAGE_PATH = '/Users/monicaconte/PhD/Projects/Contrastive_Learning_Memristors/Learning_Neural_Networks_true/codes/ahkab'

# --------- NETWORKS PLOTS STYLES ---------

color_dots = ['forestgreen','blueviolet','silver'] # color of the three types of nodes: input, output, hidden
font_size_nw = 12 #size font on nodes
nodes_size = 3000 #size nodes
width = 5 #width connecting edges

color_edges = 'lightblue'
color_font_edges = 'white'

# --------- PLOTS STYLES ---------

figsize_2horizontal = (11,5)
figsize_3horizontal = (13,5)
figsize_1 = (4.4,4)

legend_size = 15
axis_fontsize = 19
size_ticks = 20
marker_size = 10


mse_styles = {
    'allostery_length': dict(c = "mediumpurple", lw=3, label = rf'$L$'),
    'allostery_rho': dict(c = "plum", lw=3, label = rf'$\rho$'),
    'allostery_rbrt': dict(c = "darkgray", lw=3, label = rf'$R_b/R_t$'),
    'allostery_pressure': dict(c = 'mediumpurple', lw=3, label = rf'$P$')
}




