import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
G = nx.Graph()
G.add_node(1)
# G.add_nodes_from(["B","C","D"])
G.add_edge(1, 2)
G.add_edges_from([(1,3), (1,2), (3,4)])
pos = nx.circular_layout(G)
nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=12)
plt.show()
# Read the stdout of the .cu program or save to a file, run it as a bash script