
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import networkx.algorithms.isomorphism as iso
import numpy as np

# generate random graph
G = nx.generators.fast_gnp_random_graph(10, 0.4)
# check planarity and draw the graph
#nx.is_planar(G)
is_planar, P = nx.check_planarity(G)
print(is_planar)
print(G.edges)
nodePos = nx.circular_layout(G)
#pos = graphviz_layout(G, prog='neato')

print(nodePos)
#nx.draw(G)
nx.draw_networkx(G, with_labels=True, pos= nodePos)
plt.show()
