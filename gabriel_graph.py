import matplotlib.pyplot as plt
import networkx as nx
import math
import json


G = nx.Graph()

with open('pos1.txt', 'r') as filehandle:
    pos_list = json.load(filehandle)

xy = []
for ps in pos_list:
    xy.append(tuple(ps))

print('xy:', xy)

with open('edges1.txt', 'r') as filehandle:
    edges_list = json.load(filehandle)

list_unweighted_edges = []
for ed in edges_list:
    list_unweighted_edges.append(tuple(ed))

for i in range(len(xy)):
	G.add_node(i, pos=xy[i])

p = []
for node in sorted(G):
    p.append(G.nodes[node]['pos'])

print('p:', p)

for u, v in list_unweighted_edges:
    distance = math.sqrt(math.pow((p[u][0] - p[v][0]), 2) + math.pow((p[u][1] - p[v][1]), 2))
    G.add_edge(u, v, weight = math.ceil(distance))


node_pos = nx.get_node_attributes(G, 'pos')
edge_weight = nx.get_edge_attributes(G, 'weight')

node_col = ['green']
edge_col = ['black']

# Draw the nodes
nx.draw_networkx(G, node_pos, node_color=node_col, node_size=10)
# Draw the node labels
#nx.draw_networkx_labels(G, node_pos, node_color=node_col)
# Draw the edges
nx.draw_networkx_edges(G, node_pos, edge_color=edge_col)
# Draw the edge labels
#nx.draw_networkx_edge_labels(G, node_pos, edge_color=edge_col, edge_labels=edge_weight)
# Show the axis
plt.axis('off')
# Show the plot
plt.show()