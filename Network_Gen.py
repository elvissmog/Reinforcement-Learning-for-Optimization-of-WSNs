import networkx as nx
from scipy.spatial.distance import pdist, squareform
import math
import json
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


txr = 150   #Transmission radius

G = nx.Graph()

with open('pos.txt', 'r') as filehandle:
    pos_list = json.load(filehandle)

xy = []
for ps in pos_list:
    xy.append(tuple(ps))

print('xy:', xy)

with open('edges.txt', 'r') as filehandle:
    edges_list = json.load(filehandle)

list_unweighted_edges = []
for ed in edges_list:
    list_unweighted_edges.append(tuple(ed))

print('Initial_edges:', len(list_unweighted_edges))


for i in range(len(xy)):
	G.add_node(i, pos=xy[i])


position_array = []
for node in sorted(G):
    position_array.append(G.nodes[node]["pos"])


for u, v in list_unweighted_edges:
    if u != v:
        distance = math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow(
            (position_array[u][1] - position_array[v][1]), 2))

        if distance <= txr:

            G.add_edge(u, v, weight=math.ceil(distance))

list_edges = list(G.edges())

print('No of edges:', len(list_edges))


node_pos = nx.get_node_attributes(G, 'pos')
edge_weight = nx.get_edge_attributes(G, 'weight')

node_col = ['green']
edge_col = ['black']

'''
# Draw the nodes
nx.draw_networkx(G, node_pos, node_color=node_col, node_size=200)
# Draw the node labels
nx.draw_networkx_labels(G, node_pos, node_color=node_col)
# Draw the edges
nx.draw_networkx_edges(G, node_pos, edge_color=edge_col)
# Draw the edge labels
#nx.draw_networkx_edge_labels(G, node_pos, edge_color=edge_col, edge_labels=edge_weight)
# Show the axis
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')
#plt.title('Cummulative Energy Consumption per Round')
plt.legend()
plt.axis('on')
# Show the plot
plt.show()
'''
# Draw the nodes

#G.remove_edges_from(G.edges())
fig, ax = plt.subplots()
nx.draw(G, pos=node_pos,  ax=ax) #notice we call draw, and not draw_networkx_nodes
limits=plt.axis('on') # turns on axis
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

plt.show()

