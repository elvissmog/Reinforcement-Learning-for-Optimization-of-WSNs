import networkx as nx
import matplotlib.pyplot as plt
from pqdict import PQDict
import math

g = nx.Graph()

xy = [(1, 3), (2.5, 5), (2.5, 1), (4.5, 5), (4.5, 1), (6, 3)]
list_unweighted_edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]
for i in range(len(xy)):
	g.add_node(i, pos=xy[i])

position_array = []
for node in sorted(g):
    position_array.append(g.nodes[node]['pos'])
for u, v in list_unweighted_edges:
    distance = math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow(
        (position_array[u][1] - position_array[v][1]), 2))

    g.add_edge(u, v, weight = distance)


node_pos = nx.get_node_attributes(g, 'pos')
edge_weight = nx.get_edge_attributes(g, 'weight')

node_col = ['green']
edge_col = ['black']

# Draw the nodes
nx.draw_networkx(g, node_pos, node_color=node_col, node_size=200)
# Draw the node labels
nx.draw_networkx_labels(g, node_pos, node_color=node_col)
# Draw the edges
nx.draw_networkx_edges(g, node_pos, edge_color=edge_col)
nx.draw_networkx_edge_labels(g, node_pos, edge_color=edge_col, edge_labels=edge_weight)

plt.axis('on')
#plt.show()



def prim(G, start):
    """Function recives a graph and a starting node, and returns a MST"""
    stopN = G.number_of_nodes() - 1
    current = start
    closedSet = set()
    pq = PQDict()
    mst = []

    while len(mst) < stopN:
        for node in G.neighbors(current):
            if node not in closedSet and current not in closedSet:
                if (current, node) not in pq and (node, current) not in pq:
                    w = G.edges[(current, node)]['weight']
                    pq.additem((current, node), w)

        closedSet.add(current)

        tup, wght = pq.popitem()
        while(tup[1] in closedSet):
            tup, wght = pq.popitem()
        mst.append(tup)
        current = tup[1]

    h = nx.Graph()

    for j in range(len(xy)):
        h.add_node(j, pos=xy[j])

    pos_array = []
    for node in sorted(h):
        pos_array.append(h.nodes[node]['pos'])
    # print(T_edges)
    for (x, y) in mst:
        distance = math.sqrt(math.pow((position_array[x][0] - position_array[y][0]), 2) + math.pow(
            (position_array[x][1] - position_array[y][1]), 2))

        h.add_edge(x, y, weight=distance)

    return h




MST = []

for i in range(len(g.nodes)):
    y = prim(g,i)
    MST.append(y)


print(len(g.nodes))
print(len(MST))

auxiliaryMST = []
for mst in MST:
    if mst not in auxiliaryMST:
        auxiliaryMST.append(mst)

print(len(auxiliaryMST))

'''
for w in MST:
    n_pos = nx.get_node_attributes(w, 'pos')
    e_weight = nx.get_edge_attributes(w, 'weight')

    n_col = ['green']
    e_col = ['black']

    # Draw the nodes
    nx.draw_networkx(w, n_pos, node_color=n_col, node_size=200)
    # Draw the node labels
    nx.draw_networkx_labels(w, n_pos, node_color=n_col)
    # Draw the edges
    nx.draw_networkx_edges(w, n_pos, edge_color=e_col)
    # Draw the edge labels
    nx.draw_networkx_edge_labels(w, n_pos, edge_color=e_col, edge_labels=e_weight)
    # Show the axis
    plt.axis('on')
    # Show the plot
    plt.axis('on')
    plt.title('Minimum Spanning Tree of Graph G')
    plt.show()
'''



