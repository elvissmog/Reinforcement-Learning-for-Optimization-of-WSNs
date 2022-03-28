import random
import networkx as nx
import math
import json


G = nx.Graph()
n = 1000
w = 5000

for i in range(n):
    G.add_node(list(range(n))[i], pos=(random.randint(0, 1000), random.randint(0, 1000)))

list_unweighted_edges = []
for i in range(w):
    u = random.choice(range(n))
    v = random.choice((range(n)))
    if u != v:
        list_unweighted_edges.append((u, v))

print('No of edges:', len(list_unweighted_edges))
position_array = []
for node in sorted(G):
    position_array.append(G.nodes[node]["pos"])

print('No of nodes:', len(position_array))

for u, v in list_unweighted_edges:
    distance = math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow(
        (position_array[u][1] - position_array[v][1]), 2))

    G.add_edge(u, v, weight=math.ceil(distance))

list_edges = list(G.edges())

# open output file for writing
with open('edges.txt', 'w') as filehandle:
    json.dump(list_edges, filehandle)

with open('pos.txt', 'w') as filehandle:
    json.dump(position_array, filehandle)

