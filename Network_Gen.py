import matplotlib.pyplot as plt
import networkx as nx
import math
import json
import random


G = nx.Graph()

'''
n = 101
w = 5050

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

print('p:', position_array)
print('e:', list_unweighted_edges)

for u, v in list_unweighted_edges:
    distance = math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow(
        (position_array[u][1] - position_array[v][1]), 2))

    G.add_edge(u, v, weight=math.ceil(distance))

list_edges = list(G.edges())


# open output file for writing
with open('edges1.txt', 'w') as filehandle:
    json.dump(list_edges, filehandle)

with open('pos1.txt', 'w') as filehandle:
    json.dump(position_array, filehandle)


traffic = {}
for node in G.nodes:
    if node != sink_node:
        traffic[node] = random.randint(1, 5)

with open('traffic.txt', 'w') as filehandle:
    json.dump(traffic, filehandle)

'''


