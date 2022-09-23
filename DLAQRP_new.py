import numpy as np
import networkx as nx
import random
import math
from AllMst import Yamada
#import matplotlib.pyplot as plt
from collections import Counter
import json
import time

# initialization of network parameters
discount_factor = 0
learning_rate = 0.9
initial_energy = 100        # Joules
data_packet_size = 1024      # bits
electronic_energy = 50e-9   # Joules/bit 5
e_fs = 10e-12               # Joules/bit/(meter)**2
e_mp = 0.0013e-12           #Joules/bit/(meter)**4
node_energy_limit = 0
num_pac = 1
txr = 110
epsilon = 0.1
episodes = 5000000
sink_energy = 5000000
sink_node = 100


start_time = time.time()

G = nx.Graph()

with open('pos.txt', 'r') as filehandle:
    pos_list = json.load(filehandle)

xy = []
for ps in pos_list:
    xy.append(tuple(ps))

with open('edges.txt', 'r') as filehandle:
    edges_list = json.load(filehandle)

list_unweighted_edges = []
for ed in edges_list:
    list_unweighted_edges.append(tuple(ed))

for i in range(len(xy)):
	G.add_node(i, pos=xy[i])

p = []
for node in sorted(G):
    p.append(G.nodes[node]['pos'])

Trx_dis = []
for u, v in list_unweighted_edges:
    distance = math.sqrt(math.pow((p[u][0] - p[v][0]), 2) + math.pow((p[u][1] - p[v][1]), 2))

    if distance <= txr:
        Trx_dis.append(distance)
        G.add_edge(u, v, weight = math.ceil(distance))
        #G.add_edge(u, v, weight=1)

com_range = max(Trx_dis)

print('cm_range:', com_range)

initial_E_vals = [initial_energy for i in G.nodes]
ref_E_vals = [initial_energy for i in G.nodes]

initial_E_vals[sink_node] = sink_energy
ref_E_vals[sink_node] = sink_energy

total_initial_energy = sum(initial_E_vals)

d_o = math.sqrt(e_fs/e_mp)


# Ranking nodes in terms of hop count to sink for each MST
'''
ST_paths = []
for n in G.nodes:
    ST_path = {}
    for path in nx.all_simple_paths(G, source=n, target=sink_node):
        ST_path[n] = path


    ST_paths.append(ST_path)

print(ST_paths)

'''

hop_counts = {}
for n in G.nodes:
    for path in nx.all_simple_paths(G, source=n, target=sink_node):
        hop_counts[n] = len(path) - 1

hop_counts[sink_node]= 1   #hop count of sink

print(hop_counts)


