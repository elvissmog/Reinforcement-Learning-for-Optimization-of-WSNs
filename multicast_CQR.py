import numpy as np
import networkx as nx
import random
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import time

from AllMst import Yamada

G = nx.Graph()
xy = [(1, 3), (2.5, 5), (2.5, 1), (4.5, 5), (4.5, 1), (6, 3)]
for i in range(len(xy)):
	G.add_node(i, pos=xy[i])

# Adding unweighted edges to the Graph
list_unweighted_edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]

position_array = []
for node in sorted(G):
    position_array.append(G.nodes[node]['pos'])
distances = squareform(pdist(np.array(position_array)))
for u, v in list_unweighted_edges:
    #G.add_edge(u, v, weight = 1)
    G.add_edge(u, v, weight=np.round(distances[u][v], decimals=1))

# initialization of network parameters
discount_factor = 0.5
learning_rate = 0.5
initial_energy = 0.5                  # Joules
packet_size = 512                     # bits
electronic_energy = 50e-9            # Joules/bit 5
amplifier_energy = 100e-12           # Joules/bit/square meter
transmission_range = 30               # meters
pathloss_exponent = 2                 # constant

d =[[0 for i in range(len(G))] for j in range(len(G))]
Etx=[[0 for i in range(len(G))] for j in range(len(G))]
Erx=[[0 for i in range(len(G))] for j in range(len(G))]
path_Q_values  =[[0 for i in range(len(G))] for j in range(len(G))]
E_vals = [initial_energy for i in range(len(G))]
epsilon = 0.1
episodes = 200000

sink_node = 5

for i in range(len(G)):
	for j in range(len(G)):
		if i !=j:
			d[i][j] = math.sqrt(math.pow((position_array[i][0] - position_array[j][0]), 2) + math.pow((position_array[i][1] - position_array[j][1]), 2))
			Etx[i][j] = electronic_energy * packet_size + amplifier_energy * packet_size * math.pow((d[i][j]), pathloss_exponent)
			Erx[i][j] =  electronic_energy * packet_size


Y = Yamada(G)
all_MSTs = Y.spanning_trees()

Q_matrix = np.zeros((len(all_MSTs), len(all_MSTs)))