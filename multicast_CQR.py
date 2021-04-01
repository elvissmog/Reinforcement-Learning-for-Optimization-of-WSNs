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
episodes = 1

sink_node = 5

for i in range(len(G)):
	for j in range(len(G)):
		if i !=j:
			d[i][j] = math.sqrt(math.pow((position_array[i][0] - position_array[j][0]), 2) + math.pow((position_array[i][1] - position_array[j][1]), 2))
			Etx[i][j] = electronic_energy * packet_size + amplifier_energy * packet_size * math.pow((d[i][j]), pathloss_exponent)
			Erx[i][j] =  electronic_energy * packet_size


Y = Yamada(G)
all_MSTs = Y.spanning_trees()

#the set of neighbors of all nodes in each MST
node_neigh=[]
for T in all_MSTs:
	node_neighT = {}
	for n in T.nodes:
		node_neighT[n] = list(T.neighbors(n))
	node_neigh.append(node_neighT)
#print(node_neigh)

# Ranking nodes in terms of hop count to sink for each MST
MSTs_hop_count = []
MST_paths = []
for T in all_MSTs:
	hop_counts = {}
	MST_path = {}
	for n in T.nodes:
		for path in nx.all_simple_paths(T, source=n, target=sink_node):
			hop_counts[n] = len(path) - 1
			MST_path[n]=path
	hop_counts[sink_node] = 0                  # hop count of sink
	MSTs_hop_count.append(hop_counts)
	MST_paths.append(MST_path)
	
print('All paths', MST_paths)



Q_matrix = np.zeros((len(all_MSTs), len(all_MSTs)))
initial_state = random.choice(range(0, len(all_MSTs), 1))


Q_value = []
Min_value = []
Actions = []
Episode = []
delay = []
E_consumed = []
EE_consumed = []

for i in range(episodes):

    initial_delay = 0
    tx_energy = 0
    rx_energy = 0
    Episode.append(i)
    available_actions = [*range(0, len(all_MSTs), 1)]


    current_state = initial_state

    if np.random.random() >= 1 - epsilon:
            # Get action from Q table
        action = random.choice(available_actions)
    else:
            # Get random action
        action = np.argmax(Q_matrix[current_state, :])

    Actions.append(action)

    initial_state = action
    #print('action is:', action)

   
    chosen_MST = MST_paths[action]
    print(chosen_MST)

    for node in chosen_MST:
    	counter = 0
    	while counter < len(chosen_MST[node]):
    		init_node = chosen_MST[node][counter]
    		next_node = chosen_MST[node][counter + 1]
    		E_vals[init_node] = E_vals[init_node] - Etx[init_node][next_node]  # update the start node energy
    		E_vals[next_node] = E_vals[next_node] - Erx[init_node][next_node]  # update the next hop energy
    		counter+=1

    print('Energy',E_vals)
    reward = min(E_vals)



   