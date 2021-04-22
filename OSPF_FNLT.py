import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import time



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
    # G.add_edge(u, v, weight = 1)
    G.add_edge(u, v, weight=np.round(distances[u][v], decimals=1))

# initialization of network parameters
initial_energy = 0.5  # Joules
packet_size = 512  # bits
electronic_energy = 50e-9  # Joules/bit 5
amplifier_energy = 100e-12  # Joules/bit/square meter
transmission_range = 30  # meters
pathloss_exponent = 2  # constant

d = [[0 for i in range(len(G))] for j in range(len(G))]
Etx = [[0 for i in range(len(G))] for j in range(len(G))]
Erx = [[0 for i in range(len(G))] for j in range(len(G))]
path_Q_values = [[0 for i in range(len(G))] for j in range(len(G))]
E_vals = [initial_energy for i in range(len(G))]

episodes = 200000

sink_node = 5

for i in range(len(G)):
    for j in range(len(G)):
        if i != j:
            d[i][j] = math.sqrt(math.pow((position_array[i][0] - position_array[j][0]), 2) + math.pow(
                (position_array[i][1] - position_array[j][1]), 2))
            Etx[i][j] = electronic_energy * packet_size + amplifier_energy * packet_size * math.pow((d[i][j]),
                                                                                                    pathloss_exponent)
            Erx[i][j] = electronic_energy * packet_size


shortest_paths = {}


for n in G.nodes:
    path = nx.dijkstra_path(G, source=n, target=sink_node, weight='weight')
    #path = nx.bellman_ford_path(G, source=n, target=sink_node, weight='weight')
    shortest_paths[n] = path


Min_value = []
Episode = []
delay = []
E_consumed = []
EE_consumed = []

start_time = time.time()

for i in range(episodes):

    initial_delay = 0
    tx_energy = 0
    rx_energy = 0
    Episode.append(i)


    for node in shortest_paths:
        counter = 0
        while counter < len(shortest_paths[node]) - 1:
            init_node = shortest_paths[node][counter]
            next_node = shortest_paths[node][counter + 1]
            E_vals[init_node] = E_vals[init_node] - Etx[init_node][next_node]  # update the start node energy
            # E_vals[next_node] = E_vals[next_node] - Erx[init_node][next_node]  # update the next hop energy
            tx_energy += Etx[init_node][next_node]
            rx_energy += Erx[init_node][next_node]
            counter += 1
            # print("counter", counter)

    reward = min(E_vals)


    Min_value.append(reward)
    delay.append(initial_delay)
    E_consumed.append(tx_energy + rx_energy)
    EE_consumed.append(sum(E_consumed))

    cost = True
    for item in E_vals:
        if item <= 0:
            cost = False
            print("Energy cannot be negative!")
            print("The final round is", i)

    if not cost:
        break



print("--- %s seconds ---" % (time.time() - start_time))

'''
print('Reward:', Min_value)
print('Round:', Episode)
print('Delay:', delay)
print('Total Energy:', EE_consumed)
print('Energy:', E_consumed)
'''


plt.plot(Episode, Min_value)
plt.xlabel('Round')
plt.ylabel('Minimum Energy')
plt.legend()
plt.show()


plt.plot(Episode, E_consumed)
plt.xlabel('Round')
plt.ylabel('Energy Consumption (Joules)')
#plt.title('Energy Consumption for each round')
plt.show()

plt.plot(Episode, EE_consumed)
plt.xlabel('Round')
plt.ylabel('Total Energy Consumption (Joules)')
#plt.title('Total Energy Consumption for each round')
plt.show()






