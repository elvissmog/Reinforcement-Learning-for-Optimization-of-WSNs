import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import time



G = nx.Graph()

transmission_range = 50

list_unweighted_edges = [(54, 100), (96, 14), (39, 93), (91, 53), (21, 78), (95, 72), (25, 65), (22, 32), (33, 2), (23, 68), (55, 69), (81, 97), (46, 54), (55, 67), (70, 20), (26, 38), (28, 48), (17, 35), (74, 96), (61, 53), (83, 48), (4, 5), (77, 51), (35, 15), (30, 57), (83, 90), (77, 44), (67, 39), (68, 52), (57, 52), (23, 54), (47, 61), (61, 33), (36, 49), (32, 55), (14, 41), (87, 63), (2, 79), (81, 89), (43, 57), (78, 29), (22, 87), (12, 62), (29, 73), (23, 66), (71, 99), (38, 33), (85, 13), (86, 88), (19, 46), (31, 45), (71, 45), (41, 12), (62, 51), (40, 8), (0, 61), (22, 18), (0, 34), (96, 11), (77, 10), (70, 78), (63, 75), (3, 95), (84, 16), (32, 86), (16, 85), (24, 90), (11, 83), (73, 33), (48, 2), (47, 98), (31, 34), (64, 52), (87, 58), (6, 96), (66, 40), (50, 14), (12, 10), (1, 84), (27, 31), (3, 70), (63, 19), (8, 61), (90, 9), (26, 49), (48, 6), (70, 17), (38, 50), (22, 41), (86, 24), (61, 59), (82, 48), (92, 40), (79, 68), (58, 89), (83, 32), (62, 72), (29, 12), (79, 71), (21, 76), (30, 29), (92, 96), (98, 51), (8, 27), (100, 20), (7, 43), (0, 9), (92, 90), (65, 64), (65, 50), (75, 72), (42, 40), (73, 63), (80, 33), (56, 78), (8, 27), (98, 12), (69, 65), (0, 65), (93, 46), (78, 24), (59, 31), (61, 89), (71, 60), (90, 76), (62, 29), (49, 77), (99, 95), (70, 39), (93, 100), (86, 68), (70, 44), (31, 59), (51, 77), (27, 34), (46, 69), (60, 6), (99, 62), (65, 60), (20, 61), (84, 92), (4, 43), (92, 39), (3, 61), (24, 23), (86, 72), (71, 8), (57, 65), (17, 94), (4, 40), (56, 88), (44, 43), (47, 92), (52, 8), (49, 79), (92, 60), (60, 84), (25, 89), (27, 63), (72, 69), (85, 52), (73, 25), (85, 20), (0, 2), (27, 4), (61, 59), (32, 2), (3, 21), (17, 2), (64, 48), (54, 80), (4, 21), (47, 80), (88, 22), (78, 17), (29, 66), (32, 81), (27, 89), (97, 76), (20, 28), (80, 48), (45, 24), (49, 76), (55, 79), (27, 62), (53, 28), (33, 35), (99, 43), (37, 100), (33, 54), (82, 75), (58, 8), (15, 90), (71, 63), (24, 55), (54, 97), (97, 2), (91, 84), (84, 2), (59, 72), (38, 90), (72, 69), (88, 49), (6, 31), (44, 54), (31, 61), (88, 94), (66, 28), (18, 76), (98, 51), (94, 62), (32, 6), (78, 65), (79, 31), (60, 13), (10, 28), (25, 74), (66, 39), (44, 25), (33, 25), (17, 24), (46, 39), (43, 20), (13, 57), (47, 87), (13, 28), (59, 18), (36, 91), (28, 97), (26, 27), (30, 21), (58, 52), (71, 25), (85, 61), (59, 8), (85, 9), (42, 84), (94, 5), (71, 80), (6, 84), (96, 72), (52, 81), (28, 5), (8, 18), (59, 52), (73, 77), (76, 34), (41, 89), (39, 97)]
xy = [(727, 333), (410, 921), (369, 283), (943, 142), (423, 646), (153, 477), (649, 828), (911, 989), (972, 422), (35, 419), (648, 836), (17, 688), (281, 402), (344, 909), (815, 675), (371, 908), (748, 991), (838, 45), (462, 505), (508, 474), (565, 617), (2, 979), (392, 991), (398, 265), (789, 35), (449, 952), (88, 281), (563, 839), (128, 725), (639, 35), (545, 329), (259, 294), (379, 907), (830, 466), (620, 290), (789, 579), (778, 453), (667, 663), (665, 199), (844, 732), (105, 884), (396, 411), (351, 452), (488, 584), (677, 94), (743, 659), (752, 203), (108, 318), (941, 691), (981, 702), (100, 701), (783, 822), (250, 788), (96, 902), (540, 471), (449, 473), (671, 295), (870, 246), (588, 102), (703, 121), (402, 637), (185, 645), (808, 10), (668, 617), (467, 852), (280, 39), (563, 377), (675, 334), (429, 177), (494, 637), (430, 831), (57, 726), (509, 729), (376, 311), (429, 833), (395, 417), (628, 792), (512, 259), (845, 729), (456, 110), (277, 501), (211, 996), (297, 689), (160, 87), (590, 605), (498, 557), (971, 211), (562, 326), (315, 963), (316, 471), (390, 316), (365, 755), (573, 631), (881, 532), (969, 218), (220, 388), (517, 500), (869, 670), (490, 575), (331, 992), (500, 500)]

for i in range(len(xy)):
    G.add_node(i, pos=xy[i])

position_array = []
for node in sorted(G):
    position_array.append(G.nodes[node]['pos'])
distances = squareform(pdist(np.array(position_array)))
#print('dis_max:', distances)
for u, v in list_unweighted_edges:
    #G.add_edge(u, v, weight = 1)
    #G.add_edge(u, v, weight=np.round(distances[u][v], decimals=1))
    distance = math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow(
        (position_array[u][1] - position_array[v][1]), 2))
    nor_distance = math.ceil(distance / transmission_range)
    G.add_edge(u, v, weight=nor_distance)

# initialization of network parameters

initial_energy = 500  # Joules
data_packet_size = 320  # bits
electronic_energy = 50e-9  # Joules/bit 5
e_fs = 10e-12  # Joules/bit/(meter)**2
e_mp = 0.0013e-12 #Joules/bit/(meter)**4
node_energy_limit = 0.001

d = [[0 for i in range(len(G))] for j in range(len(G))]
Etx = [[0 for i in range(len(G))] for j in range(len(G))]
Erx = [[0 for i in range(len(G))] for j in range(len(G))]
initial_E_vals = [initial_energy for i in range(len(G))]
ref_E_vals = [initial_energy for i in range(len(G))]

episodes = 5000000

sink_energy = 5000000
sink_node = 100

traffic = {0: 5, 1: 3, 2: 5, 3: 5, 4: 5, 5: 2, 6: 5, 7: 2, 8: 3, 9: 2, 10: 3, 11: 5, 12: 5, 13: 3, 14: 4, 15: 1, 16: 3, 17: 2, 18: 2, 19: 3, 20: 5, 21: 5, 22: 2, 23: 2, 24: 1, 25: 4, 26: 2, 27: 2, 28: 2, 29: 5, 30: 4, 31: 4, 32: 1, 33: 1, 34: 5, 35: 3, 36: 1, 37: 4, 38: 4, 39: 3, 40: 2, 41: 4, 42: 2, 43: 1, 44: 1, 45: 1, 46: 2, 47: 2, 48: 1, 49: 3, 50: 5, 51: 2, 52: 1, 53: 3, 54: 1, 55: 5, 56: 3, 57: 4, 58: 3, 59: 5, 60: 5, 61: 3, 62: 3, 63: 1, 64: 1, 65: 4, 66: 4, 67: 5, 68: 3, 69: 2, 70: 4, 71: 5, 72: 1, 73: 5, 74: 1, 75: 5, 76: 4, 77: 5, 78: 4, 79: 4, 80: 2, 81: 5, 82: 1, 83: 5, 84: 4, 85: 1, 86: 1, 87: 4, 88: 1, 89: 1, 90: 2, 91: 5, 92: 4, 93: 3, 94: 4, 95: 3, 96: 3, 97: 5, 98: 4, 99: 3}


initial_E_vals[sink_node] = sink_energy
ref_E_vals[sink_node] = sink_energy

d_o = math.sqrt(e_fs/e_mp)/transmission_range

for i in range(len(G)):
    for j in range(len(G)):
        if i != j:
            d[i][j] = (math.sqrt(math.pow((position_array[i][0] - position_array[j][0]), 2) + math.pow(
                (position_array[i][1] - position_array[j][1]), 2)))/transmission_range

            if d[i][j] <= d_o:
                Etx[i][j] = electronic_energy * data_packet_size + e_fs * data_packet_size * math.pow((d[i][j]), 2)
            else:
                Etx[i][j] = electronic_energy * data_packet_size + e_mp * data_packet_size * math.pow((d[i][j]), 4)
            Erx[i][j] = electronic_energy * data_packet_size

print('distance:', d)
print('dist_thres:', d_o)
print('Energy:', Etx)

shortest_paths = {}

for n in G.nodes:
    path = nx.dijkstra_path(G, source=n, target=sink_node, weight='weight')
    #path = nx.bellman_ford_path(G, source=n, target=sink_node, weight='weight')
    shortest_paths[n] = path
print('shortest_paths:', shortest_paths)

Min_value = []
Episode = []
delay = []
E_consumed = []
EE_consumed = []
Min_nodes_RE = []

start_time = time.time()

for i in range(episodes):

    initial_delay = 0
    tx_energy = 0
    rx_energy = 0
    Episode.append(i)

    ETX = []
    ERX = []
    for node in shortest_paths:
        counter = 0
        while counter < len(shortest_paths[node]) - 1:
            init_node = shortest_paths[node][counter]
            next_node = shortest_paths[node][counter + 1]
            initial_E_vals[init_node] = initial_E_vals[init_node] - traffic[node]*Etx[init_node][next_node]  # update the start node energy
            initial_E_vals[next_node] = initial_E_vals[next_node] - traffic[node]*Erx[init_node][next_node]  # update the next hop energy
            tx_energy += traffic[node]*Etx[init_node][next_node]
            rx_energy += traffic[node]*Erx[init_node][next_node]
            counter += 1
        ETX.append(tx_energy)
        ERX.append(rx_energy)
    # print('ref Evals:', ref_E_vals)
    # print('initial Energy:', initial_E_vals)
    # current_E_vals = initial_E_vals
    Energy_Consumption = [ref_E_vals[i] - initial_E_vals[i] for i in G.nodes if i != sink_node]
    # print('Energy Consumption:', Energy_Consumption)
    reward = max(Energy_Consumption)

    # reward = (tx_energy + rx_energy)
    # reward = min(E_vals)
    Min_value.append(reward)

    cost = True
    for index, item in enumerate(initial_E_vals):
        if item <= node_energy_limit:
            print('E_vals:', initial_E_vals)
            print('Index:', index)
            cost = False
            print("Energy cannot be negative!")
            print("The final round is", i)

    if not cost:
        break
    # ref_E_vals = initial_E_vals
    Min_nodes_RE.append(min(initial_E_vals))
    for i in range(len(ref_E_vals)):
        ref_E_vals[i] = initial_E_vals[i]



print("--- %s seconds ---" % (time.time() - start_time))

plt.plot(Episode, Min_value)
plt.xlabel('Round')
plt.ylabel('Minimum Energy')
plt.legend()
plt.show()

'''
print('Reward:', Min_value)
print('Round:', Episode)
print('Delay:', delay)
print('Total Energy:', EE_consumed)
print('Energy:', E_consumed)

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

'''




