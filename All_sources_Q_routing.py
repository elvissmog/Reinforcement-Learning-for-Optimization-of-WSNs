import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.spatial.distance import pdist, squareform

# Instantiating the G object

G = nx.Graph()
transmission_range = 50

# Adding nodes to the graph and their corresponding coordinates

#xy = [(1, 3), (2.5, 5), (2.5, 1), (4.5, 5), (4.5, 1), (6, 3)]
#list_unweighted_edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]

xy = [(727, 333), (410, 921), (369, 283), (943, 142), (423, 646), (153, 477), (649, 828), (911, 989), (972, 422), (35, 419), (648, 836), (17, 688), (281, 402), (344, 909), (815, 675), (371, 908), (748, 991), (838, 45), (462, 505), (508, 474), (565, 617), (2, 979), (392, 991), (398, 265), (789, 35), (449, 952), (88, 281), (563, 839), (128, 725), (639, 35), (545, 329), (259, 294), (379, 907), (830, 466), (620, 290), (789, 579), (778, 453), (667, 663), (665, 199), (844, 732), (105, 884), (396, 411), (351, 452), (488, 584), (677, 94), (743, 659), (752, 203), (108, 318), (941, 691), (981, 702), (100, 701), (783, 822), (250, 788), (96, 902), (540, 471), (449, 473), (671, 295), (870, 246), (588, 102), (703, 121), (402, 637), (185, 645), (808, 10), (668, 617), (467, 852), (280, 39), (563, 377), (675, 334), (429, 177), (494, 637), (430, 831), (57, 726), (509, 729), (376, 311), (429, 833), (395, 417), (628, 792), (512, 259), (845, 729), (456, 110), (277, 501), (211, 996), (297, 689), (160, 87), (590, 605), (498, 557), (971, 211), (562, 326), (315, 963), (316, 471), (390, 316), (365, 755), (573, 631), (881, 532), (969, 218), (220, 388), (517, 500), (869, 670), (490, 575), (331, 992), (500, 500)]
list_unweighted_edges = [(54, 100), (96, 14), (39, 93), (91, 53), (21, 78), (95, 72), (25, 65), (22, 32), (33, 2), (23, 68), (55, 69), (81, 97), (46, 54), (55, 67), (70, 20), (26, 38), (28, 48), (17, 35), (74, 96), (61, 53), (83, 48), (4, 5), (77, 51), (35, 15), (30, 57), (83, 90), (77, 44), (67, 39), (68, 52), (57, 52), (23, 54), (47, 61), (61, 33), (36, 49), (32, 55), (14, 41), (87, 63), (2, 79), (81, 89), (43, 57), (78, 29), (22, 87), (12, 62), (29, 73), (23, 66), (71, 99), (38, 33), (85, 13), (86, 88), (19, 46), (31, 45), (71, 45), (41, 12), (62, 51), (40, 8), (0, 61), (22, 18), (0, 34), (96, 11), (77, 10), (70, 78), (63, 75), (3, 95), (84, 16), (32, 86), (16, 85), (24, 90), (11, 83), (73, 33), (48, 2), (47, 98), (31, 34), (64, 52), (87, 58), (6, 96), (66, 40), (50, 14), (12, 10), (1, 84), (27, 31), (3, 70), (63, 19), (8, 61), (90, 9), (26, 49), (48, 6), (70, 17), (38, 50), (22, 41), (86, 24), (61, 59), (82, 48), (92, 40), (79, 68), (58, 89), (83, 32), (62, 72), (29, 12), (79, 71), (21, 76), (30, 29), (92, 96), (98, 51), (8, 27), (100, 20), (7, 43), (0, 9), (92, 90), (65, 64), (65, 50), (75, 72), (42, 40), (73, 63), (80, 33), (56, 78), (8, 27), (98, 12), (69, 65), (0, 65), (93, 46), (78, 24), (59, 31), (61, 89), (71, 60), (90, 76), (62, 29), (49, 77), (99, 95), (70, 39), (93, 100), (86, 68), (70, 44), (31, 59), (51, 77), (27, 34), (46, 69), (60, 6), (99, 62), (65, 60), (20, 61), (84, 92), (4, 43), (92, 39), (3, 61), (24, 23), (86, 72), (71, 8), (57, 65), (17, 94), (4, 40), (56, 88), (44, 43), (47, 92), (52, 8), (49, 79), (92, 60), (60, 84), (25, 89), (27, 63), (72, 69), (85, 52), (73, 25), (85, 20), (0, 2), (27, 4), (61, 59), (32, 2), (3, 21), (17, 2), (64, 48), (54, 80), (4, 21), (47, 80), (88, 22), (78, 17), (29, 66), (32, 81), (27, 89), (97, 76), (20, 28), (80, 48), (45, 24), (49, 76), (55, 79), (27, 62), (53, 28), (33, 35), (99, 43), (37, 100), (33, 54), (82, 75), (58, 8), (15, 90), (71, 63), (24, 55), (54, 97), (97, 2), (91, 84), (84, 2), (59, 72), (38, 90), (72, 69), (88, 49), (6, 31), (44, 54), (31, 61), (88, 94), (66, 28), (18, 76), (98, 51), (94, 62), (32, 6), (78, 65), (79, 31), (60, 13), (10, 28), (25, 74), (66, 39), (44, 25), (33, 25), (17, 24), (46, 39), (43, 20), (13, 57), (47, 87), (13, 28), (59, 18), (36, 91), (28, 97), (26, 27), (30, 21), (58, 52), (71, 25), (85, 61), (59, 8), (85, 9), (42, 84), (94, 5), (71, 80), (6, 84), (96, 72), (52, 81), (28, 5), (8, 18), (59, 52), (73, 77), (76, 34), (41, 89), (39, 97)]


for i in range(len(xy)):
	G.add_node(i, pos=xy[i])

# Adding unweighted edges to the Graph


# Extracting the (x,y) coordinate of each node to enable the calculation of the Euclidean distances of the G edges
position_array = []
for node in sorted(G):
    position_array.append(G.nodes[node]['pos'])
distances = squareform(pdist(np.array(position_array)))
#print('dis_max:', distances)
for u, v in list_unweighted_edges:

    distance = math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow(
        (position_array[u][1] - position_array[v][1]), 2))
    nor_distance = math.ceil(distance/transmission_range)
    G.add_edge(u, v, weight=nor_distance)


# the set of neighbors of all nodes in the graph
node_neigh = {}
for n in G.nodes:
    node_neigh[n] = list(G.neighbors(n))

# initialization of network parameters
learning_rate = 0.7
initial_energy = 1000  # Joules
data_packet_size = 512  # bits
electronic_energy = 50e-9  # Joules/bit 50e-9
e_fs = 10e-12  # Joules/bit/(meter)**2
e_mp = 0.0013e-12 #Joules/bit/(meter)**4
node_energy_limit = 10

R = [[[0 for i in range(len(G))] for j in range(len(G))] for n in range(len(G))]
d = [[0 for i in range(len(G))] for j in range(len(G))]
Etx = [[0 for i in range(len(G))] for j in range(len(G))]
Erx = [[0 for i in range(len(G))] for j in range(len(G))]
path_Q_values = [[[0 for i in range(len(G))] for j in range(len(G))] for n in range(len(G))]
Q_vals = [[0 for i in range(len(G))] for j in range(len(G))]
E_vals = [initial_energy for i in range(len(G))]


epsilon = 0.0

sink_node = 100
E_vals[sink_node] = 500000

d_o = math.sqrt(e_fs/e_mp)

for i in range(len(G)):
    for j in range(len(G)):
        if i != j:
            d[i][j] = math.ceil((math.sqrt(math.pow((position_array[i][0] - position_array[j][0]), 2) + math.pow(
                (position_array[i][1] - position_array[j][1]), 2)))/transmission_range)

            if d[i][j] <= d_o:
                Etx[i][j] = electronic_energy * data_packet_size + e_fs * data_packet_size * math.pow((d[i][j]), 2)
            else:
                Etx[i][j] = electronic_energy * data_packet_size + e_mp * data_packet_size * math.pow((d[i][j]), 4)
            Erx[i][j] = electronic_energy * data_packet_size

# initialize starting point

num_of_episodes = 2000000
round = []
Av_mean_Q = []
Av_E_consumed = []
Av_delay = []

for i in range(num_of_episodes):

    # print("The start node in episode number {} is {} ".format(i+1, start))
    cost = True
    for item in E_vals:
        if item <= node_energy_limit:
            cost = False
            print("Energy cannot be negative!")
            #print('E_vals:', E_vals)
            print("The final round is:", i)

    if not cost:
        break

    mean_Q = []
    E_consumed = []
    EE_consumed = []
    delay = []
    path_f = []
    for node in range(len(G.nodes)-1):
        if node != sink_node:
            start = node
            queue = [start]  # first visited node
            path = str(start)  # first node
            temp_qval = {}
            initial_delay = 0
            tx_energy = 0
            rx_energy = 0

            while True:

                for neigh in node_neigh[start]:
                    temp_qval[neigh] = (1 - learning_rate) * path_Q_values[node][start][neigh] + learning_rate * (d[start][neigh] + Q_vals[node][neigh])

                copy_q_values = {key: value for key, value in temp_qval.items() if key not in queue}

                # next_hop = min(temp_qval.keys(), key=(lambda k: temp_qval[k]))  #determine the next hop based on the minimum qvalue but ignore the visited node qvalue

                if np.random.random() >= 1 - epsilon:
                    # Get action from Q table
                    next_hop = random.choice(list(copy_q_values.keys()))
                else:
                    # Get random action
                    next_hop = min(copy_q_values.keys(), key=(lambda k: copy_q_values[k]))

                initial_delay += d[start][next_hop]

                queue.append(next_hop)

                path_Q_values[node][start][next_hop] = temp_qval[next_hop]  # update the path qvalue of the next hop
                Q_vals[node][start] = temp_qval[next_hop]  # update the qvalue of the start node

                # sum_Q[i] = sum(Q_vals)/len(Q_vals)
                mean_Qvals = sum(Q_vals[node]) / (len(Q_vals[node]) * max(Q_vals[node]))
                E_vals[start] = E_vals[start] - Etx[start][next_hop]  # update the start node energy
                E_vals[next_hop] = E_vals[next_hop] - Erx[start][next_hop]  # update the next hop energy
                tx_energy += Etx[start][next_hop]
                rx_energy += Erx[start][next_hop]

                path = path + "->" + str(next_hop)  # update the path after each visit

                # print("The visited nodes are", queue)

                start = next_hop

                if next_hop == sink_node:
                    break

                delay.append(initial_delay)
                E_consumed.append(tx_energy + rx_energy)
                #EE_consumed.append(sum(E_consumed))
                mean_Q.append(mean_Qvals)

        path_f.append(path)
    #print('path:', path_f)
    Av_mean_Q.append(sum(mean_Q))
    Av_delay.append(sum(delay))
    Av_E_consumed.append(sum(E_consumed))
    round.append(i)


plt.plot(round, Av_mean_Q)
plt.xlabel('Round')
plt.ylabel('Average Q-Value')
plt.title('Q-Value Convergence ')
plt.show()

plt.plot(round, Av_delay)
plt.xlabel('Round')
plt.ylabel('Average delay (s)')
plt.title('Delay for each round')
plt.show()

plt.plot(round, Av_E_consumed)
plt.xlabel('Round')
plt.ylabel('Average Energy Consumption (Joules)')
plt.title('Energy Consumption')
plt.show()



# print("The path is", path)
# print("The Energy of the nodes after episode {} is {} ".format(i,E_vals))
# print("The Q_values of the nodes after episode {} is  {}".format(i,Q_vals))
# print("The sum of the Q_values after episode {} is {}".format(i,sum_Q[i]))


# x_round = [i for i in range(1, num_of_episodes+1)]
# plt.plot(x_round, EE_consumed)
# plt.plot(x_round, sum_Q)
# plt.plot(x_round, EE_consumed)
# plt.show()

# print("The path Q_values at episode {} is {}".format(i+1,path_Q_values))

'''
print('Round:', round)
print('Delay:', delay)
print('Total Energy:', EE_consumed)
print('Energy:', E_consumed)
print('Average_QVals:', mean_Q)
print('Length Round:', len(round))
print('Length Delay:', len(delay))
print('Length Q_Val:', len(mean_Q))
print('Length Energy:', len(EE_consumed))

plt.plot(round, EE_consumed)
plt.xlabel('Round')
plt.ylabel('Total Energy Consumption (Joules)')
plt.title('Total Energy Consumption for each round')
plt.show()

'''

