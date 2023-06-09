import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random


sink_node = 100
def build_graph(positions, links):
    G = nx.Graph()
    for i in positions:
        G.add_node(i, pos=positions[i])

    # Extracting the (x,y) coordinate of each node to enable the calculation of the Euclidean distances of the Graph edges
    global position_array
    position_array = {}
    for nd in G:
        position_array[nd] = G.nodes[nd]['pos']

    #Adding unweighted edges to the graph and calculating the distances
    for u, v in links:
        G.add_edge(u, v, weight = math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow((position_array[u][1] - position_array[v][1]), 2)))

    # The dictionary of neighbors of all nodes in the graph
    node_neigh = {}
    for n in G.nodes:
        node_neigh[n] = list(G.neighbors(n))

    q_vals = {}
    for ix in G.nodes:
        q_vals[ix] = 0

    path_q_vals = {}
    for xi in G.nodes:
        path_q_vals[xi] = q_vals

    # Energy consumption
    e_vals = {}

    for idx in G.nodes:
        if idx != sink_node:
            e_vals[idx] = initial_energy
        else:
            e_vals[idx] = 50

    return G, node_neigh, q_vals, e_vals, path_q_vals


xy = {0: (1, 3), 1: (2.5, 5), 2: (2.5, 1), 3: (4.5, 5), 4: (4.5, 1), 5: (6, 3)}
list_unweighted_edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]

'''
list_unweighted_edges = [(54, 100), (96, 14), (39, 93), (91, 53), (21, 78), (95, 72), (25, 65), (22, 32), (33, 2), (23, 68), (55, 69), (81, 97), (46, 54), (55, 67), (70, 20), (26, 38), (28, 48), (17, 35), (74, 96), (61, 53), (83, 48), (4, 5), (77, 51), (35, 15), (30, 57), (83, 90), (77, 44), (67, 39), (68, 52), (57, 52), (23, 54), (47, 61), (61, 33), (36, 49), (32, 55), (14, 41), (87, 63), (2, 79), (81, 89), (43, 57), (78, 29), (22, 87), (12, 62), (29, 73), (23, 66), (71, 99), (38, 33), (85, 13), (86, 88), (19, 46), (31, 45), (71, 45), (41, 12), (62, 51), (40, 8), (0, 61), (22, 18), (0, 34), (96, 11), (77, 10), (70, 78), (63, 75), (3, 95), (84, 16), (32, 86), (16, 85), (24, 90), (11, 83), (73, 33), (48, 2), (47, 98), (31, 34), (64, 52), (87, 58), (6, 96), (66, 40), (50, 14), (12, 10), (1, 84), (27, 31), (3, 70), (63, 19), (8, 61), (90, 9), (26, 49), (48, 6), (70, 17), (38, 50), (22, 41), (86, 24), (61, 59), (82, 48), (92, 40), (79, 68), (58, 89), (83, 32), (62, 72), (29, 12), (79, 71), (21, 76), (30, 29), (92, 96), (98, 51), (8, 27), (100, 20), (7, 43), (0, 9), (92, 90), (65, 64), (65, 50), (75, 72), (42, 40), (73, 63), (80, 33), (56, 78), (8, 27), (98, 12), (69, 65), (0, 65), (93, 46), (78, 24), (59, 31), (61, 89), (71, 60), (90, 76), (62, 29), (49, 77), (99, 95), (70, 39), (93, 100), (86, 68), (70, 44), (31, 59), (51, 77), (27, 34), (46, 69), (60, 6), (99, 62), (65, 60), (20, 61), (84, 92), (4, 43), (92, 39), (3, 61), (24, 23), (86, 72), (71, 8), (57, 65), (17, 94), (4, 40), (56, 88), (44, 43), (47, 92), (52, 8), (49, 79), (92, 60), (60, 84), (25, 89), (27, 63), (72, 69), (85, 52), (73, 25), (85, 20), (0, 2), (27, 4), (61, 59), (32, 2), (3, 21), (17, 2), (64, 48), (54, 80), (4, 21), (47, 80), (88, 22), (78, 17), (29, 66), (32, 81), (27, 89), (97, 76), (20, 28), (80, 48), (45, 24), (49, 76), (55, 79), (27, 62), (53, 28), (33, 35), (99, 43), (37, 100), (33, 54), (82, 75), (58, 8), (15, 90), (71, 63), (24, 55), (54, 97), (97, 2), (91, 84), (84, 2), (59, 72), (38, 90), (72, 69), (88, 49), (6, 31), (44, 54), (31, 61), (88, 94), (66, 28), (18, 76), (98, 51), (94, 62), (32, 6), (78, 65), (79, 31), (60, 13), (10, 28), (25, 74), (66, 39), (44, 25), (33, 25), (17, 24), (46, 39), (43, 20), (13, 57), (47, 87), (13, 28), (59, 18), (36, 91), (28, 97), (26, 27), (30, 21), (58, 52), (71, 25), (85, 61), (59, 8), (85, 9), (42, 84), (94, 5), (71, 80), (6, 84), (96, 72), (52, 81), (28, 5), (8, 18), (59, 52), (73, 77), (76, 34), (41, 89), (39, 97)]
xz = [(727, 333), (410, 921), (369, 283), (943, 142), (423, 646), (153, 477), (649, 828), (911, 989), (972, 422), (35, 419), (648, 836), (17, 688), (281, 402), (344, 909), (815, 675), (371, 908), (748, 991), (838, 45), (462, 505), (508, 474), (565, 617), (2, 979), (392, 991), (398, 265), (789, 35), (449, 952), (88, 281), (563, 839), (128, 725), (639, 35), (545, 329), (259, 294), (379, 907), (830, 466), (620, 290), (789, 579), (778, 453), (667, 663), (665, 199), (844, 732), (105, 884), (396, 411), (351, 452), (488, 584), (677, 94), (743, 659), (752, 203), (108, 318), (941, 691), (981, 702), (100, 701), (783, 822), (250, 788), (96, 902), (540, 471), (449, 473), (671, 295), (870, 246), (588, 102), (703, 121), (402, 637), (185, 645), (808, 10), (668, 617), (467, 852), (280, 39), (563, 377), (675, 334), (429, 177), (494, 637), (430, 831), (57, 726), (509, 729), (376, 311), (429, 833), (395, 417), (628, 792), (512, 259), (845, 729), (456, 110), (277, 501), (211, 996), (297, 689), (160, 87), (590, 605), (498, 557), (971, 211), (562, 326), (315, 963), (316, 471), (390, 316), (365, 755), (573, 631), (881, 532), (969, 218), (220, 388), (517, 500), (869, 670), (490, 575), (331, 992), (500, 500)]
xy = {}
for ixz in range(len(xz)):
    xy[ixz] = xz[ixz]
'''

# initialization of network parameters
learning_rate = 0.7
initial_energy = 20  # Joules
data_packet_size = 512  # bits
electronic_energy = 50e-9  # Joules/bit 50e-9
amplifier_energy = 10e-12  # Joules/bit/square meter 100e-12
transmission_range = 30  # meters
node_energy_limit = 2
epsilon = 0.0

# initialize starting point

num_of_episodes = 84000
mean_Q = []
E_consumed = []
delay = []
round = []

graph, node_neighbors, q_values, e_values, path_q_values = build_graph(xy, list_unweighted_edges)

for rdn in range(num_of_episodes):

    start = 0
    queue = [start]  # first visited node
    path = str(start)  # first node
    end = sink_node
    temp_qval = dict()
    initial_delay = 0
    tx_energy = 0
    rx_energy = 0

    while True:
        for neigh in node_neighbors[start]:
            reward = math.sqrt(math.pow((xy[start][0] - xy[neigh][0]), 2) + math.pow((xy[start][1] - xy[neigh][1]), 2))

            temp_qval[neigh] = (1 - learning_rate) * path_q_values[start][neigh] + learning_rate * (reward + q_values[neigh])

        copy_q_values = {key: value for key, value in temp_qval.items() if key not in queue}
        # next_hop = min(temp_qval.keys(), key=(lambda k: temp_qval[k]))  #determine the next hop based on the minimum qvalue but ignore the visited node qvalue

        if np.random.random() >= 1 - epsilon:
            # Get action from Q table
            next_hop = random.choice(list(copy_q_values.keys()))
        else:
            # Get random action

            next_hop = min(copy_q_values.keys(), key=(lambda k: copy_q_values[k]))



        queue.append(next_hop)

        path_q_values[start][next_hop] = temp_qval[next_hop]    # update the path qvalue of the next hop
        q_values[start] = temp_qval[next_hop]                   # update the qvalue of the start node

        mean_Qvals = sum([q_values[k] for k in q_values]) / (len(q_values) * max([q_values[k] for k in q_values]))
        dis = math.sqrt(math.pow((xy[start][0] - xy[next_hop][0]), 2) + math.pow((xy[start][1] - xy[next_hop][1]), 2))
        etx = electronic_energy * data_packet_size + amplifier_energy * data_packet_size * math.pow(dis, 2)
        erx = electronic_energy * data_packet_size
        e_values[start] = e_values[start] - etx                      # update the start node energy
        e_values[next_hop] = e_values[next_hop] - erx                # update the next hop energy
        tx_energy += etx
        rx_energy += erx
        initial_delay += dis


        path = path + "->" + str(next_hop)  # update the path after each visit
        #print('path:', path)
        # print("The visited nodes are", queue)

        start = next_hop

        if next_hop == end:
            break

    delay.append(initial_delay)
    E_consumed.append(tx_energy + rx_energy)
    mean_Q.append(mean_Qvals)
    round.append(rdn)

    cost = 0
    dead_node = None
    update_edges = []
    for index, item in e_values.items():

        if item <= node_energy_limit:
            print('Index:', index)
            #print('Qvals:', q_values)
            #print('PQ_vals:', path_q_values)
            cost = cost + 1
            print("Energy of a node has gone behold the threshold!")
            print("The final round is", rdn)

            xy.pop(index)

            dead_node = index

    for ind in list_unweighted_edges:
        if ind[0] != dead_node and ind[1] != dead_node:
            update_edges.append(ind)

    update_evals = {index: item for index, item in e_values.items() if item > node_energy_limit}

    if cost == 1:

        graph, node_neighbors, q_values, e_values, path_q_values = build_graph(xy, update_edges)

        e_values = update_evals
        print('node_neighbours:', node_neighbors)
        print('update_evals:', e_values)

        cost = 0

        for nd in graph.nodes:
            if len(node_neighbors[nd]) == 0:
                break

    list_unweighted_edges = update_edges


'''
plt.plot(round, mean_Q)
plt.xlabel('Round')
plt.ylabel('Average Q-Value')
plt.title('Q-Value Convergence ')
plt.show()

plt.plot(round, delay)
plt.xlabel('Round')
plt.ylabel('Delay (s)')
plt.title('Delay for each round')
plt.show()

plt.plot(round, E_consumed)
plt.xlabel('Round')
plt.ylabel('Energy Consumption (Joules)')
plt.title('Energy Consumption for each round')
plt.show()
'''




