import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random


# initialization of network parameters
learning_rate = 0.7
initial_energy = 100  # Joules
sink_node_energy = 200000
data_packet_size = 512  # bits
electronic_energy = 50e-9  # Joules/bit 50e-9
e_fs = 10e-12  # Joules/bit/(meter)**2
e_mp = 0.0013e-12 #Joules/bit/(meter)**4
node_energy_limit = 10
epsilon = 0.0
transmission_range = 50
sink_node = 100
num_of_episodes = 5000000

#xy = {0: (1, 3), 1: (2.5, 5), 2: (2.5, 1), 3: (4.5, 5), 4: (4.5, 1), 5: (6, 3)}
#list_unweighted_edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]

xy = {0: (727, 333), 1: (410, 921), 2: (369, 283), 3: (943, 142), 4: (423, 646), 5: (153, 477), 6: (649, 828), 7: (911, 989), 8: (972, 422), 9: (35, 419), 10: (648, 836), 11: (17, 688), 12: (281, 402), 13: (344, 909), 14: (815, 675), 15: (371, 908), 16: (748, 991), 17: (838, 45), 18: (462, 505), 19: (508, 474), 20: (565, 617), 21: (2, 979), 22: (392, 991), 23: (398, 265), 24: (789, 35), 25: (449, 952), 26: (88, 281), 27: (563, 839), 28: (128, 725), 29: (639, 35), 30: (545, 329), 31: (259, 294), 32: (379, 907), 33: (830, 466), 34: (620, 290), 35: (789, 579), 36: (778, 453), 37: (667, 663), 38: (665, 199), 39: (844, 732), 40: (105, 884), 41: (396, 411), 42: (351, 452), 43: (488, 584), 44: (677, 94), 45: (743, 659), 46: (752, 203), 47: (108, 318), 48: (941, 691), 49: (981, 702), 50: (100, 701), 51: (783, 822), 52: (250, 788), 53: (96, 902), 54: (540, 471), 55: (449, 473), 56: (671, 295), 57: (870, 246), 58: (588, 102), 59: (703, 121), 60: (402, 637), 61: (185, 645), 62: (808, 10), 63: (668, 617), 64: (467, 852), 65: (280, 39), 66: (563, 377), 67: (675, 334), 68: (429, 177), 69: (494, 637), 70: (430, 831), 71: (57, 726), 72: (509, 729), 73: (376, 311), 74: (429, 833), 75: (395, 417), 76: (628, 792), 77: (512, 259), 78: (845, 729), 79: (456, 110), 80: (277, 501), 81: (211, 996), 82: (297, 689), 83: (160, 87), 84: (590, 605), 85: (498, 557), 86: (971, 211), 87: (562, 326), 88: (315, 963), 89: (316, 471), 90: (390, 316), 91: (365, 755), 92: (573, 631), 93: (881, 532), 94: (969, 218), 95: (220, 388), 96: (517, 500), 97: (869, 670), 98: (490, 575), 99: (331, 992), 100: (500, 500)}

list_unweighted_edges = [(54, 100), (96, 14), (39, 93), (91, 53), (21, 78), (95, 72), (25, 65), (22, 32), (33, 2), (23, 68), (55, 69), (81, 97), (46, 54), (55, 67), (70, 20), (26, 38), (28, 48), (17, 35), (74, 96), (61, 53), (83, 48), (4, 5), (77, 51), (35, 15), (30, 57), (83, 90), (77, 44), (67, 39), (68, 52), (57, 52), (23, 54), (47, 61), (61, 33), (36, 49), (32, 55), (14, 41), (87, 63), (2, 79), (81, 89), (43, 57), (78, 29), (22, 87), (12, 62), (29, 73), (23, 66), (71, 99), (38, 33), (85, 13), (86, 88), (19, 46), (31, 45), (71, 45), (41, 12), (62, 51), (40, 8), (0, 61), (22, 18), (0, 34), (96, 11), (77, 10), (70, 78), (63, 75), (3, 95), (84, 16), (32, 86), (16, 85), (24, 90), (11, 83), (73, 33), (48, 2), (47, 98), (31, 34), (64, 52), (87, 58), (6, 96), (66, 40), (50, 14), (12, 10), (1, 84), (27, 31), (3, 70), (63, 19), (8, 61), (90, 9), (26, 49), (48, 6), (70, 17), (38, 50), (22, 41), (86, 24), (61, 59), (82, 48), (92, 40), (79, 68), (58, 89), (83, 32), (62, 72), (29, 12), (79, 71), (21, 76), (30, 29), (92, 96), (98, 51), (8, 27), (100, 20), (7, 43), (0, 9), (92, 90), (65, 64), (65, 50), (75, 72), (42, 40), (73, 63), (80, 33), (56, 78), (8, 27), (98, 12), (69, 65), (0, 65), (93, 46), (78, 24), (59, 31), (61, 89), (71, 60), (90, 76), (62, 29), (49, 77), (99, 95), (70, 39), (93, 100), (86, 68), (70, 44), (31, 59), (51, 77), (27, 34), (46, 69), (60, 6), (99, 62), (65, 60), (20, 61), (84, 92), (4, 43), (92, 39), (3, 61), (24, 23), (86, 72), (71, 8), (57, 65), (17, 94), (4, 40), (56, 88), (44, 43), (47, 92), (52, 8), (49, 79), (92, 60), (60, 84), (25, 89), (27, 63), (72, 69), (85, 52), (73, 25), (85, 20), (0, 2), (27, 4), (61, 59), (32, 2), (3, 21), (17, 2), (64, 48), (54, 80), (4, 21), (47, 80), (88, 22), (78, 17), (29, 66), (32, 81), (27, 89), (97, 76), (20, 28), (80, 48), (45, 24), (49, 76), (55, 79), (27, 62), (53, 28), (33, 35), (99, 43), (37, 100), (33, 54), (82, 75), (58, 8), (15, 90), (71, 63), (24, 55), (54, 97), (97, 2), (91, 84), (84, 2), (59, 72), (38, 90), (72, 69), (88, 49), (6, 31), (44, 54), (31, 61), (88, 94), (66, 28), (18, 76), (98, 51), (94, 62), (32, 6), (78, 65), (79, 31), (60, 13), (10, 28), (25, 74), (66, 39), (44, 25), (33, 25), (17, 24), (46, 39), (43, 20), (13, 57), (47, 87), (13, 28), (59, 18), (36, 91), (28, 97), (26, 27), (30, 21), (58, 52), (71, 25), (85, 61), (59, 8), (85, 9), (42, 84), (94, 5), (71, 80), (6, 84), (96, 72), (52, 81), (28, 5), (8, 18), (59, 52), (73, 77), (76, 34), (41, 89), (39, 97)]

d_o = math.ceil(math.sqrt(e_fs/e_mp)/transmission_range)

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
        distance = math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow((position_array[u][1] - position_array[v][1]), 2))
        nor_distance = math.ceil(distance / transmission_range)
        G.add_edge(u, v, weight=nor_distance)
    # The dictionary of neighbors of all nodes in the graph
    node_neigh = {}
    for n in G.nodes:
        node_neigh[n] = list(G.neighbors(n))

    q_vals = {}
    for ix in G.nodes:
        q_vals[ix] = 0

    all_q_vals = {}
    for iix in G.nodes:
        all_q_vals[iix] = q_vals

    path_q_vals = {}
    for xi in G.nodes:
        path_q_vals[xi] = q_vals

    all_path_q_vals = {}
    for xii in G.nodes:
        all_path_q_vals[xii] = path_q_vals

    # Energy consumption
    e_vals = {}

    for idx in G.nodes:
        if idx != sink_node:
            e_vals[idx] = initial_energy
        else:
            e_vals[idx] = sink_node_energy

    return G, node_neigh, all_q_vals, e_vals, all_path_q_vals


Av_mean_Q = []
Av_E_consumed = []
Av_delay = []
No_Alive_Node = []
round = []

graph, node_neighbors, q_values, e_values, path_q_values = build_graph(xy, list_unweighted_edges)

for rdn in range(num_of_episodes):

    mean_Q = []
    E_consumed = []
    EE_consumed = []
    delay = []
    path_f = []


    for node in graph.nodes:
        if node != sink_node:
            start = node
            queue = [start]  # first visited node
            path = str(start)  # first node
            temp_qval = dict()
            initial_delay = 0
            tx_energy = 0
            rx_energy = 0


            while True:
                for neigh in node_neighbors[start]:
                    reward = math.sqrt(math.pow((xy[start][0] - xy[neigh][0]), 2) + math.pow((xy[start][1] - xy[neigh][1]), 2))

                    temp_qval[neigh] = (1 - learning_rate) * path_q_values[node][start][neigh] + learning_rate * (reward + q_values[node][neigh])

                copy_q_values = {key: value for key, value in temp_qval.items() if key not in queue}
                # next_hop = min(temp_qval.keys(), key=(lambda k: temp_qval[k]))  #determine the next hop based on the minimum qvalue but ignore the visited node qvalue

                if np.random.random() >= 1 - epsilon:
                    # Get action from Q table
                    next_hop = random.choice(list(copy_q_values.keys()))
                else:
                    # Get random action

                    next_hop = min(copy_q_values.keys(), key=(lambda k: copy_q_values[k]))

                queue.append(next_hop)

                path_q_values[node][start][next_hop] = temp_qval[next_hop]    # update the path qvalue of the next hop
                q_values[node][start] = temp_qval[next_hop]                   # update the qvalue of the start node

                mean_Qvals = sum([q_values[node][k] for k in q_values[node]]) / (len(q_values[node]) * max([q_values[node][k] for k in q_values[node]]))
                dis = math.sqrt(math.pow((xy[start][0] - xy[next_hop][0]), 2) + math.pow((xy[start][1] - xy[next_hop][1]), 2))
                nor_dis = math.ceil(dis / transmission_range)
                if nor_dis <= d_o:
                    etx = electronic_energy * data_packet_size + e_fs * data_packet_size * math.pow(nor_dis, 2)
                else:
                    etx = electronic_energy * data_packet_size + e_mp * data_packet_size * math.pow(nor_dis, 4)
                erx = electronic_energy * data_packet_size
                e_values[start] = e_values[start] - etx                      # update the start node energy
                e_values[next_hop] = e_values[next_hop] - erx                # update the next hop energy
                tx_energy += etx
                rx_energy += erx
                initial_delay += dis

                path = path + "->" + str(next_hop)  # update the path after each visit

                start = next_hop

                if next_hop == sink_node:
                    break

            delay.append(initial_delay)
            E_consumed.append(tx_energy + rx_energy)
            mean_Q.append(mean_Qvals)

        #path_f.append(path)
        Av_mean_Q.append(sum(mean_Q) / len(mean_Q))
        Av_delay.append(sum(delay) / len(delay))
        Av_E_consumed.append(sum(E_consumed))
        No_Alive_Node.append(len(graph.nodes) - 1)
        round.append(rdn)


    dead_node = []

    for index, item in e_values.items():

        if item <= node_energy_limit:


            dead_node.append(index)

            if index in xy.keys():
                xy.pop(index)

    test = [(item1, item2) for item1, item2 in list_unweighted_edges if item1 not in dead_node and item2 not in dead_node]

    list_unweighted_edges = test

    update_evals = {index: item for index, item in e_values.items() if item > node_energy_limit}

    if len(dead_node) >= 1:
        #print('Energy of node has gone below a threshold')
        #print('dead nodes:', dead_node)
        #print("The lifetime at this point is", rdn)


        try:
            graph, node_neighbors, q_values, e_values, path_q_values = build_graph(xy, list_unweighted_edges)

            e_values = update_evals
            #print('new nodes:', xy)
            #print('Updated Evals:', update_evals)
            #print('updated_node_neighbours:', node_neighbors)

        except ValueError:

            break

    profit = True

    for nd in graph.nodes:
        if node_neighbors[nd] == [] or len(graph.nodes) == 1:
            profit = False

    if not profit:
        print('lifetime:', rdn)
        break


plt.plot(round, Av_mean_Q)
plt.xlabel('Round')
plt.ylabel('Average Q-Value')
plt.title('Q-Value Convergence ')
plt.show()

plt.plot(round, Av_delay)
plt.xlabel('Round')
plt.ylabel('Delay (s)')
plt.title('Delay for each round')
plt.show()

plt.plot(round, Av_E_consumed)
plt.xlabel('Round')
plt.ylabel('Energy Consumption (Joules)')
plt.title('Energy Consumption for each round')
plt.show()

plt.plot(round, No_Alive_Node)
plt.xlabel('Round')
plt.ylabel('NAN')
plt.title('No of Alive Nodes in each round')
plt.show()






