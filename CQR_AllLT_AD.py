import numpy as np
import networkx as nx
import random
import math
import matplotlib.pyplot as plt
import time
from AllMst import Yamada

transmission_range = 50  # meters
def build_graph(positions, links):
    G = nx.Graph()
    for i in positions:
        G.add_node(i, pos=positions[i])
    # print('graph nodes:', G.nodes)
    global position_array
    position_array = {}
    for nd in G:
        position_array[nd] = G.nodes[nd]['pos']

    for u, v in links:
        #G.add_edge(u, v, weight=(math.ceil(math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow((position_array[u][1] - position_array[v][1]), 2))) / transmission_range))
        distance = math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow((position_array[u][1] - position_array[v][1]), 2))
        nor_distance = math.ceil(distance / transmission_range)
        G.add_edge(u, v, weight=nor_distance)

    z = Yamada(graph = G, n_trees = np.inf)
    all_msts = z.spanning_trees()
    node_neigh = []
    for T in all_msts:
        node_neighT = {}
        for n in T.nodes:
            node_neighT[n] = list(T.neighbors(n))
        node_neigh.append(node_neighT)
    MSTs_hop_count = []
    MST_paths = []
    for T in all_msts:
        hop_counts = {}
        MST_path = {}
        for n in T.nodes:
            for path in nx.all_simple_paths(T, source=n, target=sink_node):
                hop_counts[n] = len(path) - 1
                MST_path[n] = path
        hop_counts[sink_node] = 0  # hop count of sink
        MSTs_hop_count.append(hop_counts)
        MST_paths.append(MST_path)

    # Q_matrix of all Msts
    Q_matrix = np.zeros((len(all_msts), len(all_msts)))
    # Energy consumption
    e_vals = {}

    for idx in G.nodes:
        if idx != sink_node:
            e_vals[idx] = initial_energy
        else:
            e_vals[idx] = sink_node_energy

    return G, all_msts, MST_paths, e_vals, Q_matrix
'''
xy = {0: (1, 3), 1: (2.5, 5), 2: (2.5, 1), 3: (4.5, 5), 4: (4.5, 1), 5: (6, 3)}

# Adding unweighted edges to the Graph
list_unweighted_edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]

'''
list_unweighted_edges = [(54, 100), (96, 14), (39, 93), (91, 53), (21, 78), (95, 72), (25, 65), (22, 32), (33, 2), (23, 68), (55, 69), (81, 97), (46, 54), (55, 67), (70, 20), (26, 38), (28, 48), (17, 35), (74, 96), (61, 53), (83, 48), (4, 5), (77, 51), (35, 15), (30, 57), (83, 90), (77, 44), (67, 39), (68, 52), (57, 52), (23, 54), (47, 61), (61, 33), (36, 49), (32, 55), (14, 41), (87, 63), (2, 79), (81, 89), (43, 57), (78, 29), (22, 87), (12, 62), (29, 73), (23, 66), (71, 99), (38, 33), (85, 13), (86, 88), (19, 46), (31, 45), (71, 45), (41, 12), (62, 51), (40, 8), (0, 61), (22, 18), (0, 34), (96, 11), (77, 10), (70, 78), (63, 75), (3, 95), (84, 16), (32, 86), (16, 85), (24, 90), (11, 83), (73, 33), (48, 2), (47, 98), (31, 34), (64, 52), (87, 58), (6, 96), (66, 40), (50, 14), (12, 10), (1, 84), (27, 31), (3, 70), (63, 19), (8, 61), (90, 9), (26, 49), (48, 6), (70, 17), (38, 50), (22, 41), (86, 24), (61, 59), (82, 48), (92, 40), (79, 68), (58, 89), (83, 32), (62, 72), (29, 12), (79, 71), (21, 76), (30, 29), (92, 96), (98, 51), (8, 27), (100, 20), (7, 43), (0, 9), (92, 90), (65, 64), (65, 50), (75, 72), (42, 40), (73, 63), (80, 33), (56, 78), (8, 27), (98, 12), (69, 65), (0, 65), (93, 46), (78, 24), (59, 31), (61, 89), (71, 60), (90, 76), (62, 29), (49, 77), (99, 95), (70, 39), (93, 100), (86, 68), (70, 44), (31, 59), (51, 77), (27, 34), (46, 69), (60, 6), (99, 62), (65, 60), (20, 61), (84, 92), (4, 43), (92, 39), (3, 61), (24, 23), (86, 72), (71, 8), (57, 65), (17, 94), (4, 40), (56, 88), (44, 43), (47, 92), (52, 8), (49, 79), (92, 60), (60, 84), (25, 89), (27, 63), (72, 69), (85, 52), (73, 25), (85, 20), (0, 2), (27, 4), (61, 59), (32, 2), (3, 21), (17, 2), (64, 48), (54, 80), (4, 21), (47, 80), (88, 22), (78, 17), (29, 66), (32, 81), (27, 89), (97, 76), (20, 28), (80, 48), (45, 24), (49, 76), (55, 79), (27, 62), (53, 28), (33, 35), (99, 43), (37, 100), (33, 54), (82, 75), (58, 8), (15, 90), (71, 63), (24, 55), (54, 97), (97, 2), (91, 84), (84, 2), (59, 72), (38, 90), (72, 69), (88, 49), (6, 31), (44, 54), (31, 61), (88, 94), (66, 28), (18, 76), (98, 51), (94, 62), (32, 6), (78, 65), (79, 31), (60, 13), (10, 28), (25, 74), (66, 39), (44, 25), (33, 25), (17, 24), (46, 39), (43, 20), (13, 57), (47, 87), (13, 28), (59, 18), (36, 91), (28, 97), (26, 27), (30, 21), (58, 52), (71, 25), (85, 61), (59, 8), (85, 9), (42, 84), (94, 5), (71, 80), (6, 84), (96, 72), (52, 81), (28, 5), (8, 18), (59, 52), (73, 77), (76, 34), (41, 89), (39, 97)]
xy2 = [(727, 333), (410, 921), (369, 283), (943, 142), (423, 646), (153, 477), (649, 828), (911, 989), (972, 422), (35, 419), (648, 836), (17, 688), (281, 402), (344, 909), (815, 675), (371, 908), (748, 991), (838, 45), (462, 505), (508, 474), (565, 617), (2, 979), (392, 991), (398, 265), (789, 35), (449, 952), (88, 281), (563, 839), (128, 725), (639, 35), (545, 329), (259, 294), (379, 907), (830, 466), (620, 290), (789, 579), (778, 453), (667, 663), (665, 199), (844, 732), (105, 884), (396, 411), (351, 452), (488, 584), (677, 94), (743, 659), (752, 203), (108, 318), (941, 691), (981, 702), (100, 701), (783, 822), (250, 788), (96, 902), (540, 471), (449, 473), (671, 295), (870, 246), (588, 102), (703, 121), (402, 637), (185, 645), (808, 10), (668, 617), (467, 852), (280, 39), (563, 377), (675, 334), (429, 177), (494, 637), (430, 831), (57, 726), (509, 729), (376, 311), (429, 833), (395, 417), (628, 792), (512, 259), (845, 729), (456, 110), (277, 501), (211, 996), (297, 689), (160, 87), (590, 605), (498, 557), (971, 211), (562, 326), (315, 963), (316, 471), (390, 316), (365, 755), (573, 631), (881, 532), (969, 218), (220, 388), (517, 500), (869, 670), (490, 575), (331, 992), (500, 500)]
xy = {}

for index in range(len(xy2)):
    xy[index] = xy2[index]


# initialization of network parameters

discount_factor = 0
learning_rate = 0.7
initial_energy = 100  # Joules
sink_node_energy = 200000  # Joules
data_packet_size = 512  # bits
electronic_energy = 50e-9  # Joules/bit 5
e_fs = 10e-12  # Joules/bit/(meter)**2
e_mp = 0.0013e-12 #Joules/bit/(meter)**4

node_energy_limit = 10
sink_node = 100

epsilon = 0.1
d_o = math.sqrt(e_fs/e_mp)
"""
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = epsilon // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

"""
episodes = 5000000

Q_value = []
Min_value = []
Actions = []
Episode = []
Average_Delay = []
E_consumed = []
EE_consumed = []
No_Alive_Node = []

graph, rts, rtp, E_vals, q_matrix = build_graph(xy, list_unweighted_edges)
#print('len_rts:', len(rtp))


start_time = time.time()

for rdn in range(episodes):

    # print("Edges:", list_unweighted_edges)

    initial_state = random.choice(range(0, len(rts), 1))
    delay = 0
    tx_energy = 0
    rx_energy = 0
    Episode.append(rdn)
    available_actions = [*range(0, len(rts), 1)]

    current_state = initial_state

    if np.random.random() > epsilon:
        # Get random action
        action = np.argmin(q_matrix[current_state, :])

    else:
        # Get action from Q table
        action = random.choice(available_actions)

    #print('actions:', action)
    Actions.append(action + 1)
    # print('Actions:', Actions)

    initial_state = action
    # print('action is:', action)

    chosen_MST = rtp[action]
    #print('chosen MST:', chosen_MST)
    Delay = []
    # Data transmission
    for node in chosen_MST:
        counter = 0
        while counter < len(chosen_MST[node]) - 1:
            init_node = chosen_MST[node][counter]
            next_node = chosen_MST[node][counter + 1]
            dis = math.sqrt(math.pow((xy[init_node][0] - xy[next_node][0]), 2) + math.pow((xy[init_node][1] - xy[next_node][1]), 2))
            nor_dis = math.ceil(dis / transmission_range)
            if nor_dis <= d_o:
                etx = electronic_energy * data_packet_size + e_fs * data_packet_size * math.pow(nor_dis, 2)
            else:
                etx = electronic_energy * data_packet_size + e_mp * data_packet_size * math.pow(nor_dis, 4)
            erx = electronic_energy * data_packet_size
            E_vals[init_node] = E_vals[init_node] - etx  # update the start node energy
            E_vals[next_node] = E_vals[next_node] - erx  # update the next hop energy
            tx_energy += etx
            rx_energy += erx
            delay += dis
            counter += 1
        Delay.append(delay)
        # print("counter", counter)
    #reward = E_vals[min(E_vals.keys(), key=(lambda k: E_vals[k]))]

    reward = tx_energy + rx_energy

    Min_value.append(reward)

    # Maximum possible Q value in next step (for new state)
    max_future_q = np.max(q_matrix[action, :])

    # Current Q value (for current state and performed action)
    current_q = q_matrix[current_state, action]
    # And here's our equation for a new Q value for current state and action
    new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
    # new_q = (1 - learning_rate) * current_q + learning_rate * discount_factor *reward
    Q_value.append(new_q)

    # Update Q table with new Q value
    q_matrix[current_state, action] = new_q

    Average_Delay.append(sum(Delay) / len(Delay))
    E_consumed.append(tx_energy + rx_energy )
    No_Alive_Node.append(len(xy) - 1)

    dead_node = []

    for index, item in E_vals.items():

        if item <= node_energy_limit:

            dead_node.append(index)

            if index in xy.keys():
                xy.pop(index)


    test = [(item1, item2) for item1,item2 in list_unweighted_edges if item1 not in dead_node and item2 not in dead_node]
    

    list_unweighted_edges = test

    update_evals = {index: item for index, item in E_vals.items() if item > node_energy_limit}


    # print('Original edges:', list_unweighted_edges)
    if len(dead_node) >= 1:

        try:
            graph, rts, rtp, E_vals, q_matrix = build_graph(xy, list_unweighted_edges)

            E_vals = update_evals
            #update_qmatrix = np.ones((len(rts), len(rts)))
            #q_matrix = update_qmatrix * new_q

        except ValueError:
            print('lifetime:', rdn)
            break

    # Decaying is being done every episode if episode number is within decaying range
    """
    if END_EPSILON_DECAYING >= episodes >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    """

    dead_node = []
    #print('Q-matrix:', q_matrix)

# print('Actions:', Actions)
# print('Reward:', Min_value)

print("--- %s seconds ---" % (time.time() - start_time))

# print('Round:', Episode)
# print('Average_Delay:', Average_Delay)
# print('Total Energy:', EE_consumed)
# print('Energy:', E_consumed)
# print('QVals:', Q_value)
'''
plt.plot(Episode, Q_value, label="Q-Value")
plt.plot(Episode, Min_value, label="Reward")
plt.xlabel('Round')
plt.ylabel('Reward (Joules)')
plt.title('Q-Value Convergence')
plt.legend()
plt.show()

plt.plot(Episode, Actions)
plt.xlabel('Round')
plt.ylabel('Discrete Action')
plt.title('Selected Action for each round')
plt.show()
plt.plot(Episode, Average_Delay)
plt.xlabel('Round')
plt.ylabel('Average_Delay (s)')
# plt.title('Delay for each round')
plt.show()

'''
plt.plot(Episode, No_Alive_Node)
plt.xlabel('Round')
plt.ylabel('NAN')
plt.title('Number of Alive Nodes in each Round')
plt.show()

plt.plot(Episode, E_consumed)
plt.xlabel('Round')
plt.ylabel('Energy Consumption (Joules)')
# plt.title('Energy Consumption for each round')
plt.show()


