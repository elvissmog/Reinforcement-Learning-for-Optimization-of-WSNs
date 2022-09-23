import numpy as np
import networkx as nx
import random
import math
import time
from AllMst import Yamada
import json

start_time = time.time()

sink_node = 1000


def build_graph(positions, links):
    G = nx.Graph()
    for i in positions:
        G.add_node(i, pos=positions[i])

    global position_array
    position_array = {}
    for nd in G:
        position_array[nd] = G.nodes[nd]['pos']

    for u, v in links:
        distance = math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow(
            (position_array[u][1] - position_array[v][1]), 2))
        nor_distance = math.ceil(distance / transmission_range)
        G.add_edge(u, v, weight=nor_distance)

    z = Yamada(graph=G, n_trees=10)  # np.inf
    all_msts = z.spanning_trees()
    node_neigh = []
    for T in all_msts:
        node_neighT = {}
        for n in T.nodes:
            node_neighT[n] = list(T.neighbors(n))
        node_neigh.append(node_neighT)

    MST_paths = []
    for T in all_msts:

        MST_path = {}
        for n in T.nodes:
            for path in nx.all_simple_paths(T, source=n, target=sink_node):
                MST_path[n] = path

        MST_paths.append(MST_path)

    # Q_matrix of all Msts
    Q_matrix = np.zeros((len(all_msts), len(all_msts)))
    # Energy consumption

    initial_e_vals = {}
    for idx in G.nodes:
        if idx != sink_node:
            initial_e_vals[idx] = initial_energy
        else:
            initial_e_vals[idx] = sink_energy

    ref_e_vals = {}
    for idx in G.nodes:
        if idx != sink_node:
            ref_e_vals[idx] = initial_energy
        else:
            ref_e_vals[idx] = sink_energy




    return G, all_msts, MST_paths, initial_e_vals, ref_e_vals, Q_matrix


with open('edges.txt', 'r') as filehandle:
    edges_list = json.load(filehandle)

list_unweighted_edges = []
for ed in edges_list:
    list_unweighted_edges.append(tuple(ed))

with open('pos.txt', 'r') as filehandle:
    pos_list = json.load(filehandle)

x_y = []
for ps in pos_list:
    x_y.append(tuple(ps))

xy = {}
for index in range(len(x_y)):
    xy[index] = x_y[index]

# initialization of network parameters
discount_factor = 0
learning_rate = 0.9
initial_energy = 10  # Joules
data_packet_size = 1024  # bits
electronic_energy = 50e-9  # Joules/bit 5
transmission_range = 1
e_fs = 10e-12  # Joules/bit/(meter)**2
e_mp = 0.0013e-12  # Joules/bit/(meter)**4
node_energy_limit = 1
epsilon = 0.1
episodes = 5000000
sink_energy = 5000000
num_pac = 1

d_o = math.sqrt(e_fs / e_mp) / transmission_range

#Cum_reward = []
#Q_value = []
#Min_value = []
#Episode = []
#E_consumed = []
#EE_consumed = []
No_Alive_Node = []

graph, rts, rtp, initial_E_vals, ref_E_vals, q_matrix = build_graph(xy, list_unweighted_edges)

for rdn in range(episodes):

    initial_state = random.choice(range(0, len(rts), 1))
    tx_energy = 0
    rx_energy = 0
    #Episode.append(rdn)
    available_actions = [*range(0, len(rts), 1)]

    current_state = initial_state

    if np.random.random() > epsilon:
        # Get random action
        action = np.argmin(q_matrix[current_state, :])

    else:
        # Get action from Q table
        action = random.choice(available_actions)

    initial_state = action

    chosen_MST = rtp[action]
    # print('chosen MST:', chosen_MST)

    # Data transmission
    for node in chosen_MST:
        counter = 0
        while counter < len(chosen_MST[node]) - 1:
            init_node = chosen_MST[node][counter]
            next_node = chosen_MST[node][counter + 1]
            dis = math.sqrt(math.pow((xy[init_node][0] - xy[next_node][0]), 2) + math.pow((xy[init_node][1] - xy[next_node][1]), 2)) / transmission_range
            if dis <= d_o:
                etx = electronic_energy * data_packet_size + e_fs * data_packet_size * math.pow(dis, 2)
            else:
                etx = electronic_energy * data_packet_size + e_mp * data_packet_size * math.pow(dis, 4)
            erx = electronic_energy * data_packet_size

            initial_E_vals[init_node] = initial_E_vals[init_node] - num_pac * etx  # update the start node energy

            initial_E_vals[next_node] = initial_E_vals[next_node] - num_pac * erx  # update the next hop energy

            tx_energy += num_pac * etx

            rx_energy += num_pac * erx

            counter += 1

    # reward = initial_E_vals[max(initial_E_vals.keys(), key=(lambda k: initial_E_vals[k]))]


    Energy_Consumption = [ref_E_vals[i] - initial_E_vals[i] for i in graph.nodes if i != sink_node]
    reward = max(Energy_Consumption)

    #Min_value.append(reward)
    #Cum_reward.append(sum(Min_value))

    # Maximum possible Q value in next step (for new state)
    max_future_q = np.max(q_matrix[action, :])

    # Current Q value (for current state and performed action)
    current_q = q_matrix[current_state, action]
    # And here's our equation for a new Q value for current state and action
    new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
    # new_q = (1 - learning_rate) * current_q + learning_rate * discount_factor *reward
    #Q_value.append(new_q)

    # Update Q table with new Q value
    q_matrix[current_state, action] = new_q

    #E_consumed.append(tx_energy + rx_energy)
    #EE_consumed.append(sum(E_consumed))
    No_Alive_Node.append(len(xy) - 1)
    cost = 0

    dead_node = []

    for index, item in initial_E_vals.items():

        if item <= node_energy_limit:

            dead_node.append(index)

            if index in xy.keys():
                xy.pop(index)

    test = [(item1, item2) for item1, item2 in list_unweighted_edges if item1 not in dead_node and item2 not in dead_node]


    list_unweighted_edges = test

    update_evals = {index: item for index, item in initial_E_vals.items() if item > node_energy_limit}

    if len(dead_node) >= 1:
        #print('Round:', rdn)
        #print('Dead node:', dead_node)

        try:
            graph, rts, rtp, initial_E_vals, ref_E_vals, q_matrix = build_graph(xy, list_unweighted_edges)

            initial_E_vals = update_evals
            ref_E_vals = initial_E_vals
            update_qmatrix = np.ones((len(rts), len(rts)))
            q_matrix = update_qmatrix * new_q

        except ValueError:
            # Error = messagebox.showinfo("Enter proper values")
            print('lifetime:', rdn)
            break

    dead_node = []

print("--- %s seconds ---" % (time.time() - start_time))

with open('onan.txt', 'w') as f:
    f.write(json.dumps(No_Alive_Node))

# Now read the file back into a Python list object
with open('onan.txt', 'r') as f:
    No_Alive_Node = json.loads(f.read())

