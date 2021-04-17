from tkinter import messagebox

import numpy as np
import networkx as nx
import random
import math
import matplotlib.pyplot as plt

import time

from AllMst import Yamada


sink_node = 5

def build_graph(positions, links):
    G = nx.Graph()
    for i in positions:
        G.add_node(i, pos=positions[i])
    print('graph nodes:', G.nodes)
    global position_array
    position_array = {}
    for nd in G:
        position_array[nd] = G.nodes[nd]['pos']
    print('Position array:', position_array)

    for u, v in links:
        # G.add_edge(u, v, weight = 1)
        G.add_edge(u, v, weight = math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow((position_array[u][1] - position_array[v][1]), 2)))
    z = Yamada(graph = G, n_trees=100)
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

    #Energy consumption
    e_vals = {}

    for idx in G.nodes:
    	if idx == sink_node:
    		e_vals[idx] = 50

    	e_vals[idx] = initial_energy


    return G, all_msts, MST_paths, e_vals


xy = {0: (1, 3), 1: (2.5, 5), 2: (2.5, 1), 3: (4.5, 5), 4: (4.5, 1), 5: (6, 3)}


# Adding unweighted edges to the Graph
list_unweighted_edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]


# initialization of network parameters

discount_factor = 0
learning_rate = 0.7
initial_energy = 0.1  # Joules
control_packet_size = 96 #bits
data_packet_size = 216  # bits
electronic_energy = 50e-9  # Joules/bit 5
amplifier_energy = 100e-12  # Joules/bit/square meter
transmission_range = 30  # meters
packet_gen_rate = 2    # per seconds
learning_period = 10    # secs


epsilon = 0.1
episodes = 10000




Q_value = []
Min_value = []
Actions = []
Episode = []
Average_Delay = []
E_consumed = []
EE_consumed = []

graph, rts, rtp, E_vals = build_graph(xy, list_unweighted_edges)

"""
E_vals = {}
for idx in graph.nodes:
    if idx == sink_node:
        E_vals[idx] = 50
    else:
        E_vals[idx] = initial_energy
"""
for rdn in range(episodes):

    Q_matrix = np.zeros((len(rts), len(rts)))
    initial_state = random.choice(range(0, len(rts), 1))
    delay = 0
    tx_energy = 0
    rx_energy = 0
    ctx_energy = 0
    crx_energy = 0
    Episode.append(rdn)
    available_actions = [*range(0, len(rts), 1)]

    current_state = initial_state

    if np.random.random() >= 1 - epsilon:
        # Get action from Q table
        action = random.choice(available_actions)
    else:
        # Get random action
        action = np.argmax(Q_matrix[current_state, :])

    y = action + 1
    Actions.append(y)
    #print('Actions:', Actions)

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
            dis = math.sqrt(math.pow((position_array[init_node][0] - position_array[next_node][0]), 2) + math.pow((position_array[init_node][1] - position_array[next_node][1]), 2))
            etx = electronic_energy * data_packet_size + amplifier_energy * data_packet_size  * math.pow(dis, 2)
            erx = electronic_energy * data_packet_size
            E_vals[init_node] = E_vals[init_node] - etx  # update the start node energy
            E_vals[next_node] = E_vals[next_node] - erx  # update the next hop energy
            tx_energy += etx
            rx_energy += erx
            delay += dis
            counter += 1
        Delay.append(delay)
            #print("counter", counter)
    reward = E_vals[min(E_vals.keys(), key=(lambda k: E_vals[k]))]

    Min_value.append(reward)

    #print('Min Val:', Min_value)

    # Status Update
    for node in chosen_MST:
        counter = 0
        while counter < len(chosen_MST[node]) - 1:
            init_node = chosen_MST[node][counter]
            next_node = chosen_MST[node][counter + 1]
            dis = math.sqrt(math.pow((position_array[init_node][0] - position_array[next_node][0]), 2) + math.pow((position_array[init_node][1] - position_array[next_node][1]), 2))
            cetx = electronic_energy * data_packet_size + amplifier_energy * control_packet_size * math.pow(dis, 2)
            cerx = electronic_energy * control_packet_size
            E_vals[init_node] = E_vals[init_node] - cetx  # update the start node energy
            E_vals[next_node] = E_vals[next_node] - cerx  # update the next hop energy
            ctx_energy += cetx
            crx_energy += cerx
            counter += 1



    #reward = min(E_vals) / sum(E_vals)

    # Maximum possible Q value in next step (for new state)
    max_future_q = np.max(Q_matrix[action, :])

    # Current Q value (for current state and performed action)
    current_q = Q_matrix[current_state, action]
    # And here's our equation for a new Q value for current state and action
    new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
    #new_q = (1 - learning_rate) * current_q + learning_rate * discount_factor *reward
    Q_value.append(new_q)

    # Update Q table with new Q value
    Q_matrix[current_state, action] = new_q

    Average_Delay.append(sum(Delay)/len(Delay))
    E_consumed.append(tx_energy + rx_energy + ctx_energy + crx_energy)
    EE_consumed.append(sum(E_consumed))

    cost = 0
    dead_node = None
    update_edges = []
    for index, item in E_vals.items():

        if item <= 0:
            print('Index:', index)
            print('Evals:', E_vals)
            cost = cost + 1
            print("Energy cannot be negative!")
            print("The final round is", rdn)
            #yz = xy.copy()
            #yz.pop(index)
            xy.pop(index)
            #E_vals.pop(index)
            dead_node = index
    for ind in list_unweighted_edges:
        if ind[0] != dead_node and ind[1] != dead_node:
            update_edges.append(ind)

    

    #print('E_vals:', E_vals)
    #print('E_tx:', Etx)
    #print('Episode:',round)



    if cost == 1:
        print('cost:', cost)
        print('new nodes:', xy)
        print('Updated edges:', update_edges)
        print('Original edges:', list_unweighted_edges)
        try:
            graph, rts, rtp, E_vals = build_graph(xy, update_edges)
        except ValueError:
            #Error = messagebox.showinfo("Enter proper values")
            break


        print('new_nodes:', graph.nodes)
        print('new_edges:', graph.edges)
        cost = 0
    #graph, distances = ngraph, ndistances

    list_unweighted_edges = update_edges



print('Actions:', Actions)
print('Reward:', Min_value)

#print("--- %s seconds ---" % (time.time() - start_time))

print('Round:', Episode)
print('Average_Delay:', Average_Delay)
print('Total Energy:', EE_consumed)
print('Energy:', E_consumed)
print('QVals:', Q_value)

plt.plot(Episode, Q_value, label="Q-Value")
plt.plot(Episode, Min_value, label="Reward")
plt.xlabel('Round')
plt.ylabel('Q-Value (Joules)')
plt.title('Q-Value Convergence')
plt.legend()
plt.show()

plt.plot(Episode, Actions)
plt.xlabel('Round')
plt.ylabel('Discrete Action')
# plt.title('Selected Action for each round')
plt.show()

plt.plot(Episode, Average_Delay)
plt.xlabel('Round')
plt.ylabel('Average_Delay (s)')
# plt.title('Delay for each round')
plt.show()

plt.plot(Episode, E_consumed)
plt.xlabel('Round')
plt.ylabel('Energy Consumption (Joules)')
# plt.title('Energy Consumption for each round')
plt.show()

plt.plot(Episode, EE_consumed)
plt.xlabel('Round')
plt.ylabel('Total Energy Consumption (Joules)')
# plt.title('Total Energy Consumption for each round')
plt.show()