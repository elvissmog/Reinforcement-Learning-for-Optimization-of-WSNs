import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random


sink_node = 5
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


# initialization of network parameters
learning_rate = 0.5
initial_energy = 5  # Joules
data_packet_size = 512  # bits
electronic_energy = 50e-9  # Joules/bit 50e-9
amplifier_energy = 10e-12  # Joules/bit/square meter 100e-12
transmission_range = 30  # meters
node_energy_limit = 2
epsilon = 0.0


# initialize starting point

num_of_episodes = 10000000
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
            print('Evals:', e_values)
            cost = cost + 1
            print("Energy cannot be negative!")
            print("The final round is", rdn)

            xy.pop(index)

            dead_node = index

    for ind in list_unweighted_edges:
        if ind[0] != dead_node and ind[1] != dead_node:
            update_edges.append(ind)

    update_evals = {index: item for index, item in e_values.items() if item > node_energy_limit}

    # print('Original edges:', list_unweighted_edges)
    if cost == 1:
        #print('cost:', cost)

        try:
            graph, node_neighbors, q_values, e_values, path_q_values = build_graph(xy, update_edges)

            e_values = update_evals


        except ValueError:
            break


        cost = 0
    # graph, distances = ngraph, ndistances

    list_unweighted_edges = update_edges




'''print('Round:', round)
print('Delay:', delay)
print('Total Energy:', EE_consumed)
print('Energy:', E_consumed)
print('Average_QVals:', mean_Q)
print('Length Round:', len(round))
print('Length Delay:', len(delay))
print('Length Q_Val:', len(mean_Q))
print('Length Energy:', len(EE_consumed))

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

plt.plot(round, EE_consumed)
plt.xlabel('Round')
plt.ylabel('Total Energy Consumption (Joules)')
plt.title('Total Energy Consumption for each round')
plt.show()'''




