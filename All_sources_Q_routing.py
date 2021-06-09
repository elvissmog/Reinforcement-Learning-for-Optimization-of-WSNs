import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.spatial.distance import pdist, squareform

# Instantiating the G object

G = nx.Graph()
transmission_range = 4

# Adding nodes to the graph and their corresponding coordinates

xy = [(1, 3), (2.5, 5), (2.5, 1), (4.5, 5), (4.5, 1), (6, 3)]
for i in range(len(xy)):
	G.add_node(i, pos=xy[i])

# Adding unweighted edges to the Graph
list_unweighted_edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]

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
initial_energy = 1  # Joules
data_packet_size = 512  # bits
electronic_energy = 50e-9  # Joules/bit 50e-9
e_fs = 10e-12  # Joules/bit/(meter)**2
e_mp = 0.0013e-12 #Joules/bit/(meter)**4
node_energy_limit = 0.001

R = [[[0 for i in range(len(G))] for j in range(len(G))] for n in range(len(G))]
d = [[0 for i in range(len(G))] for j in range(len(G))]
Etx = [[0 for i in range(len(G))] for j in range(len(G))]
Erx = [[0 for i in range(len(G))] for j in range(len(G))]
path_Q_values = [[[0 for i in range(len(G))] for j in range(len(G))] for n in range(len(G))]
Q_vals = [[0 for i in range(len(G))] for j in range(len(G))]
E_vals = [initial_energy for i in range(len(G))]


epsilon = 0.0

sink_node = 5
E_vals[sink_node] = 50000

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

num_of_episodes = 200000
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
    #Av_mean_Q.append(sum(mean_Q))
    #Av_delay.append(sum(delay))
    #Av_E_consumed.append(sum(E_consumed))
    #round.append(i)

'''
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

'''

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

