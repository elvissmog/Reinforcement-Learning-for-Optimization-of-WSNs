import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.spatial.distance import pdist, squareform

# Instantiating the Graph object

G = nx.Graph()

# Adding nodes to the graph and their corresponding coordinates

x_y_cordinates = [(1, 3), (2.5, 5), (2.5, 1), (4.5, 5), (4.5, 1), (6, 3)]
for i in range(len(x_y_cordinates)):
	G.add_node(i, pos=x_y_cordinates[i])

# Adding unweighted edges to the Graph
list_unweighted_edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]

# Extracting the (x,y) coordinate of each node to enable the calculation of the Euclidean distances of the Graph edges
position_array = []
for node in sorted(G):
    position_array.append(G.nodes[node]['pos'])

# Calculating the distance matrix for the graph

distances = squareform(pdist(np.array(position_array)))

for u, v in list_unweighted_edges:
    G.add_edge(u, v, weight=np.round(distances[u][v], decimals=1))

node_pos = nx.get_node_attributes(G, 'pos')
edge_weight = nx.get_edge_attributes(G, 'weight')

node_col = ['yellow']
edge_col = ['black']

# The dictionary of neighbors of all nodes in the graph
node_neigh = {}
for n in G.nodes:
    node_neigh[n] = list(G.neighbors(n))

print('node_neigh_keys:', list(node_neigh.keys()))
#node_neigh.pop(3)
#print('Updated node_neigh:', node_neigh)
# initialization of network parameters
learning_rate = 0.5
initial_energy = 0.005  # Joules
packet_size = 512  # bits
electronic_energy = 50e-9  # Joules/bit 50e-9
amplifier_energy = 100e-12  # Joules/bit/square meter 100e-12
transmission_range = 30  # meters
pathloss_exponent = 2  # constant
R = [[0 for i in range(len(G))] for j in range(len(G))]
Etx = [[0 for i in range(len(G))] for j in range(len(G))]
Erx = [[0 for i in range(len(G))] for j in range(len(G))]
path_Q_values = [[0 for i in range(len(G))] for j in range(len(G))]
Q_vals = [0 for i in range(len(G))]
E_vals = [initial_energy for i in range(len(G))]

epsilon = 0.0

for i in range(len(G)):
    for j in range(len(G)):
        if i != j:
            R[i][j] = math.sqrt(math.pow((position_array[i][0] - position_array[j][0]), 2) + math.pow(
                (position_array[i][1] - position_array[j][1]), 2))
            Etx[i][j] = electronic_energy * packet_size + amplifier_energy * packet_size * math.pow((R[i][j]), 2)
            Erx[i][j] = electronic_energy * packet_size

# initialize starting point

num_of_episodes = 1000
start = 0
queue = [start]
mean_Q = []
E_consumed = []
EE_consumed = []
delay = []
round = []
deleted_neigh = []
for i in range(num_of_episodes):

    start = 0
    # start = random.choice(range(0,len(G.nodes)-1,1))
    queue = [start]  # first visited node
    path = str(start)  # first node
    end = 5
    temp_qval = dict()
    initial_delay = 0
    tx_energy = 0
    rx_energy = 0


    #cost = True
    for item in E_vals:
        if item <= 0:
            #cost = False
            print("Energy cannot be negative!")
            print("The E_Vals in episode number {} is {} ".format(i+1, E_vals))

    '''if not cost:
        continue'''
        #break

    print('node_neigh:', node_neigh)
    while True:
        for neigh in node_neigh[start]:
            #print('node_neigh(start1):', node_neigh[start])
            if E_vals[neigh] <= 0:
                #print("The neigh and the others are {} and {} ".format(neigh, node_neigh[start]))
                deleted_neigh.append(neigh)
                try:
                    del node_neigh[neigh]
                except KeyError:
                    pass

                '''for item in node_neigh.keys():
                    for index in range(len(node_neigh[item])):
                        if  neigh == node_neigh[item][index]:
                            node_neigh[item].remove(node_neigh[item][index])
                            print('node_neigh[item][index]:', node_neigh[item][index])
                            #node_neigh[item].remove(node_neigh[item][index])'''
                #node_neigh[start].remove(neigh)
            #print('node_neigh(start2):', node_neigh[start])
            temp_qval[neigh] = (1 - learning_rate) * path_Q_values[start][neigh] + learning_rate * (R[start][neigh] + Q_vals[neigh])

        copy_q_values = {key: value for key, value in temp_qval.items() if key not in queue}
        # next_hop = min(temp_qval.keys(), key=(lambda k: temp_qval[k]))  #determine the next hop based on the minimum qvalue but ignore the visited node qvalue

        if np.random.random() >= 1 - epsilon:
            # Get action from Q table
            next_hop = random.choice(list(copy_q_values.keys()))
        else:
            # Get random action
            next_hop = min(copy_q_values.keys(), key=(lambda k: copy_q_values[k]))

        initial_delay += R[start][next_hop]

        # copy_q_values = {key: value for key,value in temp_qval.items() if key not in queue}

        # print("The updated qvalue is", copy_q_values)
        # print("The next hop is", next_hop)
        queue.append(next_hop)

        path_Q_values[start][next_hop] = temp_qval[next_hop]  # update the path qvalue of the next hop
        Q_vals[start] = temp_qval[next_hop]  # update the qvalue of the start node

        # sum_Q[i] = sum(Q_vals)/len(Q_vals)
        mean_Qvals = sum(Q_vals) / (len(Q_vals) * max(Q_vals))
        E_vals[start] = E_vals[start] - Etx[start][next_hop]  # update the start node energy
        E_vals[next_hop] = E_vals[next_hop] - Erx[start][next_hop]  # update the next hop energy
        tx_energy += Etx[start][next_hop]
        rx_energy += Erx[start][next_hop]


        path = path + "->" + str(next_hop)  # update the path after each visit
        print('path:', path)
        # print("The visited nodes are", queue)

        start = next_hop

        if next_hop == end:
            break

    print('Deleted_neigh:', deleted_neigh)
    print('updated_node_neigh:', node_neigh)
    delay.append(initial_delay)
    E_consumed.append(tx_energy + rx_energy)
    EE_consumed.append(sum(E_consumed))
    mean_Q.append(mean_Qvals)
    round.append(i)



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




