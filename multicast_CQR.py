import numpy as np
import networkx as nx
import random
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import time

from AllMst import Yamada

G = nx.Graph()
xy = [(1, 3), (2.5, 5), (2.5, 1), (4.5, 5), (4.5, 1), (6, 3)]
for i in range(len(xy)):
	G.add_node(i, pos=xy[i])

# Adding unweighted edges to the Graph
list_unweighted_edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]

position_array = []
for node in sorted(G):
    position_array.append(G.nodes[node]['pos'])
distances = squareform(pdist(np.array(position_array)))
for u, v in list_unweighted_edges:
    #G.add_edge(u, v, weight = 1)
    G.add_edge(u, v, weight=np.round(distances[u][v], decimals=1))

# initialization of network parameters
discount_factor = 0
learning_rate = 0.7
initial_energy = 0.5                  # Joules
packet_size = 512                     # bits
electronic_energy = 50e-9            # Joules/bit 5
amplifier_energy = 100e-12           # Joules/bit/square meter
transmission_range = 30               # meters
pathloss_exponent = 2                 # constant

d =[[0 for i in range(len(G))] for j in range(len(G))]
Etx=[[0 for i in range(len(G))] for j in range(len(G))]
Erx=[[0 for i in range(len(G))] for j in range(len(G))]
path_Q_values  =[[0 for i in range(len(G))] for j in range(len(G))]
E_vals = [initial_energy for i in range(len(G))]
epsilon = 0.1
episodes = 200000

sink_node = 5

for i in range(len(G)):
	for j in range(len(G)):
		if i !=j:
			d[i][j] = math.sqrt(math.pow((position_array[i][0] - position_array[j][0]), 2) + math.pow((position_array[i][1] - position_array[j][1]), 2))
			Etx[i][j] = electronic_energy * packet_size + amplifier_energy * packet_size * math.pow((d[i][j]), pathloss_exponent)
			Erx[i][j] =  electronic_energy * packet_size


Y = Yamada(G)
all_MSTs = Y.spanning_trees()

#the set of neighbors of all nodes in each MST
node_neigh=[]
for T in all_MSTs:
	node_neighT = {}
	for n in T.nodes:
		node_neighT[n] = list(T.neighbors(n))
	node_neigh.append(node_neighT)
#print(node_neigh)

# Ranking nodes in terms of hop count to sink for each MST
MSTs_hop_count = []
MST_paths = []
for T in all_MSTs:
	hop_counts = {}
	MST_path = {}
	for n in T.nodes:
		for path in nx.all_simple_paths(T, source=n, target=sink_node):
			hop_counts[n] = len(path) - 1
			MST_path[n]=path
	hop_counts[sink_node] = 0                  # hop count of sink
	MSTs_hop_count.append(hop_counts)
	MST_paths.append(MST_path)
	
#print('All paths', MST_paths)



Q_matrix = np.zeros((len(all_MSTs), len(all_MSTs)))
initial_state = random.choice(range(0, len(all_MSTs), 1))


Q_value = []
Min_value = []
Actions = []
Episode = []
delay = []
E_consumed = []
EE_consumed = []


start_time = time.time()

for i in range(episodes):

    initial_delay = 0
    tx_energy = 0
    rx_energy = 0
    Episode.append(i)
    available_actions = [*range(0, len(all_MSTs), 1)]


    current_state = initial_state

    if np.random.random() >= 1 - epsilon:
            # Get action from Q table
        action = random.choice(available_actions)
    else:
            # Get random action
        action = np.argmax(Q_matrix[current_state, :])

    Actions.append(action)

    initial_state = action
    #print('action is:', action)

   
    chosen_MST = MST_paths[action]
    #print(chosen_MST)

    for node in chosen_MST:
        counter = 0
        while counter < len(chosen_MST[node])-1:
            init_node = chosen_MST[node][counter]
            next_node = chosen_MST[node][counter + 1]
            E_vals[init_node] = E_vals[init_node] - Etx[init_node][next_node]  # update the start node energy
            # E_vals[next_node] = E_vals[next_node] - Erx[init_node][next_node]  # update the next hop energy
            tx_energy += Etx[init_node][next_node]
            rx_energy += Erx[init_node][next_node]
            counter += 1
            #print("counter", counter)

    		

    
    reward = min(E_vals)
    Min_value.append(reward)
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

    delay.append(initial_delay)
    E_consumed.append(tx_energy + rx_energy)
    EE_consumed.append(sum(E_consumed))

    cost = True
    for item in E_vals:
        if item <= 0:
            cost = False
            print("Energy cannot be negative!")
            print("The final round is", i)

    if not cost:
        break





print("--- %s seconds ---" % (time.time() - start_time))

'''
print('Reward:', Min_value)
print('Round:', Episode)
#print('Delay:', delay)
print('Total Energy:', EE_consumed)
print('Energy:', E_consumed)
print('QVals:', Q_value)
'''

plt.plot(Episode, Q_value, label = "Q-Value")
plt.plot(Episode, Min_value, label = "Reward")
plt.xlabel('Round')
plt.ylabel('Q-Value')
#plt.title('Q-Value Convergence')
plt.legend()
plt.show()

plt.plot(Episode, Actions)
plt.xlabel('Round')
plt.ylabel('Discrete Action')
#plt.title('Selected Action for each round')
plt.show()

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
