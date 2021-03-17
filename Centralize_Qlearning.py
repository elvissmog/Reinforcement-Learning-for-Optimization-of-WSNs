import numpy as np
import networkx as nx
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

from AllMst import Yamada

G = nx.Graph()

# Adding nodes to the graph and their corresponding coordinates
G.add_node(0, pos=(1, 3))
G.add_node(1, pos=(2.5, 5))
G.add_node(2, pos=(2.5, 1))
G.add_node(3, pos=(4.5, 5))
G.add_node(4, pos=(4.5, 1))
G.add_node(5, pos=(6, 3))

# Adding unweighted edges to the Graph
list_unweighted_edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5),
                             (4, 5)]

position_array = []
for node in sorted(G):
    position_array.append(G.nodes[node]['pos'])
distances = squareform(pdist(np.array(position_array)))
for u, v in list_unweighted_edges:
    G.add_edge(u, v, weight=np.round(distances[u][v], decimals=1))



Y = Yamada(G)
all_MSTs = Y.spanning_trees()

start = 0



#Fill in a few edges
#sink_nodes = [node for node, outdegree in T.out_degree(T.nodes()).items() if outdegree == 0]
sink_node = 5



#Initial Q values

# initialization of network parameters
learning_rate = 0.5
initial_energy = 0.5                  # Joules
packet_size = 512                     # bits
electronic_energy = 50e-9             # Joules/bit 50e-9
amplifier_energy = 100e-12          # Joules/bit/square meter 100e-12
transmission_range = 30               # meters
pathloss_exponent = 2                 # constant

d =[[0 for i in range(len(G))] for j in range(len(G))]
R =[[0 for i in range(len(G))] for j in range(len(G))]
Etx=[[0 for i in range(len(G))] for j in range(len(G))]
Erx=[[0 for i in range(len(G))] for j in range(len(G))]
path_Q_values  =[[0 for i in range(len(G))] for j in range(len(G))]
E_vals = [5 for i in range(len(G))]
epsilon = 0.1
episodes = 10

for i in range(len(G)):
	for j in range(len(G)):
		if i !=j:
			d[i][j] = math.sqrt(math.pow((position_array[i][0] - position_array[j][0]), 2) + math.pow((position_array[i][1] - position_array[j][1]), 2))
			Etx[i][j] = electronic_energy * packet_size + amplifier_energy * packet_size * math.pow((d[i][j]), pathloss_exponent)
			Erx[i][j] =  electronic_energy * packet_size


state_space = []
for g in all_MSTs:
    for path in nx.all_simple_paths(g, source=start, target=sink_node):
        state_space.append(path)
print(state_space)

Q_matrix = np.zeros((len(state_space), len(state_space)))
#Q_matrix =[[0 for i in range(len(state_space))] for j in range(len(state_space))]

initial_state = random.choice(range(0, len(state_space), 1))
print('initial state:', initial_state)

Q_value = []
Min_value = []
Actions = []
Episode = []

for i in range(episodes):
    Episode.append(i)
    available_actions = [*range(0, len(state_space), 1)]


    current_state = initial_state

    if np.random.random() >= 1 - epsilon:
            # Get action from Q table
        action = random.choice(available_actions)
    else:
            # Get random action
        action = np.argmax(Q_matrix[current_state, :])

    Actions.append(action)

    initial_state = action
    print('action is:', action)

    chosen_MST = state_space[action]

    #transmit with chosen MST and compute the residual energy of each node

    counter = 0
    while counter < len(chosen_MST)-1:
        init_node = chosen_MST[counter]
        next_node = chosen_MST[counter + 1]

        E_vals[init_node] = E_vals[init_node] - Etx[init_node][next_node]  # update the start node energy
        E_vals[next_node] = E_vals[next_node] - Erx[init_node][next_node]  # update the next hop energy
        counter+=1
    print("The Energy is, ",E_vals)

    reward = min(E_vals)
    Min_value.append(reward)

    # Maximum possible Q value in next step (for new state)
    max_future_q = np.max(Q_matrix[action, :])

    # Current Q value (for current state and performed action)
    current_q = Q_matrix[current_state, action]

    # And here's our equation for a new Q value for current state and action
    new_q = (1 - learning_rate) * current_q + learning_rate * (reward +  max_future_q)
    Q_value.append(new_q)


    # Update Q table with new Q value
    Q_matrix[current_state, action] = new_q

    cost = True
    for item in E_vals:
        if item <= 0:
            cost = False
            print("Energy cannot be negative!")
            print("The final round is", i)

    if not cost:
        break


#plt.plot(Episode, Min_value, label = "Min NELT")

print('reward:', Min_value)

plt.plot(Episode, Q_value, label = "Q-Value")
#plt.plot(Episode, M, label = "MOSPF")
plt.xlabel('Round')
plt.ylabel('Lifetime (s)')
#plt.title('Convergence Rate')
plt.legend()
plt.show()

plt.plot(Episode, Actions)

plt.xlabel('Round')
plt.ylabel('MST')
#plt.title('Convergence Rate')
plt.legend()
plt.show()





