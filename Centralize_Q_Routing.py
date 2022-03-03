import numpy as np
import networkx as nx
import random
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import time

from AllMst import Yamada

G = nx.Graph()

# Adding nodes to the graph and their corresponding coordinates

xy = [(1, 2), (7, 2), (4, 0), (4, 6)]
for i in range(len(xy)):
	G.add_node(i, pos=xy[i])

list_unweighted_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]




position_array = []
for node in sorted(G):
    position_array.append(G.nodes[node]['pos'])
distances = squareform(pdist(np.array(position_array)))
for u, v in list_unweighted_edges:
    #G.add_edge(u, v, weight = 1)
    G.add_edge(u, v, weight=np.round(distances[u][v], decimals=1))


# Initialize source node to send packet
start = 0

# Initialize the sink node to receive packet
sink_node = 3

# initialization of network parameters
discount_factor = 0
learning_rate = 0.5
initial_energy = 0.5                  # Joules
packet_size = 256                     # bits
electronic_energy = 50e-9            # Joules/bit 5
amplifier_energy = 100e-12           # Joules/bit/square meter
transmission_range = 30               # meters
pathloss_exponent = 2                 # constant

d =[[0 for i in range(len(G))] for j in range(len(G))]
R =[[0 for i in range(len(G))] for j in range(len(G))]
Etx=[[0 for i in range(len(G))] for j in range(len(G))]
Erx=[[0 for i in range(len(G))] for j in range(len(G))]
path_Q_values  =[[0 for i in range(len(G))] for j in range(len(G))]
E_vals = [initial_energy for i in range(len(G))]
epsilon = 0.0
episodes = 200000

for i in range(len(G)):
	for j in range(len(G)):
		if i !=j:
			d[i][j] = math.sqrt(math.pow((position_array[i][0] - position_array[j][0]), 2) + math.pow((position_array[i][1] - position_array[j][1]), 2))
			Etx[i][j] = electronic_energy * packet_size + amplifier_energy * packet_size * math.pow((d[i][j]), pathloss_exponent)
			Erx[i][j] =  electronic_energy * packet_size


start_time = time.time()
Y = Yamada(G)
all_MSTs = Y.spanning_trees()
state_space = []
for T in all_MSTs:
    for path in nx.all_simple_paths(T, source=start, target=sink_node):
        state_space.append(path)
print(state_space)

Q_matrix = np.zeros((len(state_space), len(state_space)))
#Q_matrix =[[0 for i in range(len(state_space))] for j in range(len(state_space))]

initial_state = random.choice(range(0, len(state_space), 1))
#print('initial state:', initial_state)

Q_value = []
Min_value = []
Actions = []
Episode = []
delay = []
E_consumed = []
EE_consumed = []

for i in range(episodes):

    initial_delay = 0
    tx_energy = 0
    rx_energy = 0
    Episode.append(i)
    available_actions = [*range(0, len(state_space), 1)]


    current_state = initial_state

    if random.random() >= 1 - epsilon:
        # Get random action
        action = random.choice(available_actions)
    else:
        # Get action from Q table
        action = np.argmin(Q_matrix[current_state, :])

    Actions.append(action)

    initial_state = action
    #print('action is:', action)

    chosen_MST = state_space[action]

    #transmit with chosen MST and compute the residual energy of each node

    counter = 0
    while counter < len(chosen_MST)-1:
        init_node = chosen_MST[counter]
        next_node = chosen_MST[counter + 1]
        E_vals[init_node] = E_vals[init_node] - Etx[init_node][next_node]  # update the start node energy
        E_vals[next_node] = E_vals[next_node] - Erx[init_node][next_node]  # update the next hop energy
        tx_energy += Etx[init_node][next_node]
        rx_energy += Erx[init_node][next_node]
        initial_delay += d[init_node][next_node]
        counter += 1
    #print("The Energy is, ", E_vals)

    #reward = tx_energy

    reward = tx_energy + rx_energy  # Normalization of the reward and a coresponding normalization of the Q-Value
    Min_value.append(reward)

    # Maximum possible Q value in next step (for new state)
    max_future_q = np.min(Q_matrix[action, :])

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

'''print('Reward:', Min_value)
print('Round:', Episode)
print('Delay:', delay)
print('Total Energy:', EE_consumed)
print('Energy:', E_consumed)
print('QVals:', Q_value)'''



plt.plot(Episode, Q_value, label = "Q-Value")
plt.plot(Episode, Min_value, label = "Reward")
plt.xlabel('Round')
plt.ylabel('Q-Value')
plt.show()

plt.plot(Episode, Actions)
plt.xlabel('Round')
plt.ylabel('Discrete Action')
plt.show()

plt.plot(Episode, delay)
plt.xlabel('Round')
plt.ylabel('Delay (s)')
plt.show()

plt.plot(Episode, E_consumed)
plt.xlabel('Round')
plt.ylabel('Energy Consumption (Joules)')
plt.show()

plt.plot(Episode, EE_consumed)
plt.xlabel('Round')
plt.ylabel('Total Energy Consumption (Joules)')
plt.show()







