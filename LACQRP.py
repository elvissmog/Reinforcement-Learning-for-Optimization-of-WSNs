import numpy as np
import networkx as nx
import random
import math
from AllMst import Yamada
from collections import Counter
import json
import time

start_time = time.time()

G = nx.Graph()
txr = 1

with open('pos1.txt', 'r') as filehandle:
    pos_list = json.load(filehandle)

xy = []
for ps in pos_list:
    xy.append(tuple(ps))

with open('edges1.txt', 'r') as filehandle:
    edges_list = json.load(filehandle)

list_unweighted_edges = []
for ed in edges_list:
    list_unweighted_edges.append(tuple(ed))

for i in range(len(xy)):
	G.add_node(i, pos=xy[i])

position_array = []
for node in sorted(G):
    position_array.append(G.nodes[node]['pos'])
for u, v in list_unweighted_edges:
    distance = math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow((position_array[u][1] - position_array[v][1]), 2))
    G.add_edge(u, v, weight = math.ceil(distance/txr))


# initialization of network parameters
discount_factor = 0
learning_rate = 0.7
initial_energy = 1000        # Joules
data_packet_size = 1024      # bits
electronic_energy = 50e-9   # Joules/bit 5
e_fs = 10e-12               # Joules/bit/(meter)**2
e_mp = 0.0013e-12           #Joules/bit/(meter)**4
node_energy_limit = 10

d = [[0 for i in range(len(G))] for j in range(len(G))]
Etx = [[0 for i in range(len(G))] for j in range(len(G))]
Erx = [[0 for i in range(len(G))] for j in range(len(G))]
initial_E_vals = [initial_energy for i in range(len(G))]
ref_E_vals = [initial_energy for i in range(len(G))]
epsilon = 0.0
episodes = 5000000
sink_energy = 5000000
#sink_node = 100

sink_node = 1000


initial_E_vals[sink_node] = sink_energy
ref_E_vals[sink_node] = sink_energy

d_o = math.sqrt(e_fs/e_mp)/txr

for i in range(len(G)):
    for j in range(len(G)):
        if i != j:
            d[i][j] = (math.sqrt(math.pow((position_array[i][0] - position_array[j][0]), 2) + math.pow(
                (position_array[i][1] - position_array[j][1]), 2)))/txr

            if d[i][j] <= d_o:
                Etx[i][j] = electronic_energy * data_packet_size + e_fs * data_packet_size * math.pow((d[i][j]), 2)
            else:
                Etx[i][j] = electronic_energy * data_packet_size + e_mp * data_packet_size * math.pow((d[i][j]), 4)
            Erx[i][j] = electronic_energy * data_packet_size




Y = Yamada(graph=G, n_trees = np.inf)  #np.inf
all_STs = Y.spanning_trees()
print('no of MSTs:', len(all_STs))

# Ranking nodes in terms of hop count to sink for each MST

ST_paths = []
for T in all_STs:
    ST_path = {}
    for n in T.nodes:
        for path in nx.all_simple_paths(T, source=n, target=sink_node):
            ST_path[n] = path


    ST_paths.append(ST_path)


Q_matrix = np.zeros((len(ST_paths), len(ST_paths)))
initial_state = random.choice(range(0, len(ST_paths), 1))

NQ_value = []
Q_value = []
Action = []
Min_value = []
Episode = []
delay = []
E_consumed = []
EE_consumed = []
Min_nodes_RE = []


for i in range(episodes):

    tx_energy = 0
    rx_energy = 0

    Episode.append(i)

    available_actions = [*range(0, len(ST_paths), 1)]

    current_state = initial_state

    if random.random() >= 1 - epsilon:
        # Get random action
        action = random.choice(available_actions)
    else:
        # Get action from Q table
        action = np.argmin(Q_matrix[current_state, :])


    initial_state = action

    chosen_ST = ST_paths[action]
    Action.append(action + 1)

    for node in chosen_ST:
        counter = 0
        while counter < len(chosen_ST[node]) - 1:
            init_node = chosen_ST[node][counter]
            next_node = chosen_ST[node][counter + 1]
            initial_E_vals[init_node] = initial_E_vals[init_node] - Etx[init_node][next_node]  # update the start node energy
            initial_E_vals[next_node] = initial_E_vals[next_node] - Erx[init_node][next_node]  # update the next hop energy
            tx_energy += Etx[init_node][next_node]
            rx_energy += Erx[init_node][next_node]
            counter += 1


    Energy_Consumption = [ref_E_vals[i] - initial_E_vals[i] for i in G.nodes if i!=sink_node]

    reward = max(Energy_Consumption)

    Min_value.append(reward)

    # Maximum possible Q value in next step (for new state)
    max_future_q = np.max(Q_matrix[action, :])

    # Current Q value (for current state and performed action)
    current_q = Q_matrix[current_state, action]

    # And here's our equation for a new Q value for current state and action

    new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)

    Q_value.append(new_q)

    # Update Q table with new Q value
    Q_matrix[current_state, action] = new_q

    E_consumed.append(tx_energy + rx_energy)

    cost = True
    for index, item in enumerate(initial_E_vals):
        if item <= node_energy_limit:
            print('Index:', index)
            cost = False
            print("The final round is", i)
            #print('Average Energy Consumption:', sum(E_consumed)/i)
            #print('Average remaining Energy:', sum([initial_E_vals[i] for i in G.nodes if i != sink_node])/(len(G.nodes)-1))

    if not cost:
        break

    for i in range(len(ref_E_vals)):
        ref_E_vals[i] = initial_E_vals[i]


print("--- %s seconds ---" % (time.time() - start_time))

for qv in Q_value:
    NQ_value.append(qv/max(Q_value))

my_data = Counter(Action)
print('RT_UT:', my_data.most_common())  # Returns all unique items and their counts

with open('lqals.txt', 'w') as f:
    f.write(json.dumps(NQ_value))

# Now read the file back into a Python list object
with open('lqals.txt', 'r') as f:
    NQ_value = json.loads(f.read())


