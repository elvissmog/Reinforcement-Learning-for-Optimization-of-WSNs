import numpy as np
import networkx as nx
import random
import math
from AllMst import Yamada
#import matplotlib.pyplot as plt
from collections import Counter
import json
import time

# initialization of network parameters
discount_factor = 0
learning_rate = 0.9
initial_energy = 100        # Joules
data_packet_size = 1024      # bits
electronic_energy = 50e-9   # Joules/bit 5
e_fs = 10e-12               # Joules/bit/(meter)**2
e_mp = 0.0013e-12           #Joules/bit/(meter)**4
node_energy_limit = 0
num_pac = 1
txr = 150
epsilon = 0.1
episodes = 5000000
sink_energy = 5000000
sink_node = 100


start_time = time.time()

G = nx.Graph()

with open('pos.txt', 'r') as filehandle:
    pos_list = json.load(filehandle)

xy = []
for ps in pos_list:
    xy.append(tuple(ps))

with open('edges.txt', 'r') as filehandle:
    edges_list = json.load(filehandle)

list_unweighted_edges = []
for ed in edges_list:
    list_unweighted_edges.append(tuple(ed))

for i in range(len(xy)):
	G.add_node(i, pos=xy[i])

p = []
for node in sorted(G):
    p.append(G.nodes[node]['pos'])

Trx_dis = []
for u, v in list_unweighted_edges:
    distance = math.sqrt(math.pow((p[u][0] - p[v][0]), 2) + math.pow((p[u][1] - p[v][1]), 2))

    if distance <= txr:
        Trx_dis.append(distance)
        G.add_edge(u, v, weight = math.ceil(distance))
        #G.add_edge(u, v, weight=1)

com_range = max(Trx_dis)

print('cm_range:', com_range)

initial_E_vals = [initial_energy for i in G.nodes]
ref_E_vals = [initial_energy for i in G.nodes]

initial_E_vals[sink_node] = sink_energy
ref_E_vals[sink_node] = sink_energy

total_initial_energy = sum(initial_E_vals)

d_o = math.sqrt(e_fs/e_mp)

Y = Yamada(graph=G, n_trees = np.inf)  #np.inf
#Y = Yamada(graph=G, n_trees = 1000)  #np.inf
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


#print('ST_paths:', ST_paths)

Q_matrix = np.zeros((len(ST_paths), len(ST_paths)))
initial_state = random.choice(range(0, len(ST_paths), 1))

NQ_value = []
Sum_Q_value = []
Q_value = []
Action = []
Min_value = []
Episode = []
E_consumed = []
EE_consumed = []


for episode in range(episodes):

    txe = 0
    rxe = 0

    available_actions = [*range(0, len(ST_paths), 1)]

    current_state = initial_state

    if random.random() >= 1 - epsilon:
        # Get random action
        action = random.choice(available_actions)
    else:
        # Get action from Q table
        action = np.argmin(Q_matrix[current_state, :])


    initial_state = action

    chosen_MST = ST_paths[action]
    #print('Chosen MST:', chosen_MST)
    Action.append(action + 1)

    for node in chosen_MST:
        counter = 0
        while counter < len(chosen_MST[node]) - 1:
            ini = chosen_MST[node][counter]
            nex = chosen_MST[node][counter + 1]
            dis = math.ceil(math.sqrt(math.pow((p[ini][0] - p[nex][0]), 2) + math.pow((p[ini][1] - p[nex][1]), 2)))
            #print('dis:', dis)
            if dis <= d_o:
                etx = electronic_energy * data_packet_size + e_fs * data_packet_size * math.pow(dis, 2)
            else:
                etx = electronic_energy * data_packet_size + e_mp * data_packet_size * math.pow(dis, 4)
            erx = electronic_energy * data_packet_size

            initial_E_vals[ini] = initial_E_vals[ini] - num_pac * etx  # update the start node energy

            initial_E_vals[nex] = initial_E_vals[nex] - num_pac * erx  # update the next hop energy

            #txr += num_pac * etx
            #rxe += num_pac * erx

            counter += 1



    rew = [ref_E_vals[i] - initial_E_vals[i] for i in G.nodes if i != sink_node]    # Energy comsumption of each node for data transmission


    reward = max(rew)

    #reward = sum(rew)


    Min_value.append(reward)

    #E_consumed.append(txr + rxe)


    # Maximum possible Q value in next step (for new state)
    max_future_q = np.max(Q_matrix[action, :])

    # Current Q value (for current state and performed action)
    current_q = Q_matrix[current_state, action]

    # And here's our equation for a new Q value for current state and action

    new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)

    Q_value.append(new_q)
    Episode.append(episode)
    #Sum_Q_value.append(sum(Q_value))

    # Update Q table with new Q value
    Q_matrix[current_state, action] = new_q

    cost = True
    for index, item in enumerate(initial_E_vals):
        if item <= node_energy_limit:
            print('Index:', index)
            cost = False
            print("The final round is", episode)
            #print('Average Energy Consumption:', sum(E_consumed)/i)
            #print('Average remaining Energy:', sum([initial_E_vals[i] for i in G.nodes if i != sink_node])/(len(G.nodes)-1))

    if not cost:
        break

    for i in range(len(ref_E_vals)):
        ref_E_vals[i] = initial_E_vals[i]

    #print('Energy:', ref_E_vals)


print("--- %s seconds ---" % (time.time() - start_time))

print('EC:', (total_initial_energy - sum(initial_E_vals))/len(Episode))

for qv in Q_value:
    NQ_value.append(qv/max(Q_value))

my_data = Counter(Action)
print('RT_UT:', my_data.most_common())  # Returns all unique items and their counts

with open('lqals.txt', 'w') as f:
    f.write(json.dumps(NQ_value))

# Now read the file back into a Python list object
with open('lqals.txt', 'r') as f:
    NQ_value = json.loads(f.read())

#print('EC:', sum(E_consumed))


'''
plt.plot(Episode, NQ_value)
plt.xlabel('Round')
plt.ylabel('Average Q-Value')
plt.title('Q-Value Convergence ')
plt.show()
'''

