import numpy as np
import networkx as nx
import random
import math
import matplotlib.pyplot as plt
from pqdict import PQDict
from collections import Counter
import json
import time

# initialization of network parameters


initial_energy = 10        # Joules
data_packet_size = 1024      # bits
electronic_energy = 50e-9   # Joules/bit 5
e_fs = 10e-12               # Joules/bit/(meter)**2
e_mp = 0.0013e-12           #Joules/bit/(meter)**4
node_energy_limit = 0
num_pac = 1
txr = 120
episodes = 5000000
sink_energy = 5000000
sink_node = 100

cr = 100   # crossover rate
mr = 100   # mutation rate
ng = 1   # number of generations

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

def is_tree_of_graph(child, parent):

    parent_edges = parent.edges()
    for child_edge in child.edges():
        if child_edge not in parent_edges:
            return False
    return nx.is_tree(child)


for i in range(len(xy)):
	G.add_node(i, pos=xy[i])

Trx_dis = []
p = []
for node in sorted(G):
    p.append(G.nodes[node]['pos'])
for u, v in list_unweighted_edges:
    distance = math.sqrt(math.pow((p[u][0] - p[v][0]), 2) + math.pow((p[u][1] - p[v][1]), 2))
    if distance <= txr:
        Trx_dis.append(distance)
        G.add_edge(u, v, weight = math.ceil(distance))

com_range = max(Trx_dis)

#print('cm_range:', com_range)

mst = nx.minimum_spanning_tree(G)
mst_edge = mst.edges()
# Calculating the minimum spanning tree cost of g
mst_edge_cost = []
for tu, tv in mst_edge:
    mst_edge_dis = math.sqrt(math.pow((p[tu][0] - p[tv][0]), 2) + math.pow((p[tu][1] - p[tv][1]), 2))
    mst_edge_cost.append(math.ceil(mst_edge_dis))

mst_cost = sum(mst_edge_cost)
#print('mst_cost:', mst_cost)

def prim(G, start):
    """Function recives a graph and a starting node, and returns a MST"""
    stopN = G.number_of_nodes() - 1
    current = start
    closedSet = set()
    pq = PQDict()
    mst = []

    while len(mst) < stopN:
        for node in G.neighbors(current):
            if node not in closedSet and current not in closedSet:
                if (current, node) not in pq and (node, current) not in pq:
                    w = G.edges[(current, node)]['weight']
                    pq.additem((current, node), w)

        closedSet.add(current)

        tup, wght = pq.popitem()
        while(tup[1] in closedSet):
            tup, wght = pq.popitem()
        mst.append(tup)
        current = tup[1]

    h = nx.Graph()

    for j in range(len(xy)):
        h.add_node(j, pos=xy[j])


    for (x, y) in mst:
        dt = math.sqrt(math.pow((p[x][0] - p[y][0]), 2) + math.pow((p[x][1] - p[y][1]), 2))
        h.add_edge(x, y, weight=math.ceil(dt))

    return h

MST = []

# Extracting the edges of unique MST from Prims algorithm and storing in MST_edges
MST_edges = []

for i in range(len(G.nodes)):
    y = prim(G, i)
    MST.append(y)
    #print(list(y.edges))
    if set(y.edges()) not in MST_edges:
        MST_edges.append(set(y.edges()))

# Converting the unique MST edges into graph and storing in unique_MSTs

unique_MSTs = []
for M_edges in MST_edges:
    t = nx.Graph()
    for pp in range(len(xy)):
        t.add_node(pp, pos=xy[pp])

    for (u, v) in M_edges:
        dis = math.sqrt(math.pow((p[u][0] - p[v][0]), 2) + math.pow(
            (p[u][1] - p[v][1]), 2))

        t.add_edge(u, v, weight=math.ceil(dis))
    unique_MSTs.append(t)





# Generating new MSTs from the unique_MSTs using genetic algorithm

for idx in range(ng):
    ind_pop = []                       # initializing an empty intermidiate population to store unique children
    nc = int(100/cr)

    # Crossover Operator
    for n_c in range(nc):
        two_ind = random.sample(unique_MSTs, 2)    # Randomly selecting two parents from the population(unique_MST) to perform crosss over operator

        gr_ed = []
        for gr in two_ind:
            gr_ed.append(list(gr.edges()))
        #print(gr_ed)
        sub_graph_edges = list(set(gr_ed[0] + gr_ed[1]))
        #print(sub_graph_edges)
        sub_graph = nx.Graph()
        for sp in range(len(xy)):
            sub_graph.add_node(sp, pos=xy[sp])

        for (su, sv) in sub_graph_edges:
            dis = math.sqrt(math.pow((p[su][0] - p[sv][0]), 2) + math.pow(
                (p[su][1] - p[sv][1]), 2))

            sub_graph.add_edge(su, sv, weight=math.ceil(dis))

        new_mst = nx.minimum_spanning_tree(sub_graph)
        #print(list(new_mst.edges))

        if set(new_mst.edges()) not in ind_pop:
            ind_pop.append(set(new_mst.edges()))

    # Mutation Operator
    nm = int(100 / mr)
    for n_m in range(nm):
        one_ind = random.sample(unique_MSTs, 1)   # Randomly selecting two parents from the population(unique_MST) to perform crosss over operator
        sgr_ed = []
        for sgr in one_ind:
            sgr_ed.append(list(sgr.edges()))
        sgr_ed = sgr_ed[0]
        #print('sgr ed:', sgr_ed)
        one_edge = random.sample(sgr_ed, 1)
        #print('edge to delete:', one_edge)
        sgr_ed.remove(one_edge[0])
        #print('sgr ed:', sgr_ed)
        su_graph_edges = [value for value in list_unweighted_edges if value not in one_edge]
        #print('sub_graph_edges:', su_graph_edges)
        su_graph = nx.Graph()
        for su in range(len(xy)):
            su_graph.add_node(su, pos=xy[su])


        for (us, vs) in su_graph_edges:
            dis = math.sqrt(math.pow((p[us][0] - p[vs][0]), 2) + math.pow(
                (p[us][1] - p[vs][1]), 2))

            su_graph.add_edge(us, vs, weight=math.ceil(dis))

        cut_set = [value for value in su_graph_edges if value not in sgr_ed]
        #print('cut_set:', cut_set)
        add_edge = random.sample(cut_set, 1)
        #print('add_edge:', add_edge)
        sgr_ed = sgr_ed + add_edge
        #print('sgr ed:', sgr_ed)
        mu_graph = nx.Graph()
        for mu in range(len(xy)):
            su_graph.add_node(mu, pos=xy[mu])

        for (ms, vu) in sgr_ed:
            dis = math.sqrt(math.pow((p[ms][0] - p[vu][0]), 2) + math.pow(
                (p[ms][1] - p[vu][1]), 2))

            mu_graph.add_edge(ms, vu, weight=math.ceil(dis))

        if is_tree_of_graph(mu_graph, G):
            if set(mu_graph.edges()) not in ind_pop:
                ind_pop.append(set(mu_graph.edges()))


    # Calculate the fitness of each individual in the intermidate population.
    # Converting each individual in the intermidate population to graph and store in ind_pop_graph
    ind_pop_graphs = []
    for ipg_edges in ind_pop:
        ipg = nx.Graph()
        for pg in range(len(xy)):
            ipg.add_node(pg, pos=xy[pg])

        for (pu, pv) in ipg_edges:
            dis = math.sqrt(math.pow((p[pu][0] - p[pv][0]), 2) + math.pow(
                (p[pu][1] - p[pv][1]), 2))

            ipg.add_edge(pu, pv, weight=math.ceil(dis))
        ind_pop_graphs.append(ipg)

    #print('ind_pop_graphs:', ind_pop_graphs)

    # Calculating the spanning tree cost of each individual in Population.

    for pop in ind_pop_graphs:
        pop_edges = set(pop.edges())  # Extracting the spanning tree graph edges
        #print(pop_edges)
        pop_Spanning_Tree_Edge_Distances = []
        for up, vp in pop_edges:
            dist_edge = math.sqrt(math.pow((p[up][0] - p[vp][0]), 2) + math.pow(
                (p[up][1] - p[vp][1]), 2))
            pop_Spanning_Tree_Edge_Distances.append(math.ceil(dist_edge))
        pop_Tree_Cost = sum(pop_Spanning_Tree_Edge_Distances)

        #print("pop spanning tree cost is:", pop_Tree_Cost)
        if pop_Tree_Cost <= mst_cost:
            if pop_edges not in MST_edges:
                MST_edges.append(pop_edges)

    unique_sol = []

    for p_edges in MST_edges:
        us = nx.Graph()
        for up in range(len(xy)):
            us.add_node(up, pos=xy[up])
        up_array = []
        for node in sorted(us):
            up_array.append(us.nodes[node]['pos'])
        # print(T_edges)
        for (su, sv) in p_edges:
            dis = math.sqrt(math.pow((p[su][0] - p[sv][0]), 2) + math.pow(
                (p[su][1] - p[sv][1]), 2))

            us.add_edge(su, sv, weight=math.ceil(dis))
        unique_sol.append(us)

    final_pop = []
    for uni_pop in unique_sol:
        if set(uni_pop.edges()) not in final_pop:
            final_pop.append(set(uni_pop.edges()))

    MST_edges = final_pop
    unique_MSTs = unique_sol


initial_E_vals = [initial_energy for i in range(len(G))]
ref_E_vals = [initial_energy for i in range(len(G))]

initial_E_vals[sink_node] = sink_energy
ref_E_vals[sink_node] = sink_energy

total_initial_energy = sum(initial_E_vals)

d_o = math.sqrt(e_fs/e_mp)


all_STs = unique_MSTs
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

samples = []

k = 1
#weights = np.random.uniform(-1.0, 1.0, size=k)
initial_weight = 0
stop_criterium = 10**-5
num_actions = len(ST_paths)
learning_rate = 0.7
discount_factor = 0
gamma = 0.99
epsilon = 1
sample_size = 5                       #len(ST_paths)

A = 0
b = 0
#A = np.zeros((k, k))
#b = np.zeros((k, 1))
#np.fill_diagonal(A, .1)
R = np.zeros((sample_size, 1))

phi_matrix = np.zeros((len(ST_paths), len(ST_paths)))

for s in range(len(ST_paths)):
    for a in range(len(ST_paths)):
        EC = []
        for node in ST_paths[a]:
            count = 0
            while count < len(ST_paths[a][node]) - 1:
                ini = ST_paths[a][node][count]
                nex = ST_paths[a][node][count + 1]
                dis = math.ceil(math.sqrt(math.pow((p[ini][0] - p[nex][0]), 2) + math.pow((p[ini][1] - p[nex][1]), 2)))
                # print('dis:', dis)
                if dis <= d_o:
                    etx = electronic_energy * data_packet_size + e_fs * data_packet_size * math.pow(dis, 2)
                else:
                    etx = electronic_energy * data_packet_size + e_mp * data_packet_size * math.pow(dis, 4)
                erx = electronic_energy * data_packet_size

                EC.append((etx + erx))
                count += 1
                phi_matrix[s, a] = sum(EC)

#print('phi_matrix:', phi_matrix)

for i in range(sample_size):

    tx_energy = 0
    rx_energy = 0

    sam = []
    
    available_actions = [*range(0, len(ST_paths), 1)]

    current_state = initial_state

    sam.append(initial_state)

    if random.random() >= 1 - epsilon:
        # Get random action
        action = random.choice(available_actions)
    else:
        # Get action from Q table
        action = np.argmin(Q_matrix[current_state, :])

    sam.append(action)

    initial_state = action

    chosen_MST = ST_paths[action]

    for node in chosen_MST:
        counter = 0
        while counter < len(chosen_MST[node]) - 1:
            ini = chosen_MST[node][counter]
            nex = chosen_MST[node][counter + 1]
            dis = math.ceil(math.sqrt(math.pow((p[ini][0] - p[nex][0]), 2) + math.pow((p[ini][1] - p[nex][1]), 2)))
            # print('dis:', dis)
            if dis <= d_o:
                etx = electronic_energy * data_packet_size + e_fs * data_packet_size * math.pow(dis, 2)
            else:
                etx = electronic_energy * data_packet_size + e_mp * data_packet_size * math.pow(dis, 4)
            erx = electronic_energy * data_packet_size

            initial_E_vals[ini] = initial_E_vals[ini] - num_pac * etx  # update the start node energy

            initial_E_vals[nex] = initial_E_vals[nex] - num_pac * erx  # update the next hop energy

            #tx_energy += num_pac * etx
            #rx_energy += num_pac * erx

            counter += 1


    Energy_Consumption = [ref_E_vals[i] - initial_E_vals[i] for i in G.nodes if i!=sink_node]

    reward = max(Energy_Consumption) + sum(Energy_Consumption)

    sam.append(reward)

    sam.append(action)

    # Maximum possible Q value in next step (for new state)
    max_future_q = np.max(Q_matrix[action, :])

    # Current Q value (for current state and performed action)
    current_q = Q_matrix[current_state, action]

    # And here's our equation for a new Q value for current state and action

    new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)

    # Update Q table with new Q value
    Q_matrix[current_state, action] = new_q

    cost = True
    for index, item in enumerate(initial_E_vals):
        if item <= node_energy_limit:
            print('Index:', index)
            cost = False


    samples.append(sam)

    if not cost:
        break

    for i in range(len(ref_E_vals)):
        ref_E_vals[i] = initial_E_vals[i]


print('samples:', samples)


'''
for sample in range(len(samples)):
    R[sample, 0] = samples[sample][2]

#print('reward:', R)

phi_matrix_sample = np.zeros((len(ST_paths), len(ST_paths)))

trans_phi_matrix_sample = np.zeros((len(ST_paths), len(ST_paths)))

#print(phi_matrix_sample)

for sample in range(len(samples)):
    for a in range(len(ST_paths)):
        #print(phi_matrix[sample[0], a])
        phi_matrix_sample[samples[sample][0], a] = phi_matrix[samples[sample][0], a]

for sample in range(len(samples)):
    act = np.argmin(phi_matrix[samples[sample][1], :])
    for a in range(len(ST_paths)):
        #print(phi_matrix[sample[0], a])
        #print('index:',  np.argmin(phi_matrix[samples[sample][1], :]))
        trans_phi_matrix_sample[samples[sample][1], a] = phi_matrix[samples[sample][1], act]

#print('trans_phi_matrix_sample:', trans_phi_matrix_sample)

for sample in range(len(samples)):
    print('index:', phi_matrix_sample[samples[sample][0], :])
'''

NQ_value = []
Q_value = []
CQ_value = []
Action = []
Min_value = []
Episode = []
delay = []
E_consumed = []
EE_consumed = []
Min_nodes_RE = []


initial_energy = 10
sink_energy = 5000000
initial_E_vals = [initial_energy for i in range(len(G))]
ref_E_vals = [initial_energy for i in range(len(G))]

initial_E_vals[sink_node] = sink_energy
ref_E_vals[sink_node] = sink_energy

total_initial_energy = sum(initial_E_vals)

initial_state = random.choice(range(0, len(ST_paths), 1))

# Routing with the LSPI method

for i in range(episodes):

    tx_energy = 0
    rx_energy = 0


    available_actions = [*range(0, len(ST_paths), 1)]

    current_state = initial_state

    for sample in range(len(samples)):
        phi = phi_matrix[samples[sample][0], samples[sample][1]]
        act = np.argmin(phi_matrix[samples[sample][1], :])
        print('act:', act)
        phi_next = phi_matrix[samples[sample][1], act]
        r = samples[sample][3]
        loss = phi - gamma * phi_next
        A = A + phi * loss
        b = b + phi * r
        new_w = b / A
        print('new_w:', new_w)

    phi_matrix = new_w * phi_matrix


    if random.random() >= 1 - epsilon:
        # Get random action
        action = random.choice(available_actions)
    else:
        # Get action from Q table
        action = np.argmin(phi_matrix[current_state, :])

    new_q = phi_matrix[current_state, action]

    initial_state = action

    chosen_MST = ST_paths[action]
    Action.append(action)

    for node in chosen_MST:
        counter = 0
        while counter < len(chosen_MST[node]) - 1:
            ini = chosen_MST[node][counter]
            nex = chosen_MST[node][counter + 1]
            dis = math.ceil(math.sqrt(math.pow((p[ini][0] - p[nex][0]), 2) + math.pow((p[ini][1] - p[nex][1]), 2)))
            # print('dis:', dis)
            if dis <= d_o:
                etx = electronic_energy * data_packet_size + e_fs * data_packet_size * math.pow(dis, 2)
            else:
                etx = electronic_energy * data_packet_size + e_mp * data_packet_size * math.pow(dis, 4)
            erx = electronic_energy * data_packet_size

            initial_E_vals[ini] = initial_E_vals[ini] - num_pac * etx  # update the start node energy

            initial_E_vals[nex] = initial_E_vals[nex] - num_pac * erx  # update the next hop energy

            #tx_energy += num_pac * etx
            #rx_energy += num_pac * erx

            counter += 1

    Q_value.append(new_q)
    Episode.append(i)
    CQ_value.append(sum(Q_value))

    cost = True
    for index, item in enumerate(initial_E_vals):
        if item <= node_energy_limit:
            print('Index:', index)
            cost = False
            print("The final round is", i)
            #print('Average Energy Consumption:', sum(E_consumed))
            #print('Average remaining Energy:', sum([initial_E_vals[i] for i in G.nodes if i != sink_node])/(len(G.nodes)-1))

    if not cost:
        break

    for i in range(len(ref_E_vals)):
        ref_E_vals[i] = initial_E_vals[i]



print("--- %s seconds ---" % (time.time() - start_time))

print('EC:', (total_initial_energy - sum(initial_E_vals))/len(Episode))

for qv in Q_value:
    NQ_value.append(qv/max(Q_value))

my_data = Counter(Action)
print('RT_UT:', my_data.most_common())  # Returns all unique items and their counts

with open('slqals.txt', 'w') as f:
    f.write(json.dumps(NQ_value))

# Now read the file back into a Python list object
with open('slqals.txt', 'r') as f:
    NQ_value = json.loads(f.read())



plt.plot(Episode, Action)
plt.xlabel('Round')
plt.ylabel('Action')
plt.title('Action Convergence ')
plt.show()

plt.plot(Episode, NQ_value)
plt.xlabel('Round')
plt.ylabel('Average Q-Value')
plt.title('Q-Value Convergence ')
plt.show()

plt.plot(Episode, Q_value)
plt.plot(Episode, Min_value)
plt.xlabel('Round')
plt.ylabel('Q-Value')
plt.title('Q-Value Convergence ')
plt.show()

plt.plot(Episode, CQ_value)
plt.xlabel('Round')
plt.ylabel('Q-Value')
plt.title('Q-Value Convergence ')
plt.show()