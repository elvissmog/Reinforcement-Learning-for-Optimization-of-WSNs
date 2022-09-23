import numpy as np
import networkx as nx
import random
import math
import matplotlib.pyplot as plt
from pqdict import PQDict
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

def is_tree_of_graph(child, parent):

    parent_edges = parent.edges()
    for child_edge in child.edges():
        if child_edge not in parent_edges:
            return False
    return nx.is_tree(child)


for i in range(len(xy)):
	G.add_node(i, pos=xy[i])

position_array = []
for node in sorted(G):
    position_array.append(G.nodes[node]['pos'])
for u, v in list_unweighted_edges:
    distance = math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow((position_array[u][1] - position_array[v][1]), 2))
    G.add_edge(u, v, weight = math.ceil(distance/txr))

mst = nx.minimum_spanning_tree(G)
mst_edge = mst.edges()
# Calculating the minimum spanning tree cost of g
mst_edge_cost = []
for tu, tv in mst_edge:
    mst_edge_dis = math.sqrt(math.pow((position_array[tu][0] - position_array[tv][0]), 2) + math.pow((position_array[tu][1] - position_array[tv][1]), 2))
    mst_edge_cost.append(math.ceil(mst_edge_dis/txr))

mst_cost = sum(mst_edge_cost)

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
        distance = math.sqrt(math.pow((position_array[x][0] - position_array[y][0]), 2) + math.pow((position_array[x][1] - position_array[y][1]), 2))
        h.add_edge(x, y, weight=math.ceil(distance/txr))

    return h

MST = []

# Extracting the edges of unique MST from Prims algorithm and storing in MST_edges
MST_edges = []

for i in range(len(G.nodes)):
    y = prim(G,i)
    MST.append(y)
    #print(list(y.edges))
    if set(y.edges()) not in MST_edges:
        MST_edges.append(set(y.edges()))

# Converting the unique MST edges into graph and storing in unique_MSTs

unique_MSTs = []
for M_edges in MST_edges:
    t = nx.Graph()
    for p in range(len(xy)):
        t.add_node(p, pos=xy[p])

    for (u, v) in M_edges:
        dis = math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow(
            (position_array[u][1] - position_array[v][1]), 2))

        t.add_edge(u, v, weight=math.ceil(dis/txr))
    unique_MSTs.append(t)


cr = 1   # crossover rate
mr = 50   # mutation rate
ng = 100   # number of generations


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
            dis = math.sqrt(math.pow((position_array[su][0] - position_array[sv][0]), 2) + math.pow(
                (position_array[su][1] - position_array[sv][1]), 2))

            sub_graph.add_edge(su, sv, weight=math.ceil(dis/txr))

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
            dis = math.sqrt(math.pow((position_array[us][0] - position_array[vs][0]), 2) + math.pow(
                (position_array[us][1] - position_array[vs][1]), 2))

            su_graph.add_edge(us, vs, weight=math.ceil(dis)/txr)

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
            dis = math.sqrt(math.pow((position_array[ms][0] - position_array[vu][0]), 2) + math.pow(
                (position_array[ms][1] - position_array[vu][1]), 2))

            mu_graph.add_edge(ms, vu, weight=math.ceil(dis/txr))

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
            dis = math.sqrt(math.pow((position_array[pu][0] - position_array[pv][0]), 2) + math.pow(
                (position_array[pu][1] - position_array[pv][1]), 2))

            ipg.add_edge(pu, pv, weight=math.ceil(dis/txr))
        ind_pop_graphs.append(ipg)

    #print('ind_pop_graphs:', ind_pop_graphs)

    # Calculating the spanning tree cost of each individual in Population.

    for pop in ind_pop_graphs:
        pop_edges = set(pop.edges())  # Extracting the spanning tree graph edges
        #print(pop_edges)
        pop_Spanning_Tree_Edge_Distances = []
        for up, vp in pop_edges:
            dist_edge = math.sqrt(math.pow((position_array[up][0] - position_array[vp][0]), 2) + math.pow(
                (position_array[up][1] - position_array[vp][1]), 2))
            pop_Spanning_Tree_Edge_Distances.append(math.ceil(dist_edge/txr))
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
            dis = math.sqrt(math.pow((position_array[su][0] - position_array[sv][0]), 2) + math.pow(
                (position_array[su][1] - position_array[sv][1]), 2))

            us.add_edge(su, sv, weight=math.ceil(dis/txr))
        unique_sol.append(us)

    final_pop = []
    for uni_pop in unique_sol:
        if set(uni_pop.edges()) not in final_pop:
            final_pop.append(set(uni_pop.edges()))

    MST_edges = final_pop
    unique_MSTs = unique_sol

# initialization of network parameters
discount_factor = 0
learning_rate = 0.9
initial_energy = 1        # Joules
data_packet_size = 1024      # bits
electronic_energy = 50e-9   # Joules/bit 5
e_fs = 10e-12               # Joules/bit/(meter)**2
e_mp = 0.0013e-12           #Joules/bit/(meter)**4
node_energy_limit = 0

d = [[0 for i in range(len(G))] for j in range(len(G))]
Etx = [[0 for i in range(len(G))] for j in range(len(G))]
Erx = [[0 for i in range(len(G))] for j in range(len(G))]
initial_E_vals = [initial_energy for i in range(len(G))]
ref_E_vals = [initial_energy for i in range(len(G))]
epsilon = 0.1
episodes = 5000000
sink_energy = 5000000

sink_node = 100


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


H_counts = []
for T in all_STs:
    hop_counts = {}
    for n in T.nodes:
        for path in nx.all_simple_paths(T, source=n, target=sink_node):
            hop_counts[n] = len(path) - 1

    hop_counts[sink_node] = 1  # hop count of sink

    H_counts.append(hop_counts)


Q_matrix = np.zeros((len(ST_paths), len(ST_paths)))
initial_state = random.choice(range(0, len(ST_paths), 1))

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


for i in range(episodes):

    tx_energy = 0
    rx_energy = 0



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
            #tx_energy += Etx[init_node][next_node]
            #rx_energy += Erx[init_node][next_node]
            counter += 1

    rew = [(ref_E_vals[i] - initial_E_vals[i]) / H_counts[action][i] for i in G.nodes if i != sink_node]  # Energy consumption per hop count


    reward = min(rew)


    Min_value.append(reward)

    # Maximum possible Q value in next step (for new state)
    max_future_q = np.max(Q_matrix[action, :])

    # Current Q value (for current state and performed action)
    current_q = Q_matrix[current_state, action]

    # And here's our equation for a new Q value for current state and action

    new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)

    Q_value.append(new_q)
    Episode.append(i)
    #CQ_value.append(sum(Q_value))

    # Update Q table with new Q value
    Q_matrix[current_state, action] = new_q

    #E_consumed.append(tx_energy + rx_energy)

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

with open('islqals.txt', 'w') as f:
    f.write(json.dumps(NQ_value))

# Now read the file back into a Python list object
with open('islqals.txt', 'r') as f:
    NQ_value = json.loads(f.read())




