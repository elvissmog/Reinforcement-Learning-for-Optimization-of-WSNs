import numpy as np
import networkx as nx
import random
import math
import time
from pqdict import PQDict
import json

start_time = time.time()

sink_node = 1000

def build_graph(positions, links):
    G = nx.Graph()
    for i in positions:
        G.add_node(i, pos=positions[i])

    global position_array
    position_array = {}
    for nd in G:
        position_array[nd] = G.nodes[nd]['pos']

    for u, v in links:
        distance = math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow(
            (position_array[u][1] - position_array[v][1]), 2))
        nor_distance = math.ceil(distance / txr)
        G.add_edge(u, v, weight=nor_distance)

    def is_tree_of_graph(child, parent):

        parent_edges = parent.edges()
        for child_edge in child.edges():
            if child_edge not in parent_edges:
                return False
        return nx.is_tree(child)

    mst = nx.minimum_spanning_tree(G)
    mst_edge = mst.edges()
    # Calculating the minimum spanning tree cost of G
    mst_edge_cost = []
    for tu, tv in mst_edge:
        mst_edge_dis = math.sqrt(math.pow((position_array[tu][0] - position_array[tv][0]), 2) + math.pow(
            (position_array[tu][1] - position_array[tv][1]), 2))
        mst_edge_cost.append(math.ceil(mst_edge_dis / txr))

    mst_cost = sum(mst_edge_cost)

    # start_time = time.time()
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
            while (tup[1] in closedSet):
                tup, wght = pq.popitem()
            mst.append(tup)
            current = tup[1]

        h = nx.Graph()

        for j in xy:
            h.add_node(j, pos=xy[j])

        for (x, y) in mst:
            distance = math.sqrt(math.pow((position_array[x][0] - position_array[y][0]), 2) + math.pow((position_array[x][1] - position_array[y][1]), 2))

            h.add_edge(x, y, weight=math.ceil(distance / txr))

        return h

    MST = []

    # Extracting the edges of unique MST from Prims algorithm and storing in MST_edges
    MST_edges = []

    for i in G.nodes():
        y = prim(G, i)
        MST.append(y)
        # print(list(y.edges))
        if set(y.edges()) not in MST_edges:
            MST_edges.append(set(y.edges()))

    # Converting the unique MST edges into graph and storing in unique_MSTs

    unique_MSTs = []
    for M_edges in MST_edges:
        t = nx.Graph()
        for p in xy:
            t.add_node(p, pos=xy[p])

        for (u, v) in M_edges:
            dis = math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow(
                (position_array[u][1] - position_array[v][1]), 2))

            t.add_edge(u, v, weight=math.ceil(dis / txr))
        unique_MSTs.append(t)

    # Generating new MSTs from the unique_MSTs using genetic algorithm

    for idx in range(ng):
        ind_pop = []  # initializing an empty intermidiate population to store unique children
        nc = int(100 / cr)

        # Crossover Operator
        for n_c in range(nc):
            two_ind = random.sample(unique_MSTs, 2)  # Randomly selecting two parents from the population(unique_MST) to perform crosss over operator

            gr_ed = []
            for gr in two_ind:
                gr_ed.append(list(gr.edges()))
            # print(gr_ed)
            sub_graph_edges = list(set(gr_ed[0] + gr_ed[1]))
            # print(sub_graph_edges)
            sub_graph = nx.Graph()
            for sp in xy:
                sub_graph.add_node(sp, pos=xy[sp])

            for (su, sv) in sub_graph_edges:
                dis = math.sqrt(math.pow((position_array[su][0] - position_array[sv][0]), 2) + math.pow((position_array[su][1] - position_array[sv][1]), 2))
                sub_graph.add_edge(su, sv, weight=math.ceil(dis / txr))

            new_mst = nx.minimum_spanning_tree(sub_graph)
            # print(list(new_mst.edges))

            if set(new_mst.edges()) not in ind_pop:
                ind_pop.append(set(new_mst.edges()))

        # Mutation Operator
        nm = int(100 / mr)
        for n_m in range(nm):
            one_ind = random.sample(unique_MSTs, 1)  # Randomly selecting two parents from the population(unique_MST) to perform crosss over operator

            sgr_ed = []
            for sgr in one_ind:
                sgr_ed.append(list(sgr.edges()))
            sgr_ed = sgr_ed[0]
            # print('sgr ed:', sgr_ed)
            one_edge = random.sample(sgr_ed, 1)
            # print('edge to delete:', one_edge)
            sgr_ed.remove(one_edge[0])
            # print('sgr ed:', sgr_ed)
            su_graph_edges = [value for value in list_unweighted_edges if value not in one_edge]
            # print('sub_graph_edges:', su_graph_edges)
            su_graph = nx.Graph()
            for su in xy:
                su_graph.add_node(su, pos=xy[su])

            for (us, vs) in su_graph_edges:
                dis = math.sqrt(math.pow((position_array[us][0] - position_array[vs][0]), 2) + math.pow(
                    (position_array[us][1] - position_array[vs][1]), 2))

                su_graph.add_edge(us, vs, weight=math.ceil(dis) / txr)

            cut_set = [value for value in su_graph_edges if value not in sgr_ed]
            # print('cut_set:', cut_set)
            add_edge = random.sample(cut_set, 1)
            # print('add_edge:', add_edge)
            sgr_ed = sgr_ed + add_edge
            # print('sgr ed:', sgr_ed)
            mu_graph = nx.Graph()
            for mu in xy:
                su_graph.add_node(mu, pos=xy[mu])

            for (ms, vu) in sgr_ed:
                dis = math.sqrt(math.pow((position_array[ms][0] - position_array[vu][0]), 2) + math.pow(
                    (position_array[ms][1] - position_array[vu][1]), 2))

                mu_graph.add_edge(ms, vu, weight=math.ceil(dis / txr))

            if is_tree_of_graph(mu_graph, G):
                if set(mu_graph.edges()) not in ind_pop:
                    ind_pop.append(set(mu_graph.edges()))

        # Calculate the fitness of each individual in the intermidate population.
        # Converting each individual in the intermidate population to graph and store in ind_pop_graph
        ind_pop_graphs = []
        for ipg_edges in ind_pop:
            ipg = nx.Graph()
            for pg in xy:
                ipg.add_node(pg, pos=xy[pg])

            for (pu, pv) in ipg_edges:
                dis = math.sqrt(math.pow((position_array[pu][0] - position_array[pv][0]), 2) + math.pow(
                    (position_array[pu][1] - position_array[pv][1]), 2))

                ipg.add_edge(pu, pv, weight=math.ceil(dis / txr))
            ind_pop_graphs.append(ipg)

        # print('ind_pop_graphs:', ind_pop_graphs)

        # Calculating the spanning tree cost of each individual in Population.

        for pop in ind_pop_graphs:
            pop_edges = set(pop.edges())  # Extracting the spanning tree graph edges
            # print(pop_edges)
            pop_Spanning_Tree_Edge_Distances = []
            for up, vp in pop_edges:
                dist_edge = math.sqrt(math.pow((position_array[up][0] - position_array[vp][0]), 2) + math.pow(
                    (position_array[up][1] - position_array[vp][1]), 2))
                pop_Spanning_Tree_Edge_Distances.append(math.ceil(dist_edge / txr))
            pop_Tree_Cost = sum(pop_Spanning_Tree_Edge_Distances)

            # print("pop spanning tree cost is:", pop_Tree_Cost)
            if pop_Tree_Cost <= mst_cost:
                if pop_edges not in MST_edges:
                    MST_edges.append(pop_edges)

        unique_sol = []

        for p_edges in MST_edges:
            us = nx.Graph()
            for up in xy:
                us.add_node(up, pos=xy[up])

            for (su, sv) in p_edges:
                dis = math.sqrt(math.pow((position_array[su][0] - position_array[sv][0]), 2) + math.pow(
                    (position_array[su][1] - position_array[sv][1]), 2))

                us.add_edge(su, sv, weight=math.ceil(dis / txr))
            unique_sol.append(us)

        final_pop = []
        for uni_pop in unique_sol:
            if set(uni_pop.edges()) not in final_pop:
                final_pop.append(set(uni_pop.edges()))

        MST_edges = final_pop
        unique_MSTs = unique_sol


    all_msts = unique_MSTs


    MST_paths = []
    for T in all_msts:

        MST_path = {}
        for n in T.nodes:
            for path in nx.all_simple_paths(T, source=n, target=sink_node):
                MST_path[n] = path

        MST_paths.append(MST_path)

    # Q_matrix of all Msts
    Q_matrix = np.zeros((len(all_msts), len(all_msts)))
    # Energy consumption

    initial_e_vals = {}
    for idx in G.nodes:
        if idx != sink_node:
            initial_e_vals[idx] = initial_energy
        else:
            initial_e_vals[idx] = sink_energy

    ref_e_vals = {}
    for idx in G.nodes:
        if idx != sink_node:
            ref_e_vals[idx] = initial_energy
        else:
            ref_e_vals[idx] = sink_energy





    return G, all_msts, MST_paths, initial_e_vals, ref_e_vals, Q_matrix


with open('edges.txt', 'r') as filehandle:
    edges_list = json.load(filehandle)

list_unweighted_edges = []
for ed in edges_list:
    list_unweighted_edges.append(tuple(ed))

with open('pos.txt', 'r') as filehandle:
    pos_list = json.load(filehandle)

x_y = []
for ps in pos_list:
    x_y.append(tuple(ps))

xy = {}
for index in range(len(x_y)):
    xy[index] = x_y[index]

# initialization of network parameters
discount_factor = 0
learning_rate = 0.9
initial_energy = 1000  # Joules
data_packet_size = 1024  # bits
electronic_energy = 50e-9  # Joules/bit 5
txr = 1              # Transmission range
e_fs = 10e-12  # Joules/bit/(meter)**2
e_mp = 0.0013e-12  # Joules/bit/(meter)**4
node_energy_limit = 1
epsilon = 0.1
episodes = 5000000
sink_energy = 5000000
num_pac = 1
cr = 1  # crossover rate
mr = 50  # mutation rate
ng = 200  # number of generations


d_o = math.sqrt(e_fs / e_mp) / txr

#Cum_reward = []
#Q_value = []
#Min_value = []
#Episode = []
#E_consumed = []
#EE_consumed = []
No_Alive_Node = []

graph, rts, rtp, initial_E_vals, ref_E_vals, q_matrix = build_graph(xy, list_unweighted_edges)



for rdn in range(episodes):

    initial_state = random.choice(range(0, len(rts), 1))
    tx_energy = 0
    rx_energy = 0
    #Episode.append(rdn)
    available_actions = [*range(0, len(rts), 1)]

    current_state = initial_state

    if np.random.random() > epsilon:
        # Get random action
        action = np.argmin(q_matrix[current_state, :])

    else:
        # Get action from Q table
        action = random.choice(available_actions)

    initial_state = action

    chosen_MST = rtp[action]
    # print('chosen MST:', chosen_MST)

    # Data transmission
    for node in chosen_MST:
        counter = 0
        while counter < len(chosen_MST[node]) - 1:
            init_node = chosen_MST[node][counter]
            next_node = chosen_MST[node][counter + 1]
            dis = math.sqrt(math.pow((xy[init_node][0] - xy[next_node][0]), 2) + math.pow((xy[init_node][1] - xy[next_node][1]), 2)) / txr
            if dis <= d_o:
                etx = electronic_energy * data_packet_size + e_fs * data_packet_size * math.pow(dis, 2)
            else:
                etx = electronic_energy * data_packet_size + e_mp * data_packet_size * math.pow(dis, 4)
            erx = electronic_energy * data_packet_size

            initial_E_vals[init_node] = initial_E_vals[init_node] - num_pac * etx  # update the start node energy

            initial_E_vals[next_node] = initial_E_vals[next_node] - num_pac * erx  # update the next hop energy

            tx_energy += num_pac * etx

            rx_energy += num_pac * erx

            counter += 1

    # reward = initial_E_vals[max(initial_E_vals.keys(), key=(lambda k: initial_E_vals[k]))]


    Energy_Consumption = [ref_E_vals[i] - initial_E_vals[i] for i in graph.nodes if i != sink_node]
    reward = max(Energy_Consumption)

    #Min_value.append(reward)
    #Cum_reward.append(sum(Min_value))

    # Maximum possible Q value in next step (for new state)
    max_future_q = np.max(q_matrix[action, :])

    # Current Q value (for current state and performed action)
    current_q = q_matrix[current_state, action]
    # And here's our equation for a new Q value for current state and action
    new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
    # new_q = (1 - learning_rate) * current_q + learning_rate * discount_factor *reward
    #Q_value.append(new_q)

    # Update Q table with new Q value
    q_matrix[current_state, action] = new_q

    #E_consumed.append(tx_energy + rx_energy)
    #EE_consumed.append(sum(E_consumed))
    No_Alive_Node.append(len(xy) - 1)
    cost = 0

    dead_node = []

    for index, item in initial_E_vals.items():

        if item <= node_energy_limit:

            dead_node.append(index)

            if index in xy.keys():
                xy.pop(index)

    test = [(item1, item2) for item1, item2 in list_unweighted_edges if item1 not in dead_node and item2 not in dead_node]


    list_unweighted_edges = test

    update_evals = {index: item for index, item in initial_E_vals.items() if item > node_energy_limit}

    if len(dead_node) >= 1:
        #print('Round:', rdn)
        #print('Dead node:', dead_node)

        try:
            graph, rts, rtp, initial_E_vals, ref_E_vals, q_matrix = build_graph(xy, list_unweighted_edges)

            initial_E_vals = update_evals
            ref_E_vals = initial_E_vals
            update_qmatrix = np.ones((len(rts), len(rts)))
            q_matrix = update_qmatrix * new_q

        except (ValueError, IndexError, KeyError):
            # Error = messagebox.showinfo("Enter proper values")
            print('lifetime:', rdn)
            break

    dead_node = []

print("--- %s seconds ---" % (time.time() - start_time))

with open('sonan.txt', 'w') as f:
    f.write(json.dumps(No_Alive_Node))

# Now read the file back into a Python list object
with open('sonan.txt', 'r') as f:
    No_Alive_Node = json.loads(f.read())


