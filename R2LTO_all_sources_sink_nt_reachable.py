import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import time

start_time = time.time()

# initialization of network parameters
learning_rate = 0.9
initial_energy = 1  # Joules
sink_node_energy = 500000
data_packet_size = 1024  # bits
electronic_energy = 50e-9  # Joules/bit 50e-9
e_fs = 10e-12  # Joules/bit/(meter)**2
e_mp = 0.0013e-12 #Joules/bit/(meter)**4
node_energy_limit = 0
epsilon = 0.0
transmission_range = 1
sink_node = 100
num_of_episodes = 5000000


with open('edges1.txt', 'r') as filehandle:
    edges_list = json.load(filehandle)

list_unweighted_edges = []
for ed in edges_list:
    list_unweighted_edges.append(tuple(ed))

with open('pos1.txt', 'r') as filehandle:
    pos_list = json.load(filehandle)

x_y = []
for ps in pos_list:
    x_y.append(tuple(ps))

xy = {}
for index in range(len(x_y)):
    xy[index] = x_y[index]


d_o = math.ceil(math.sqrt(e_fs/e_mp)/transmission_range)

def build_graph(positions, links):
    G = nx.Graph()
    for i in positions:
        G.add_node(i, pos=positions[i])

    # Extracting the (x,y) coordinate of each node to enable the calculation of the Euclidean distances of the Graph edges
    global position_array
    position_array = {}
    for nd in G:
        position_array[nd] = G.nodes[nd]['pos']

    #Adding unweighted edges to the graph and calculating the distances
    for u, v in links:
        distance = math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow((position_array[u][1] - position_array[v][1]), 2))
        nor_distance = math.ceil(distance / transmission_range)
        G.add_edge(u, v, weight=nor_distance)

    # Building a minimum spanning tree sub-graph, T of the main graph, G
    T = nx.minimum_spanning_tree(G, algorithm='kruskal')

    # The dictionary of neighbors of all nodes in the graph
    node_neigh = {}
    for n in G.nodes:
        node_neigh[n] = list(G.neighbors(n))


    hop_counts = {}
    for n in T.nodes:
        for path in nx.all_simple_paths(T, source=n, target=sink_node):
            hop_counts[n] = len(path) - 1

    hop_counts[sink_node] = 1      # hop count of sink


    # Energy consumption
    e_vals = {}

    for idx in G.nodes:
        if idx != sink_node:
            e_vals[idx] = initial_energy
        else:
            e_vals[idx] = sink_node_energy

    q_vals = {}
    for ix in G.nodes:
        q_vals[ix] = (e_vals[ix] / hop_counts[ix])
        #q_vals[ix] = 0

    all_q_vals = {}
    for iix in G.nodes:
        all_q_vals[iix] = q_vals

    path_q_vals = {}
    for xi in G.nodes:
        path_q_vals[xi] = q_vals

    all_path_q_vals = {}
    for xii in G.nodes:
        all_path_q_vals[xii] = path_q_vals

    rwd = {}
    for ix in G.nodes:
        rwd[ix] = 0

    path_rwd = {}
    for xi in G.nodes:
        path_rwd[xi] = rwd

    all_path_rwds = {}
    for xii in G.nodes:
        all_path_rwds[xii] = path_rwd

    dist = {}
    for di in G.nodes:
        dist[di] = 0
    all_dist = {}
    for dii in G.nodes:
        all_dist[dii] = dist

    trx_pow = {}
    for t in G.nodes:
        trx_pow[t] = 0

    all_trx_pow = {}
    for ti in G.nodes:
        all_trx_pow[ti] = trx_pow

    for n in G.nodes:
        for nn in G.nodes:
            if n != nn:
                all_dist[n][nn] = math.ceil((math.sqrt(math.pow((xy[n][0] - xy[nn][0]), 2) + math.pow((xy[n][1] - xy[nn][1]), 2)))/transmission_range)
                if all_dist[n][nn] <= d_o:
                    all_trx_pow[n][nn] = electronic_energy * data_packet_size + e_fs * data_packet_size * math.pow((all_dist[n][nn]), 2)
                else:
                    all_trx_pow[n][nn] = electronic_energy * data_packet_size + e_mp * data_packet_size * math.pow((all_dist[n][nn]), 4)


    return G, hop_counts, node_neigh, all_q_vals, e_vals, all_path_q_vals, all_path_rwds, all_dist, all_trx_pow
    #return G, node_neigh, all_q_vals, e_vals, all_path_q_vals, all_path_rwds


Av_mean_Q = []
Av_E_consumed = []
Av_delay = []
No_Alive_Node = []
round = []

graph, h_counts, node_neighbors, q_values, e_values, path_q_values, path_rwds, a_dis, a_trx_pow = build_graph(xy, list_unweighted_edges)
#graph, node_neighbors, q_values, e_values, path_q_values, all_path_rwds = build_graph(xy, list_unweighted_edges)

for rdn in range(num_of_episodes):

    mean_Q = []
    E_consumed = []
    EE_consumed = []
    delay = []
    path_f = []


    for node in graph.nodes:
        if node != sink_node:
            start = node
            queue = [start]  # first visited node
            #path = str(start)  # first node
            temp_qval = dict()
            initial_delay = 0
            tx_energy = 0
            rx_energy = 0


            while True:
                for neigh in node_neighbors[start]:
                    dis_start_sink = math.ceil(math.sqrt(math.pow((xy[start][0] - xy[sink_node][0]), 2) + math.pow((xy[start][1] - xy[sink_node][1]), 2))/transmission_range)
                    dis_neigh_sink = math.ceil(math.sqrt(math.pow((xy[neigh][0] - xy[sink_node][0]), 2) + math.pow((xy[neigh][1] - xy[sink_node][1]), 2))/transmission_range)
                    #if dis_start_sink >= dis_neigh_sink and h_counts[start] >= h_counts[neigh]:
                    dis_start_neigh = math.ceil(math.sqrt(math.pow((xy[start][0] - xy[neigh][0]), 2) + math.pow((xy[start][1] - xy[neigh][1]), 2))/transmission_range)
                    max_tx = max([a_trx_pow[start][k] for k in a_trx_pow[start]])

                    path_rwds[node][start][neigh] = e_values[neigh] / (a_trx_pow[start][neigh] * h_counts[neigh])
                    #all_path_rwds[node][start][neigh] = e_values[neigh] / ((dis_start_neigh / d_o) ** 2)


                    temp_qval[neigh] = (1 - learning_rate) * path_q_values[node][start][neigh] + learning_rate * (path_rwds[node][start][neigh] + q_values[node][neigh])

                copy_q_values = {key: value for key, value in temp_qval.items() if key not in queue}
                # next_hop = min(temp_qval.keys(), key=(lambda k: temp_qval[k]))  #determine the next hop based on the minimum qvalue but ignore the visited node qvalue

                if np.random.random() >= 1 - epsilon:
                    # Get action from Q table
                    next_hop = random.choice(list(copy_q_values.keys()))
                else:
                    # Get random action

                    next_hop = max(copy_q_values.keys(), key=(lambda k: copy_q_values[k]))

                queue.append(next_hop)

                path_q_values[node][start][next_hop] = temp_qval[next_hop]    # update the path qvalue of the next hop
                q_values[node][start] = temp_qval[next_hop]                   # update the qvalue of the start node

                mean_Qvals = sum([q_values[node][k] for k in q_values[node]]) / (len(q_values[node]) * max([q_values[node][k] for k in q_values[node]]))
                dis = math.sqrt(math.pow((xy[start][0] - xy[next_hop][0]), 2) + math.pow((xy[start][1] - xy[next_hop][1]), 2))
                nor_dis = math.ceil(dis / transmission_range)
                if nor_dis <= d_o:
                    etx = electronic_energy * data_packet_size + e_fs * data_packet_size * math.pow(nor_dis, 2)
                else:
                    etx = electronic_energy * data_packet_size + e_mp * data_packet_size * math.pow(nor_dis, 4)
                erx = electronic_energy * data_packet_size
                e_values[start] = e_values[start] - etx                      # update the start node energy
                e_values[next_hop] = e_values[next_hop] - erx                # update the next hop energy
                #tx_energy += etx
                #rx_energy += erx
                #initial_delay += dis

                #path = path + "->" + str(next_hop)  # update the path after each visit

                start = next_hop

                if next_hop == sink_node:
                    break

            #delay.append(initial_delay)
            #E_consumed.append(tx_energy + rx_energy)
            #mean_Q.append(mean_Qvals)

        #path_f.append(path)
        #Av_mean_Q.append(sum(mean_Q) / len(mean_Q))
        #Av_delay.append(sum(delay) / len(delay))
        #Av_E_consumed.append(sum(E_consumed))
        No_Alive_Node.append(len(graph.nodes) - 1)
        round.append(rdn)

    dead_node = []

    for index, item in e_values.items():

        if item <= node_energy_limit:

            dead_node.append(index)

            if index in xy.keys():
                xy.pop(index)

    test = [(item1, item2) for item1, item2 in list_unweighted_edges if item1 not in dead_node and item2 not in dead_node]

    list_unweighted_edges = test

    update_evals = {index: item for index, item in e_values.items() if item > node_energy_limit}

    if len(dead_node) >= 1:
        #print('Energy of node has gone below a threshold')
        #print('dead nodes:', dead_node)
        #print("The lifetime at this point is", rdn)
        #print("Number of alive Nodes", len(xy))


        try:
            graph, h_counts, node_neighbors, q_values, e_values, path_q_values, path_rwds, a_dis, a_trx_pow = build_graph(xy, list_unweighted_edges)
            #graph, node_neighbors, q_values, e_values, path_q_values, all_path_rwds = build_graph(xy,list_unweighted_edges)


            e_values = update_evals


        except (ValueError, IndexError, KeyError):

            break

    profit = True

    for nd in graph.nodes:
        if node_neighbors[nd] == [] or len(graph.nodes) == 1:
            profit = False

    if not profit:
        print('lifetime:', rdn)
        break

print("--- %s seconds ---" % (time.time() - start_time))


with open('rttlonan.txt', 'w') as f:
    f.write(json.dumps(No_Alive_Node))

# Now read the file back into a Python list object
with open('rttlonan.txt', 'r') as f:
    No_Alive_Node = json.loads(f.read())


plt.plot(round, No_Alive_Node)
plt.xlabel('Round')
plt.ylabel('NAN')
plt.title('Number of Alive Node')
plt.show()

