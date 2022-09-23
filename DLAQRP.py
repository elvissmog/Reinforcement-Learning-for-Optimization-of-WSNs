import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import json



# Instantiating the G object

start_time = time.time()

# initialization of network parameters
learning_rate = 0.1
initial_energy = 100  # Joules
data_packet_size = 1024  # bits
electronic_energy = 50e-9  # Joules/bit 50e-9
e_fs = 10e-12  # Joules/bit/(meter)**2
e_mp = 0.0013e-12 #Joules/bit/(meter)**4
node_energy_limit = 0
num_pac = 10
txr = 150
epsilon = 0.1
discount_factor = 0.8
wi = 1
wii = 1
wiii = 1

sink_node = 100



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

# Extracting the (x,y) coordinate of each node to enable the calculation of the Euclidean distances of the G edges
p = []
for node in sorted(G):
    p.append(G.nodes[node]['pos'])

Trx_dis = []
for u, v in list_unweighted_edges:
    distance = math.sqrt(math.pow((p[u][0] - p[v][0]), 2) + math.pow((p[u][1] - p[v][1]), 2))

    if distance <= txr:
        Trx_dis.append(distance)
        G.add_edge(u, v, weight = distance)
        #G.add_edge(u, v, weight=1)

com_range = max(Trx_dis)

print('cm_range:', com_range)

# Building a minimum spanning tree sub-graph, T of the main graph, G
T = nx.minimum_spanning_tree(G, algorithm='kruskal')

node_neighT = {}  # the set of neighbors of all nodes in the graph
for n in T.nodes:
    node_neighT[n] = list(T.neighbors(n))

#print("The nodes and their neighbors are ", node_neighT)
#print("Minimum Spanning Tree Edges is:", red_edges)

node_neigh = {}  # the set of neighbors of all nodes in the graph
for n in G.nodes:
    node_neigh[n] = list(G.neighbors(n))



E_vals = [initial_energy for i in range(len(G))]

E_vals[sink_node] = 500000
total_initial_energy = sum(E_vals)

hop_counts = {}
for n in T.nodes:
    for path in nx.all_simple_paths(T, source=n, target=sink_node):
        hop_counts[n] = len(path) - 1

hop_counts[sink_node]= 1   #hop count of sink

# Initialize Q_values
q_values = {}

for node in hop_counts:
    #if node != sink_node:
    q_values[node] = (E_vals[node] / hop_counts[node])

d_o = math.ceil(math.sqrt(e_fs/e_mp))

Q_vals = [q_values for i in range(len(G))]
path_Q_values = [[[0 for i in range(len(G))] for j in range(len(G))] for n in range(len(G))]


#print('Q_vals:', Q_vals)
#print("The hop counts to sink for the nodes are ", hop_counts)  # format is {node:count}

num_of_episodes = 5000000
round = []
Av_mean_Q = []
EE_consumed = []
NQ = []

for i in range(num_of_episodes):

    mean_Q = []
    E_consumed = []
    delay = []
    path_f = []

    for node in range(len(G.nodes)-1):
        if node != sink_node:
            s = node
            queue = [s]  # first visited node
            path = str(s)  # first node
            temp_qval = {}
            initial_delay = 0
            txe = 0
            rxe = 0


            while True:

                for n in node_neigh[s]:

                    dss = math.ceil(math.sqrt(math.pow((p[s][0] - p[sink_node][0]), 2) + math.pow((p[s][1] - p[sink_node][1]), 2)))
                    dns = math.ceil(
                        math.sqrt(math.pow((p[n][0] - p[sink_node][0]), 2) + math.pow((p[n][1] - p[sink_node][1]), 2)))

                    dsn = math.ceil(math.sqrt(
                        math.pow((p[s][0] - p[n][0]), 2) + math.pow((p[s][1] - p[n][1]), 2)))

                    #if hop_counts[s] >= hop_counts[n]:  #dss >= dns and
                    '''
                    if dsn <= d_o:
                        rwd = E_vals[n] / (((dsn / d_o) ** 2) * hop_counts[n])

                    else:
                        rwd = E_vals[n] / (((dsn / d_o) ** 4) * hop_counts[n])

                    '''



                    rwd = wi*((E_vals[n]-E_vals[s])/initial_energy) + wii*(dss/dsn) + wiii*(hop_counts[s]/hop_counts[n])

                    #print('rwd:', rwd)

                    temp_qval[n] = (1 - learning_rate) * path_Q_values[node][s][n] + learning_rate * (rwd + discount_factor * Q_vals[node][n])



                copy_q_values = {key: value for key, value in temp_qval.items() if key not in queue}

                if np.random.random() >= 1 - epsilon:
                    # Get action from Q table
                    nh = random.choice(list(copy_q_values.keys()))
                else:
                    # Get random action
                    nh = max(copy_q_values.keys(), key=(lambda k: copy_q_values[k]))




                queue.append(nh)

                path_Q_values[node][s][nh] = temp_qval[nh]  # update the path qvalue of the next hop
                Q_vals[node][s] = temp_qval[nh]  # update the qvalue of the start node
                hop_counts[s] = hop_counts[nh] + 1


                mean_Qvals = sum([Q_vals[node][k] for k in Q_vals[node]]) / (len(Q_vals[node]) * max([Q_vals[node][k] for k in Q_vals[node]]))

                dsnh = math.ceil(math.sqrt(math.pow((p[s][0] - p[nh][0]), 2) + math.pow((p[s][1] - p[nh][1]), 2)))
                if dsnh <= d_o:
                    etx = electronic_energy * data_packet_size + e_fs * data_packet_size * math.pow(dsnh, 2)
                else:
                    etx = electronic_energy * data_packet_size + e_fs * data_packet_size * math.pow(dsnh, 2)

                erx = electronic_energy * data_packet_size

                E_vals[s] = E_vals[s] - num_pac * etx  # update the start node energy
                E_vals[nh] = E_vals[nh] - num_pac * erx  # update the next hop energy

                #txr += num_pac * etx
                #rxe += num_pac * erx

                #path = path + "->" + str(next_hop)  # update the path after each visit

                # print("The visited nodes are", queue)

                s = nh

                if nh == sink_node:
                    break


            mean_Q.append(mean_Qvals)
            E_consumed.append(txe + rxe)
            #print('E_consumed:', E_consumed)

        #path_f.append(path)
    #print('path:', path_f)
    Av_mean_Q.append(sum(mean_Q)/len(mean_Q))

    #print('EE_consumed:', EE_consumed)
    #EE_consumed.append(sum(E_consumed))

    round.append(i)

    cost = True
    for index, item in enumerate(E_vals):
        if item <= node_energy_limit:
            cost = False
            print("Energy cannot be negative!")
            print("The final round is", i)
            print('Index:', index)


    if not cost:
        break

print("--- %s seconds ---" % (time.time() - start_time))

print('EC:', (total_initial_energy - sum(E_vals))/len(round))

for qv in Av_mean_Q:
    NQ.append(qv/max(Av_mean_Q))

with open('dlaqrp.txt', 'w') as f:
    f.write(json.dumps(NQ))

# Now read the file back into a Python list object
with open('dlaqrp.txt', 'r') as f:
    NQ = json.loads(f.read())

'''
plt.plot(round, NQ)
plt.xlabel('Round')
plt.ylabel('Average Q-Value')
plt.title('Q-Value Convergence ')
plt.show()
'''



