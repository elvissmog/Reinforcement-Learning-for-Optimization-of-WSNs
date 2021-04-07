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
    # G.add_edge(u, v, weight = 1)
    G.add_edge(u, v, weight=np.round(distances[u][v], decimals=1))

# initialization of network parameters
discount_factor = 0.5
learning_rate = 0.5
initial_energy = 0.5  # Joules
packet_size = 512  # bits
electronic_energy = 50e-9  # Joules/bit 5
amplifier_energy = 100e-12  # Joules/bit/square meter
transmission_range = 30  # meters
pathloss_exponent = 2  # constant
LearningPeriod = 10                    # Learning epoch duration (in s).
HelloPeriod  = 2                       # Period to broadcast Hello packets
MaxNeighbors = 50                      # Maximum number of nodes in neighborhood.
BR = 1000000                           # bit rate of node (in bps).
RecPw = 20                             # Reception power level of node (in mW).
ProcPw = 3                             # processing power level of node (in mW).
TxTRangeMax = 100                     # Max transmission range of local node (in m)
SystemVoltage = 5                      # Voltage of processing system of node (in volt)
TransceiverOutVoltage = 5              # Voltage of transceiver in transmission mode (in volt)
TransceiverInVoltage = 3               # Voltage of transceiver in reception mode (in volt)
TransceiverIdleVoltage = 3             # Voltage of transceiver in idle mode (in volt)
ProtocolsHeadersSize = 62              # sum of lengths of MAC (34 bytes), IP (20 bytes) and UDP (8 bytes) headers
Ctl_AvgTxT = round(124*LearningPeriod/HelloPeriod)    # Initial value of average control packet transmission load (bytes/sec)
Ctl_AvgRec = round((124*LearningPeriod/HelloPeriod)*(MaxNeighbors/2))     # Initial value of average reception data load estimate.
Data_AvgTxT = 1000*(MaxNeighbors/2)    # Initial value of average data packet transmission load (bytes/sec)
Payload_AvgTxT = Data_AvgTxT       # Average data bytes sent by for sensor per learning period
Data_AvgRec = 1000*(MaxNeighbors/2)    # Initial value of average reception data load estimate.
Battery_Capacity = 100                 # The initial battery capacity of each node in As
CBR_packet_size = 100                  # Size of data of CBR traffic in bytes
CBR_Period = 1                         # Period of CBR traffic in s
ProcPw = 3                             # processing power level of node (in mW).
RecPw = 20                             # Reception power level of node (in mW).
IdlePw = 20                            # Idle power level of node (in mW).
IdleVol = 3                            # Idle Transceiver voltage (in V)
MaxTxtPower = 500                      # Maximum transmission power of current node (in mW)
MinTxtPower = 250                      # Minimum transmission power of current node (in mW)
Battery_Self_Discharge = 0.025         # 3% is the average self-discharge fcactor per month of Lithium batteries


I = [[0 for i in range(len(G))] for j in range(len(G))]
I_Tx_Data = [[0 for i in range(len(G))] for j in range(len(G))]
PTX = [[0 for i in range(len(G))] for j in range(len(G))]
d = [[0 for i in range(len(G))] for j in range(len(G))]
Etx = [[0 for i in range(len(G))] for j in range(len(G))]
Erx = [[0 for i in range(len(G))] for j in range(len(G))]
path_Q_values = [[0 for i in range(len(G))] for j in range(len(G))]
E_vals = [initial_energy for i in range(len(G))]
ERBC = [Battery_Capacity for i in range(len(G))]
NELT = [np.inf for i in range(len(G))]
epsilon = 0.1
episodes = 10000
sink_node = 5
t_Ctl_Tx = Ctl_AvgTxT * 8 / BR        # Control bits transmission time
I_Ctl_Tx = t_Ctl_Tx * (MaxTxtPower / TransceiverOutVoltage) / 1000   #  current used for control bits transmission during the learning period
t_Data_Tx = Data_AvgTxT * 8 / BR    # Data bits transmission time
t_Ctl_Rx =  Ctl_AvgRec * 8 / BR
t_Data_Rx = Data_AvgRec * 8 / BR
t_Rx = t_Ctl_Rx + t_Data_Rx
I_Rx = t_Rx * (RecPw / TransceiverInVoltage) / 1000
t_idle = LearningPeriod - t_Ctl_Tx - t_Data_Tx - t_Rx
I_idle = t_idle * (IdlePw / 1000) / IdleVol
I_proc = LearningPeriod * (ProcPw / 1000) / SystemVoltage

# PowerWM: table of power levels (in mW) to use depending on distance (in m) to neighbor.
# Each entry is composed of power level and a maximum transmission range.
PowerWM = [(100, 250), (250, 600), (500, 1300), (750, 2000), (1000, 3000)]
def pw(Dist):
    P = 0
    i = 0
    while (i < len(PowerWM)) and (P == 0):
        if PowerWM[i][0] >= Dist: P = PowerWM[i][1]
    if P == 0: P = MaxTxtPower
    return P

for i in range(len(G)):
    for j in range(len(G)):
        if i != j:
            d[i][j] = math.sqrt(math.pow((position_array[i][0] - position_array[j][0]), 2) + math.pow(
                (position_array[i][1] - position_array[j][1]), 2))
            PTX[i][j] = pw(d[i][j])
            I_Tx_Data[i][j] = t_Data_Tx * (PTX[i][j] / TransceiverOutVoltage) / 1000
            I[i][j] = I_Ctl_Tx + I_Tx_Data[i][j] + I_Rx + I_idle
            Etx[i][j] = electronic_energy * packet_size + amplifier_energy * packet_size * math.pow((d[i][j]),
                                                                                                    pathloss_exponent)
            Erx[i][j] = electronic_energy * packet_size




#-------------------------------------------------------------------------
# pw returns the transmission power level to transmit at a given distance.
#-------------------------------------------------------------------------


Y = Yamada(G)
all_MSTs = Y.spanning_trees()

# the set of neighbors of all nodes in each MST
node_neigh = []
for T in all_MSTs:
    node_neighT = {}
    for n in T.nodes:
        node_neighT[n] = list(T.neighbors(n))
    node_neigh.append(node_neighT)
# print(node_neigh)

# Ranking nodes in terms of hop count to sink for each MST
MSTs_hop_count = []
MST_paths = []
for T in all_MSTs:
    hop_counts = {}
    MST_path = {}
    for n in T.nodes:
        for path in nx.all_simple_paths(T, source=n, target=sink_node):
            hop_counts[n] = len(path) - 1
            MST_path[n] = path
    hop_counts[sink_node] = 0  # hop count of sink
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
    # print('action is:', action)

    chosen_MST = MST_paths[action]
    #print(chosen_MST)

    for node in chosen_MST:
        counter = 0
        while counter < len(chosen_MST[node]) - 1:
            init_node = chosen_MST[node][counter]
            next_node = chosen_MST[node][counter + 1]
            ERBC[init_node] = (1 - Battery_Self_Discharge) * ERBC[init_node] - I[init_node][next_node] - I_proc  # update the start node energy
            # E_vals[next_node] = E_vals[next_node] - Erx[init_node][next_node]  # update the next hop energy
            NELT[init_node]= ERBC[init_node] * LearningPeriod / I[init_node][next_node]
            counter += 1
            # print("counter", counter)

    reward = min(NELT)
    Min_value.append(reward)
    # Maximum possible Q value in next step (for new state)
    max_future_q = np.max(Q_matrix[action, :])

    # Current Q value (for current state and performed action)
    current_q = Q_matrix[current_state, action]
    # And here's our equation for a new Q value for current state and action
    new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
    # new_q = (1 - learning_rate) * current_q + learning_rate * discount_factor *reward
    Q_value.append(new_q)

    # Update Q table with new Q value
    Q_matrix[current_state, action] = new_q



    cost = True
    for item in NELT:
        if item <= 0:
            cost = False
            print("Energy cannot be negative!")
            print("The final round is", i)

    if not cost:
        break

print('Reward:', Min_value)

# print("--- %s seconds ---" % (time.time() - start_time))

print('Round:', Episode)
print('QVals:', Q_value)

plt.plot(Episode, Q_value, label="Q-Value")
plt.plot(Episode, Min_value, label="Reward")
plt.xlabel('Round')
plt.ylabel('Q-Value')
# plt.title('Q-Value Convergence')
plt.legend()
plt.show()

plt.plot(Episode, Actions)
plt.xlabel('Round')
plt.ylabel('Discrete Action')
# plt.title('Selected Action for each round')
plt.show()






