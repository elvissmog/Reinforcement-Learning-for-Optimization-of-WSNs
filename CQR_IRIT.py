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

'''
xy = [(14, 82), (10, 19), (80, 34), (54, 8), (66, 40), (1, 12), (24, 69), (56, 78), (57, 76), (38, 91), (1, 77), (77, 35), (96, 89), (0, 64), (23, 72), (49, 52), (79, 39), (39, 48), (56, 45), (63, 3), (15, 13), (80, 99), (57, 86), (9, 54), (97, 25), (17, 11), (70, 38), (92, 80), (94, 90), (5, 36), (9, 89), (18, 91), (80, 17), (41, 25), (66, 78), (21, 66), (90, 4), (64, 71), (8, 61), (89, 84), (70, 10), (83, 84), (62, 41), (22, 71), (9, 70), (23, 91), (56, 54), (72, 49), (80, 98), (75, 32), (46, 70), (65, 99), (91, 96), (85, 100), (82, 87), (92, 87), (13, 45), (28, 18), (25, 64), (41, 29), (93, 32), (58, 73), (45, 84), (4, 59), (31, 52), (40, 28), (51, 79), (2, 60), (71, 100), (17, 37), (21, 35), (31, 32), (71, 76), (89, 47), (50, 42), (40, 23), (92, 21), (53, 21), (76, 53), (95, 88), (72, 91), (93, 66), (19, 26), (83, 85), (0, 62), (84, 2), (4, 39), (41, 44), (70, 81), (19, 12), (94, 90), (57, 61), (99, 2), (94, 69), (46, 97), (22, 19), (38, 20), (90, 73), (48, 21), (54, 58)]
for i in range(len(xy)):
	G.add_node(i, pos=xy[i])

# Adding unweighted edges to the Graph

list_unweighted_edges = [(48, 50), (33, 29), (49, 68), (66, 44), (6, 44), (8, 18), (27, 50), (63, 62), (36, 63), (32, 52), (31, 2), (12, 70), (54, 83), (85, 33), (38, 78), (27, 26), (35, 41), (25, 62), (60, 77), (65, 13), (65, 8), (45, 34), (10, 28), (84, 51), (16, 15), (40, 50), (75, 74), (11, 75), (56, 76), (39, 62), (32, 90), (54, 97), (61, 64), (20, 56), (4, 48), (49, 5), (14, 98), (39, 23), (73, 76), (76, 47), (65, 61), (91, 6), (99, 14), (75, 85), (8, 53), (19, 97), (98, 18), (19, 23), (15, 3), (81, 77), (89, 36), (49, 68), (75, 22), (77, 54), (11, 25), (4, 84), (45, 19), (32, 23), (70, 59), (17, 37), (67, 30), (71, 78), (24, 46), (37, 80), (93, 71), (16, 28), (2, 73), (57, 27), (52, 49), (40, 98), (6, 11), (53, 50), (3, 16), (16, 59), (64, 25), (82, 14), (25, 27), (9, 82), (46, 36), (60, 52), (3, 86), (63, 66), (46, 89), (3, 33), (70, 90), (72, 34), (98, 50), (39, 31), (49, 81), (76, 19), (99, 76), (78, 8), (2, 49), (80, 14), (61, 0), (51, 31), (82, 61), (42, 63), (88, 97), (1, 68), (81, 47), (54, 11), (26, 2), (0, 40), (64, 25), (93, 61), (40, 8), (17, 84), (91, 55), (32, 2), (61, 47), (1, 2), (24, 4), (26, 31), (63, 69), (2, 26), (78, 29), (47, 14), (58, 4), (77, 40), (55, 30), (16, 52), (21, 13), (37, 59), (93, 86), (28, 84), (46, 35), (48, 42), (36, 4), (56, 44), (90, 1), (8, 25), (82, 15), (39, 99), (15, 89), (43, 40), (11, 98), (42, 81), (74, 16), (22, 2), (18, 65), (39, 53), (33, 34), (16, 11), (67, 22), (81, 74), (69, 6), (20, 54), (85, 89), (67, 13), (46, 0), (81, 78), (55, 63), (41, 4), (34, 76), (93, 80), (26, 62), (95, 15), (27, 75), (28, 79), (49, 80), (80, 59), (14, 75), (5, 57), (39, 4), (23, 62), (91, 79), (45, 11), (77, 32), (4, 18), (57, 89), (9, 97), (93, 60), (78, 67), (23, 36), (10, 12), (34, 69), (22, 54), (90, 46), (11, 10), (88, 12), (33, 60), (76, 60), (1, 81), (85, 9), (19, 1), (67, 93), (20, 16), (28, 30), (7, 39), (97, 86), (4, 98), (11, 3), (71, 60), (0, 64), (58, 45), (62, 38), (24, 26), (74, 60), (88, 33), (53, 92), (35, 91), (81, 78), (10, 74), (56, 64), (8, 13), (37, 94), (70, 44), (80, 20), (87, 81), (12, 63), (66, 41), (82, 17), (75, 3), (11, 5), (47, 87), (15, 92), (82, 33), (9, 39), (34, 12), (29, 67), (46, 76), (45, 99), (0, 49), (13, 95), (43, 37), (74, 59), (94, 34), (53, 47), (30, 31), (85, 2), (52, 20), (64, 79), (52, 65), (28, 24), (35, 37), (44, 34), (68, 75), (52, 32), (79, 70), (14, 40), (37, 33), (97, 84), (64, 56), (48, 22), (14, 68), (17, 38), (67, 23), (83, 18), (50, 58), (47, 41), (58, 0), (61, 25), (99, 89), (89, 35), (74, 56), (96, 31), (24, 28), (76, 32), (4, 64), (36, 88), (3, 93), (97, 83), (55, 41), (19, 64), (90, 10), (72, 10), (72, 18), (97, 71), (72, 38), (72, 40), (97, 43), (45, 57), (12, 10), (15, 17), (70, 81), (7, 96), (80, 99), (27, 85), (16, 91), (86, 12), (26, 41), (81, 34), (99, 42), (76, 46), (69, 12), (68, 82), (71, 58), (36, 20), (43, 0), (1, 20), (2, 47), (6, 54), (55, 1), (52, 12), (50, 87), (32, 99), (33, 34), (49, 18), (35, 81), (10, 87), (46, 86), (32, 31), (97, 50), (52, 62), (68, 87), (15, 40), (56, 94), (49, 65), (14, 49), (80, 14), (84, 20), (66, 7), (74, 92), (43, 69), (42, 58), (36, 85), (39, 61), (83, 33), (93, 6), (46, 30), (38, 49), (79, 17), (58, 8), (13, 43), (18, 82), (79, 88), (58, 43), (10, 94), (58, 95), (32, 78), (60, 52), (54, 1), (13, 64), (67, 58), (83, 71), (16, 17), (24, 52), (96, 17), (13, 26), (9, 39), (7, 75), (74, 86), (26, 54), (88, 16), (46, 74), (64, 7), (17, 67), (74, 0), (62, 96), (83, 17), (50, 23), (91, 51), (65, 23), (74, 48), (26, 54), (59, 21), (81, 97), (50, 45), (53, 92), (0, 85), (14, 87), (98, 83), (36, 57), (68, 89), (38, 2), (86, 53), (43, 22), (92, 23), (20, 11), (98, 61), (86, 78), (45, 99), (31, 52), (78, 25), (67, 34), (97, 6), (20, 21), (8, 99), (47, 70), (21, 80), (95, 32), (55, 52), (37, 88), (20, 59), (35, 2), (95, 42), (58, 57), (96, 76), (79, 47), (57, 61), (98, 75), (53, 88), (29, 34), (41, 1), (59, 45), (71, 44), (96, 21), (44, 28), (4, 50), (50, 2), (46, 99), (52, 73), (66, 81), (38, 88), (10, 22), (3, 68), (61, 15), (39, 46), (36, 63), (13, 73), (16, 52), (5, 61), (46, 29), (70, 73), (99, 81), (11, 28), (22, 49), (56, 39), (71, 54), (72, 77), (45, 56), (56, 92), (87, 2), (2, 46), (77, 7), (95, 14), (50, 34), (71, 91), (17, 0), (96, 73), (54, 29), (26, 49), (78, 0), (92, 59), (7, 80), (35, 12), (53, 25), (99, 67), (74, 55), (97, 66), (0, 88), (44, 98), (40, 59), (61, 8), (2, 12), (74, 92), (81, 6), (34, 84), (19, 26), (95, 18), (95, 45), (79, 86), (79, 18), (26, 54), (13, 60), (37, 42), (10, 46), (19, 63), (10, 90), (76, 94), (16, 40), (1, 95), (89, 60), (67, 55), (48, 33), (74, 60), (26, 56), (65, 44), (24, 55), (13, 8), (9, 40), (73, 3), (87, 7), (94, 51), (39, 17), (22, 81), (66, 90), (60, 58), (5, 17), (3, 53), (64, 28), (70, 6), (63, 51), (30, 40), (34, 15), (82, 38), (42, 84), (64, 96), (5, 91), (41, 69), (6, 83), (74, 54), (67, 95), (19, 52), (6, 81), (47, 89), (2, 23), (25, 55), (64, 51), (72, 29), (65, 78), (83, 16), (99, 58), (46, 22), (82, 44), (26, 19), (41, 67), (18, 82), (61, 38), (33, 81), (34, 57), (38, 77), (97, 46), (94, 22), (42, 19), (9, 24), (78, 47), (29, 80), (24, 92), (93, 44), (89, 11), (51, 12), (17, 40), (58, 71), (49, 94), (65, 15), (79, 4), (22, 19), (20, 38), (90, 55), (71, 7), (76, 79), (49, 23), (41, 93), (89, 0), (6, 5), (59, 35), (75, 24), (27, 44), (15, 96), (8, 13), (40, 26), (90, 69), (4, 39), (1, 30), (44, 66), (4, 1), (51, 40), (16, 54), (48, 5), (78, 66), (11, 18), (72, 33), (99, 79), (75, 60), (28, 96), (19, 37), (39, 63), (17, 87), (46, 80), (36, 54), (2, 83), (32, 16), (91, 34), (37, 16), (35, 27), (10, 44), (54, 1), (29, 71), (50, 80), (93, 64), (4, 11), (5, 84), (18, 81), (37, 45), (49, 32), (8, 86), (52, 98), (67, 16), (75, 55), (40, 74), (97, 37), (67, 5), (25, 54), (51, 97), (93, 18), (95, 13), (80, 57), (5, 82), (6, 80), (90, 81), (7, 83), (12, 53), (63, 34), (64, 45), (84, 16), (14, 71), (48, 71), (9, 52), (83, 37), (24, 87), (19, 15), (6, 5), (15, 67), (10, 64), (56, 55), (11, 3), (3, 12), (97, 24), (96, 1), (16, 50), (43, 84), (50, 24), (99, 6), (5, 28), (9, 20), (74, 23), (22, 45), (57, 37), (78, 30), (76, 98), (98, 62), (44, 23), (59, 37), (57, 42), (68, 99), (13, 49), (26, 39), (44, 49), (61, 97), (52, 50), (10, 84), (24, 21), (32, 6), (66, 2), (53, 12), (93, 91), (96, 32), (42, 95), (92, 52), (34, 74), (63, 28), (86, 89), (57, 51), (84, 51), (50, 11), (84, 90), (14, 53), (58, 31), (58, 8), (80, 32), (42, 92), (59, 78), (7, 77), (55, 83), (26, 96), (30, 16), (51, 47), (26, 35), (10, 89), (49, 17), (55, 47), (9, 6), (31, 59), (55, 60), (82, 65), (34, 62), (50, 62), (83, 35), (35, 85), (39, 62), (8, 68), (62, 50), (21, 29), (58, 12), (0, 41), (23, 62), (35, 96), (34, 8), (20, 84), (82, 25), (15, 2), (1, 33), (73, 60), (28, 30), (65, 55), (53, 83), (2, 90), (84, 55), (36, 29), (32, 9), (33, 9), (39, 28), (82, 85), (43, 58), (93, 24), (2, 88), (82, 18), (59, 61), (55, 23), (56, 42), (2, 29), (35, 20), (71, 97), (81, 65), (10, 48), (74, 48), (14, 58), (66, 58), (67, 70), (94, 79), (55, 84), (49, 0), (18, 44), (2, 11), (54, 29), (88, 22), (39, 72), (98, 6), (68, 79), (42, 97), (72, 10), (86, 89), (40, 19), (11, 43), (39, 36), (56, 48), (40, 85), (21, 66), (7, 27), (74, 89), (12, 45), (38, 93), (19, 13), (71, 45), (80, 87), (39, 65), (52, 65), (64, 26), (12, 30), (81, 98), (40, 37), (92, 65), (16, 79), (62, 13), (77, 89), (87, 63), (21, 11), (46, 49), (56, 5), (50, 79), (45, 28), (49, 14), (2, 65), (7, 8), (36, 86), (22, 39), (81, 32), (20, 78), (88, 4), (42, 86), (73, 33), (34, 64), (4, 92), (3, 95), (71, 47), (85, 21), (97, 15), (74, 32), (71, 46), (63, 21), (91, 74), (25, 36), (39, 97), (43, 85), (76, 73), (15, 20), (62, 70), (40, 41), (14, 32), (2, 0), (66, 27), (69, 1), (44, 96), (44, 48), (53, 75), (71, 97), (5, 80), (41, 58), (77, 35), (42, 59), (56, 2), (79, 68), (2, 98), (71, 70), (3, 86), (15, 95), (84, 98), (1, 52), (11, 60), (80, 52), (25, 29), (13, 87), (65, 89), (21, 61), (71, 48), (28, 67), (32, 19), (41, 17), (40, 37), (17, 80), (87, 70), (8, 92), (5, 95), (61, 12), (98, 61), (28, 70), (76, 82), (5, 24), (4, 23), (42, 49), (70, 97), (52, 14), (83, 16), (31, 22), (44, 91), (2, 70), (19, 74), (39, 22), (84, 90), (38, 83), (56, 10), (33, 50), (56, 39), (0, 11), (69, 74), (34, 71), (93, 30), (40, 9), (61, 17), (83, 62), (53, 37), (13, 4), (99, 26), (37, 69), (94, 23), (33, 28), (91, 63), (99, 81), (52, 23), (82, 28), (43, 33), (96, 5), (6, 82), (61, 83), (82, 22), (47, 39), (78, 62), (60, 5), (53, 19), (64, 63), (61, 12), (58, 90), (93, 62), (83, 39), (29, 23), (35, 82), (14, 82), (41, 35), (62, 49), (65, 80), (86, 3), (17, 21), (74, 18), (36, 21), (76, 15), (67, 32), (59, 2), (74, 63), (18, 39), (44, 41), (21, 57), (81, 15), (48, 51), (69, 85), (26, 13), (73, 69), (53, 41), (87, 85), (74, 27), (25, 13), (27, 64), (39, 38), (57, 86), (97, 54), (64, 75), (67, 96), (38, 18), (64, 10), (52, 53), (26, 0), (28, 95), (85, 97), (2, 3), (15, 5), (59, 88), (10, 71), (7, 2), (76, 69), (31, 99), (86, 28), (63, 0), (12, 19), (62, 59), (94, 75), (43, 27), (91, 76), (64, 2), (72, 41), (79, 73), (46, 9), (25, 42), (0, 51), (26, 62), (74, 42), (81, 37), (62, 39), (80, 88), (29, 84), (72, 62), (69, 38), (79, 82), (83, 90), (43, 41), (38, 26), (19, 4), (35, 22), (45, 67), (71, 62), (8, 36), (67, 94), (43, 32), (91, 1), (2, 57), (12, 64), (93, 20), (22, 24), (8, 32), (71, 15), (70, 76), (46, 61), (23, 11), (9, 92), (5, 26), (70, 30), (61, 72), (37, 3), (1, 56), (4, 38), (60, 52), (39, 15), (29, 39), (0, 27), (7, 0), (38, 36), (16, 82), (69, 39), (61, 78), (68, 17), (87, 68), (75, 78), (24, 1), (40, 46), (45, 19), (85, 4), (62, 25), (67, 21), (33, 30), (11, 0), (26, 51), (81, 77), (19, 31), (54, 70), (24, 63), (27, 78), (81, 63), (16, 69), (17, 73), (85, 34), (93, 81), (13, 32), (36, 63)]
'''

position_array = []
for node in sorted(G):
    position_array.append(G.nodes[node]['pos'])
distances = squareform(pdist(np.array(position_array)))
for u, v in list_unweighted_edges:
    G.add_edge(u, v, weight = 1)
    #G.add_edge(u, v, weight=np.round(distances[u][v], decimals=1))

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
Battery_Capacity = 1000000             # The initial battery capacity of each node in As
CBR_packet_size = 100                  # Size of data of CBR traffic in bytes
CBR_Period = 1                         # Period of CBR traffic in s
ProcPw = 3                             # processing power level of node (in mW).
RecPw = 20                             # Reception power level of node (in mW).
IdlePw = 20                            # Idle power level of node (in mW).
IdleVol = 3                            # Idle Transceiver voltage (in V)
MaxTxtPower = 500                      # Maximum transmission power of current node (in mW)
MinTxtPower = 250                      # Minimum transmission power of current node (in mW)
Battery_Self_Discharge = 0.025         # 3% is the average self-discharge fcactor per month of Lithium batteries
w_TxT = 0.2                            # weight for average transmission load estimate.
w_Rec = 0.2                            # weight for average reception load estimate.




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
epsilon = 0.2
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
print(Q_matrix)

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






