import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random


# initialization of network parameters
learning_rate = 0.7
initial_energy = 11  # Joules
sink_node_energy = 200000
data_packet_size = 512  # bits
electronic_energy = 50e-9  # Joules/bit 50e-9
e_fs = 10e-12  # Joules/bit/(meter)**2
e_mp = 0.0013e-12 #Joules/bit/(meter)**4
node_energy_limit = 10
epsilon = 0.1
transmission_range = 50
sink_node = 100
num_of_episodes = 5000000

list_unweighted_edges = [(4, 0), (80, 26), (73, 11), (20, 62), (19, 32), (87, 34), (19, 26), (61, 80), (11, 79), (0, 31), (82, 43), (68, 19), (35, 1), (9, 61), (17, 87), (86, 79), (4, 89), (64, 6), (45, 20), (73, 8), (4, 93), (30, 24), (19, 73), (65, 41), (50, 42), (68, 26), (47, 79), (99, 74), (43, 54), (89, 44), (39, 68), (36, 42), (71, 90), (26, 50), (66, 64), (51, 22), (62, 88), (5, 40), (85, 33), (18, 8), (86, 77), (5, 98), (95, 59), (52, 73), (74, 85), (29, 68), (43, 37), (3, 14), (36, 14), (78, 98), (75, 52), (55, 11), (24, 86), (5, 73), (49, 1), (70, 53), (76, 64), (90, 30), (88, 65), (43, 68), (25, 64), (48, 97), (49, 61), (28, 81), (20, 25), (55, 82), (7, 85), (44, 74), (17, 36), (71, 1), (39, 59), (24, 52), (29, 78), (73, 91), (71, 85), (21, 43), (34, 25), (1, 57), (98, 55), (51, 93), (54, 2), (26, 6), (54, 83), (98, 37), (23, 81), (12, 0), (0, 11), (30, 81), (16, 82), (17, 8), (93, 68), (72, 24), (58, 28), (93, 74), (14, 71), (45, 82), (16, 62), (35, 80), (33, 77), (31, 41), (83, 8), (39, 69), (0, 27), (38, 17), (4, 65), (31, 90), (2, 5), (94, 42), (81, 90), (31, 2), (94, 81), (97, 41), (100, 99), (42, 88), (91, 44), (49, 18), (18, 51), (96, 40), (81, 75), (5, 23), (71, 9), (87, 6), (57, 71), (87, 81), (18, 41), (32, 62), (69, 22), (10, 68), (8, 24), (53, 35), (24, 66), (92, 35), (68, 94), (49, 98), (38, 74), (88, 96), (84, 80), (84, 18), (82, 13), (84, 32), (26, 8), (47, 84), (85, 27), (9, 8), (18, 81), (9, 97), (82, 36), (20, 82), (50, 15), (46, 5), (75, 57), (0, 57), (77, 26), (5, 70), (64, 43), (55, 51), (66, 97), (31, 43), (13, 97), (95, 62), (73, 4), (38, 46), (96, 57), (59, 43), (13, 91), (71, 73), (89, 29), (64, 68), (48, 69), (63, 56), (82, 12), (33, 24), (47, 85), (23, 39), (58, 42), (52, 91), (50, 6), (0, 84), (36, 78), (9, 64), (91, 29), (99, 65), (52, 24), (33, 1), (52, 24), (18, 1), (40, 3), (45, 32), (2, 42), (52, 37), (85, 67), (18, 50), (64, 45), (21, 30), (3, 2), (31, 34), (57, 23), (75, 52), (49, 68), (17, 30), (20, 65), (11, 99), (54, 96), (12, 19), (40, 10), (70, 36), (96, 67), (76, 32), (13, 5), (89, 12), (15, 62), (0, 86), (75, 13), (63, 53), (34, 51), (70, 27), (82, 17), (15, 22), (19, 100), (75, 3), (5, 1), (89, 61), (44, 88), (94, 70), (52, 95), (53, 49), (75, 100), (79, 9), (43, 87), (86, 67), (60, 72), (52, 82), (7, 59), (78, 60), (26, 86), (25, 46), (25, 4), (97, 35), (73, 84), (57, 40), (42, 49), (27, 67), (51, 23), (4, 75), (67, 65), (26, 8), (42, 36), (88, 15), (96, 99), (38, 47), (85, 35), (13, 32), (86, 96), (44, 81), (91, 36), (60, 30), (88, 59), (88, 35), (53, 89), (8, 18), (61, 18), (3, 25), (90, 9), (54, 8), (100, 66), (67, 44), (69, 43), (54, 38), (65, 53), (8, 50), (38, 90), (98, 11), (67, 21), (36, 26), (64, 65), (79, 3), (41, 35), (30, 5), (8, 69), (16, 81), (0, 90), (95, 12), (63, 73), (1, 61), (1, 26), (33, 64), (30, 90), (29, 66), (100, 84), (19, 87), (52, 99), (81, 49), (83, 14), (55, 14), (27, 36), (70, 75), (100, 72), (6, 65), (19, 25), (7, 59), (62, 46), (15, 84), (30, 34), (82, 14), (90, 93), (97, 50), (86, 62), (69, 35), (100, 53), (66, 85), (80, 13), (88, 21), (10, 93), (13, 55), (88, 29), (48, 40), (58, 73), (0, 72), (12, 15), (96, 43), (98, 0), (91, 68), (38, 54), (50, 78), (85, 86), (13, 88), (46, 22), (25, 27), (47, 25), (10, 87), (100, 53), (38, 42), (54, 11), (92, 75), (96, 42), (5, 2), (61, 99), (36, 99), (50, 68), (24, 80), (15, 23), (52, 97), (33, 29), (0, 14), (18, 4), (45, 77), (94, 71), (66, 83), (42, 81), (91, 66), (13, 9), (79, 69), (52, 74), (47, 65), (41, 59), (14, 67), (32, 18), (14, 100), (11, 3), (1, 21), (29, 67), (91, 66), (8, 74), (71, 12), (28, 9), (41, 7), (45, 22), (93, 34), (48, 54), (89, 55), (80, 52), (16, 46), (20, 78), (59, 57), (32, 22), (42, 73), (14, 3), (0, 34), (9, 96), (47, 94), (30, 10), (89, 81), (37, 48), (96, 48), (88, 77), (59, 64), (71, 91), (98, 50), (79, 98), (55, 17), (45, 79), (97, 92), (37, 4), (84, 73), (53, 26), (56, 41), (20, 25), (9, 26), (56, 91), (74, 42), (24, 98), (0, 3), (16, 96), (65, 85), (61, 52), (95, 59), (66, 32), (0, 98), (2, 18), (4, 94), (57, 73), (6, 69), (82, 34), (11, 82), (27, 6), (94, 1), (2, 44), (1, 86), (56, 1), (43, 26), (89, 46), (63, 28), (31, 8), (16, 8), (36, 18), (16, 28), (74, 51), (3, 65), (90, 57), (3, 13), (50, 66), (83, 97), (7, 18), (26, 29), (51, 22), (87, 39), (59, 16), (42, 98), (69, 1), (79, 5), (95, 64), (34, 3), (60, 43), (84, 73), (97, 57), (55, 63), (22, 38), (66, 100), (15, 47), (26, 99), (76, 90), (33, 34), (22, 16), (29, 82), (14, 80), (47, 70), (3, 90), (43, 97), (99, 13), (83, 26), (56, 8), (87, 100), (83, 24), (97, 37), (34, 39), (35, 75), (92, 37), (67, 65), (21, 65), (59, 87), (32, 5), (39, 45), (17, 0), (88, 16), (83, 25), (80, 52), (76, 48), (76, 15), (93, 6), (49, 16), (3, 45), (66, 42), (32, 43), (84, 11), (90, 35), (80, 34), (12, 25), (78, 72), (4, 57), (59, 96), (29, 12), (30, 47), (14, 86), (18, 42), (46, 79), (80, 31), (22, 2), (22, 12), (41, 26), (81, 65), (25, 81), (7, 49), (54, 11), (6, 43), (80, 15), (44, 20), (19, 32), (49, 19), (25, 85), (75, 38), (100, 42), (29, 77), (41, 11), (85, 24), (91, 62), (48, 90), (84, 41), (69, 37), (50, 3), (79, 13), (96, 69), (73, 86), (80, 88), (48, 71), (78, 90), (92, 94), (5, 13), (72, 46), (5, 87), (5, 56), (32, 46), (1, 4), (31, 20), (22, 94), (76, 43), (73, 66), (19, 97), (56, 77), (70, 41), (18, 25), (35, 100), (71, 6), (67, 38), (59, 74), (90, 51), (51, 22), (0, 73), (34, 98), (79, 6), (59, 88), (1, 19), (88, 21), (83, 2), (6, 45), (40, 96), (67, 77), (82, 29), (58, 31), (20, 71), (42, 10), (82, 62), (9, 86), (22, 43), (41, 61), (52, 100), (16, 96), (61, 70), (49, 60), (92, 8), (63, 38), (87, 17), (41, 16), (56, 57), (69, 0), (77, 55), (20, 90), (82, 43), (1, 40), (39, 28), (76, 89), (35, 27), (4, 72), (99, 71), (93, 100), (16, 84), (59, 8), (2, 35), (0, 54), (25, 24), (67, 40), (44, 45), (83, 16), (51, 71), (13, 97), (68, 23), (68, 59), (86, 46), (36, 63), (75, 44), (74, 13), (17, 81), (86, 3), (16, 95), (89, 11), (87, 16), (45, 86), (98, 40), (79, 61), (40, 98), (96, 69), (2, 10), (89, 44), (92, 25), (14, 67), (9, 24), (21, 36), (79, 38), (97, 31), (5, 43), (25, 40), (32, 97), (27, 71), (4, 62), (63, 70), (41, 20), (96, 26), (60, 54), (90, 2), (50, 46), (7, 60), (50, 32), (85, 23), (60, 78), (6, 43), (7, 26), (24, 74), (100, 46), (25, 58), (97, 0), (21, 20), (59, 67), (40, 36), (88, 51), (14, 20), (93, 66), (0, 85), (16, 68), (72, 16), (43, 53), (54, 85), (96, 83), (14, 36), (75, 68), (89, 75), (22, 43), (44, 34), (14, 94), (94, 3), (72, 36), (93, 16), (47, 26), (97, 47), (17, 84), (68, 6), (9, 69), (25, 81), (80, 13), (100, 83), (7, 5), (60, 27), (28, 10), (72, 46), (49, 14), (64, 25), (89, 22), (66, 78), (84, 47), (84, 93), (94, 75), (1, 63), (12, 85), (98, 71), (4, 76), (14, 9), (42, 86), (9, 1), (30, 80), (26, 40), (67, 96), (37, 1), (40, 39), (41, 8), (26, 4), (78, 31), (61, 11), (74, 93), (5, 24), (80, 58), (28, 77), (54, 4), (29, 69), (18, 57), (64, 55), (6, 3), (49, 53), (9, 94), (2, 51), (49, 25), (58, 14), (71, 95), (38, 82), (59, 78), (83, 72), (87, 9), (77, 20), (81, 15), (66, 86), (10, 15), (97, 57), (84, 95), (48, 59), (35, 94), (6, 74), (91, 41), (80, 51), (70, 39), (68, 50), (8, 0), (12, 43), (2, 19), (67, 87), (31, 41), (25, 82), (63, 94), (35, 65), (48, 35), (73, 96), (96, 43), (27, 76), (17, 70), (42, 14), (67, 62), (0, 21), (53, 64), (14, 34), (49, 65), (37, 13), (53, 22), (18, 29), (1, 16), (73, 94), (65, 77), (42, 44), (52, 74), (87, 42), (100, 38), (2, 11), (13, 69), (30, 85), (98, 12), (59, 84), (93, 14), (70, 53), (56, 13), (36, 33), (10, 99), (50, 60), (99, 57), (20, 41), (23, 51), (79, 59), (73, 24), (45, 0), (92, 79), (19, 57), (64, 99), (27, 77), (78, 6), (91, 57), (49, 26), (37, 12), (7, 8), (98, 35), (88, 46), (38, 86), (31, 50), (86, 31), (95, 24), (76, 31), (69, 5), (67, 89), (17, 5), (92, 83), (65, 19), (54, 85), (93, 1), (10, 67), (93, 33), (72, 47), (68, 87), (8, 95), (95, 56), (98, 80), (44, 71), (76, 91), (99, 22), (84, 50), (11, 32), (64, 35), (43, 37), (1, 65), (0, 52), (92, 6), (69, 75), (65, 50), (25, 95), (16, 39), (44, 13), (45, 27), (70, 36), (81, 39), (14, 86), (85, 0), (68, 87), (9, 20), (48, 100), (41, 67), (94, 79), (72, 84), (44, 41), (83, 80), (58, 11), (48, 27), (88, 27), (87, 67), (83, 95), (50, 49), (81, 8), (34, 63), (1, 25), (55, 87), (97, 82), (92, 52), (75, 53), (78, 72), (45, 94), (64, 98), (26, 17), (69, 73), (68, 2), (100, 54), (99, 35), (36, 62), (72, 61), (41, 75), (53, 25), (36, 34), (60, 83), (43, 83), (4, 94), (42, 71), (39, 98), (22, 92), (71, 98), (26, 1), (90, 36), (16, 26), (18, 29), (70, 61), (68, 33), (31, 7), (67, 54), (86, 13), (28, 99), (16, 67), (85, 89), (19, 46), (33, 43), (75, 44), (83, 90), (96, 10), (3, 28), (29, 95), (46, 57), (13, 88), (26, 76), (77, 19), (17, 94), (9, 53), (62, 85), (24, 0), (43, 69), (10, 63), (78, 0), (38, 83), (53, 92), (70, 90), (87, 8), (98, 75), (30, 80), (100, 69), (11, 72), (1, 75), (22, 54), (50, 38), (13, 96), (34, 20), (73, 80), (93, 48), (47, 33), (27, 8), (7, 39), (74, 26), (79, 15), (93, 56), (62, 83), (78, 23), (8, 30), (18, 53), (84, 94), (29, 11), (93, 26), (6, 9), (78, 96), (53, 0), (100, 82), (30, 94), (6, 89), (16, 79), (26, 27), (36, 35), (24, 42), (9, 98), (70, 23), (99, 21), (35, 41), (5, 29), (63, 45), (3, 88), (8, 37), (13, 63), (84, 88), (92, 50), (91, 50), (61, 23), (12, 70), (54, 39), (40, 78), (99, 45), (83, 59), (71, 55), (8, 85), (70, 89), (29, 56), (89, 80), (44, 40), (36, 50), (80, 34), (50, 18), (13, 66), (29, 45), (34, 13), (61, 74), (95, 19), (24, 84), (12, 62), (69, 47), (84, 82), (44, 59), (90, 22), (25, 43), (33, 35), (78, 82), (87, 5), (16, 60), (80, 27), (71, 34), (92, 37), (63, 13), (26, 22), (29, 8), (54, 94), (5, 62), (65, 49), (99, 51), (72, 85), (38, 45), (44, 83), (0, 69), (85, 86), (45, 89)]
x_y = [(38, 16), (274, 804), (292, 860), (889, 703), (674, 597), (517, 535), (305, 134), (114, 315), (649, 638), (515, 700), (732, 124), (87, 318), (298, 25), (327, 588), (365, 609), (267, 140), (281, 276), (38, 22), (298, 199), (268, 218), (943, 677), (608, 989), (432, 850), (429, 152), (222, 688), (321, 256), (27, 262), (44, 51), (892, 167), (874, 121), (305, 748), (120, 790), (902, 984), (893, 891), (396, 667), (476, 294), (379, 441), (195, 466), (44, 671), (767, 455), (516, 367), (177, 652), (643, 392), (534, 556), (503, 680), (367, 556), (881, 861), (437, 100), (65, 712), (433, 685), (140, 223), (874, 33), (110, 833), (702, 58), (491, 987), (13, 970), (608, 201), (729, 290), (94, 994), (682, 519), (913, 207), (257, 562), (582, 503), (990, 759), (614, 716), (518, 13), (516, 598), (532, 573), (503, 56), (177, 725), (443, 732), (59, 685), (977, 188), (453, 276), (369, 385), (849, 270), (611, 645), (814, 991), (415, 58), (852, 289), (398, 308), (220, 558), (754, 186), (883, 71), (688, 978), (3, 707), (874, 368), (27, 676), (71, 838), (371, 11), (860, 121), (208, 651), (502, 631), (972, 412), (760, 454), (364, 930), (561, 159), (67, 380), (706, 373), (380, 26), (500, 500)]

xy = {}
for xx in range(len(x_y)):
    xy[xx] = x_y[xx]

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

    Traffic = {0: 5, 1: 3, 2: 5, 3: 5, 4: 5, 5: 2, 6: 5, 7: 2, 8: 3, 9: 2, 10: 3, 11: 5, 12: 5, 13: 3, 14: 4, 15: 1, 16: 3, 17: 2, 18: 2, 19: 3, 20: 5, 21: 5, 22: 2, 23: 2, 24: 1, 25: 4, 26: 2, 27: 2, 28: 2, 29: 5, 30: 4, 31: 4, 32: 1, 33: 1, 34: 5, 35: 3, 36: 1, 37: 4, 38: 4, 39: 3, 40: 2, 41: 4, 42: 2, 43: 1, 44: 1, 45: 1, 46: 2, 47: 2, 48: 1, 49: 3, 50: 5, 51: 2, 52: 1, 53: 3, 54: 1, 55: 5, 56: 3, 57: 4, 58: 3, 59: 5, 60: 5, 61: 3, 62: 3, 63: 1, 64: 1, 65: 4, 66: 4, 67: 5, 68: 3, 69: 2, 70: 4, 71: 5, 72: 1, 73: 5, 74: 1, 75: 5, 76: 4, 77: 5, 78: 4, 79: 4, 80: 2, 81: 5, 82: 1, 83: 5, 84: 4, 85: 1, 86: 1, 87: 4, 88: 1, 89: 1, 90: 2, 91: 5, 92: 4, 93: 3, 94: 4, 95: 3, 96: 3, 97: 5, 98: 4, 99: 3}

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
        #q_vals[ix] = (e_vals[ix] / hop_counts[ix])
        q_vals[ix] = 0

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


    return G, hop_counts, node_neigh, all_q_vals, e_vals, all_path_q_vals, all_path_rwds, Traffic


Av_E_consumed = []
No_Alive_Node = []
round = []

graph, h_counts, node_neighbors, q_values, e_values, path_q_values, all_path_rwds, traffic = build_graph(xy, list_unweighted_edges)


for rdn in range(num_of_episodes):

    E_consumed = []
    EE_consumed = []
    path_f = []


    for node in graph.nodes:
        if node != sink_node:
            start = node
            queue = [start]  # first visited node
            path = str(start)  # first node
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
                    if dis_start_neigh <= d_o:
                        all_path_rwds[node][start][neigh] = e_values[neigh] / (((dis_start_neigh/d_o)**2) * h_counts[neigh])
                        #all_path_rwds[node][start][neigh] = e_values[neigh] / ((dis_start_neigh / d_o) ** 2)
                    else:
                        all_path_rwds[node][start][neigh] = e_values[neigh] / (((dis_start_neigh / d_o) ** 4) * h_counts[neigh])
                        #all_path_rwds[node][start][neigh] = e_values[neigh] / ((dis_start_neigh / d_o) ** 4)

                    temp_qval[neigh] = (1 - learning_rate) * path_q_values[node][start][neigh] + learning_rate * (all_path_rwds[node][start][neigh] + q_values[node][neigh])

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

                #mean_Qvals = sum([q_values[node][k] for k in q_values[node]]) / (len(q_values[node]) * max([q_values[node][k] for k in q_values[node]]))
                dis = math.sqrt(math.pow((xy[start][0] - xy[next_hop][0]), 2) + math.pow((xy[start][1] - xy[next_hop][1]), 2))
                nor_dis = math.ceil(dis / transmission_range)
                if nor_dis <= d_o:
                    etx = electronic_energy * data_packet_size + e_fs * data_packet_size * math.pow(nor_dis, 2)
                else:
                    etx = electronic_energy * data_packet_size + e_mp * data_packet_size * math.pow(nor_dis, 4)
                erx = electronic_energy * data_packet_size
                e_values[start] = e_values[start] - traffic[node]*etx                      # update the start node energy
                e_values[next_hop] = e_values[next_hop] - traffic[node]*erx                # update the next hop energy
                tx_energy += traffic[node]*etx
                rx_energy += traffic[node]*erx
                #initial_delay += dis

                #path = path + "->" + str(next_hop)  # update the path after each visit

                start = next_hop

                if next_hop == sink_node:
                    break

            #delay.append(initial_delay)
            E_consumed.append(tx_energy + rx_energy)
            #mean_Q.append(mean_Qvals)

        #path_f.append(path)
        #Av_mean_Q.append(sum(mean_Q) / len(mean_Q))
        #Av_delay.append(sum(delay) / len(delay))
        Av_E_consumed.append(sum(E_consumed))
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
        print('dead nodes:', dead_node)
        print("The lifetime at this point is", rdn)


        try:
            graph, h_counts, node_neighbors, q_values, e_values, path_q_values, all_path_rwds, traffic = build_graph(xy, list_unweighted_edges)
            #graph, node_neighbors, q_values, e_values, path_q_values, all_path_rwds = build_graph(xy,list_unweighted_edges)


            e_values = update_evals
            #print('hopcounts:', h_counts)
            #print('Updated Evals:', e_values)
            #print('updated_node_neighbours:', node_neighbors)

        except ValueError:

            break

    profit = True

    for nd in graph.nodes:
        if node_neighbors[nd] == [] or len(graph.nodes) == 1:
            profit = False

    if not profit:
        print('lifetime:', rdn)
        break


plt.plot(round, Av_E_consumed)
plt.xlabel('Round')
plt.ylabel('Energy Consumption (Joules)')
plt.title('Energy Consumption for each round')
plt.show()

plt.plot(round, No_Alive_Node)
plt.xlabel('Round')
plt.ylabel('NAN')
plt.title('No of Alive Nodes in each round')
plt.show()






