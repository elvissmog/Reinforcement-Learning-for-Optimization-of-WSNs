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
sink_node_energy = 5000000
data_packet_size = 1024  # bits
electronic_energy = 50e-9  # Joules/bit 50e-9
e_fs = 10e-12  # Joules/bit/(meter)**2
e_mp = 0.0013e-12 #Joules/bit/(meter)**4
node_energy_limit = 0
epsilon = 0.0
txr = 150
sink_node = 100
num_of_episodes = 5000000

#xy = {0: (1, 3), 1: (2.5, 5), 2: (2.5, 1), 3: (4.5, 5), 4: (4.5, 1), 5: (6, 3)}
#list_unweighted_edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]

#xy2 = [(14, 82), (10, 19), (80, 34), (54, 8), (66, 40), (1, 12), (24, 69), (56, 78), (57, 76), (38, 91), (1, 77), (77, 35), (96, 89), (0, 64), (23, 72), (49, 52), (79, 39), (39, 48), (56, 45), (63, 3), (15, 13), (80, 99), (57, 86), (9, 54), (97, 25), (17, 11), (70, 38), (92, 80), (94, 90), (5, 36), (9, 89), (18, 91), (80, 17), (41, 25), (66, 78), (21, 66), (90, 4), (64, 71), (8, 61), (89, 84), (70, 10), (83, 84), (62, 41), (22, 71), (9, 70), (23, 91), (56, 54), (72, 49), (80, 98), (75, 32), (46, 70), (65, 99), (91, 96), (85, 100), (82, 87), (92, 87), (13, 45), (28, 18), (25, 64), (41, 29), (93, 32), (58, 73), (45, 84), (4, 59), (31, 52), (40, 28), (51, 79), (2, 60), (71, 100), (17, 37), (21, 35), (31, 32), (71, 76), (89, 47), (50, 42), (40, 23), (92, 21), (53, 21), (76, 53), (95, 88), (72, 91), (93, 66), (19, 26), (83, 85), (0, 62), (84, 2), (4, 39), (41, 44), (70, 81), (19, 12), (94, 90), (57, 61), (99, 2), (94, 69), (46, 97), (22, 19), (38, 20), (90, 73), (48, 21), (54, 58)]
#list_unweighted_edges = [(48, 50), (33, 29), (49, 68), (66, 44), (6, 44), (8, 18), (27, 50), (63, 62), (36, 63), (32, 52), (31, 2), (12, 70), (54, 83), (85, 33), (38, 78), (27, 26), (35, 41), (25, 62), (60, 77), (65, 13), (65, 8), (45, 34), (10, 28), (84, 51), (16, 15), (40, 50), (75, 74), (11, 75), (56, 76), (39, 62), (32, 90), (54, 97), (61, 64), (20, 56), (4, 48), (49, 5), (14, 98), (39, 23), (73, 76), (76, 47), (65, 61), (91, 6), (99, 14), (75, 85), (8, 53), (19, 97), (98, 18), (19, 23), (15, 3), (81, 77), (89, 36), (49, 68), (75, 22), (77, 54), (11, 25), (4, 84), (45, 19), (32, 23), (70, 59), (17, 37), (67, 30), (71, 78), (24, 46), (37, 80), (93, 71), (16, 28), (2, 73), (57, 27), (52, 49), (40, 98), (6, 11), (53, 50), (3, 16), (16, 59), (64, 25), (82, 14), (25, 27), (9, 82), (46, 36), (60, 52), (3, 86), (63, 66), (46, 89), (3, 33), (70, 90), (72, 34), (98, 50), (39, 31), (49, 81), (76, 19), (99, 76), (78, 8), (2, 49), (80, 14), (61, 0), (51, 31), (82, 61), (42, 63), (88, 97), (1, 68), (81, 47), (54, 11), (26, 2), (0, 40), (64, 25), (93, 61), (40, 8), (17, 84), (91, 55), (32, 2), (61, 47), (1, 2), (24, 4), (26, 31), (63, 69), (2, 26), (78, 29), (47, 14), (58, 4), (77, 40), (55, 30), (16, 52), (21, 13), (37, 59), (93, 86), (28, 84), (46, 35), (48, 42), (36, 4), (56, 44), (90, 1), (8, 25), (82, 15), (39, 99), (15, 89), (43, 40), (11, 98), (42, 81), (74, 16), (22, 2), (18, 65), (39, 53), (33, 34), (16, 11), (67, 22), (81, 74), (69, 6), (20, 54), (85, 89), (67, 13), (46, 0), (81, 78), (55, 63), (41, 4), (34, 76), (93, 80), (26, 62), (95, 15), (27, 75), (28, 79), (49, 80), (80, 59), (14, 75), (5, 57), (39, 4), (23, 62), (91, 79), (45, 11), (77, 32), (4, 18), (57, 89), (9, 97), (93, 60), (78, 67), (23, 36), (10, 12), (34, 69), (22, 54), (90, 46), (11, 10), (88, 12), (33, 60), (76, 60), (1, 81), (85, 9), (19, 1), (67, 93), (20, 16), (28, 30), (7, 39), (97, 86), (4, 98), (11, 3), (71, 60), (0, 64), (58, 45), (62, 38), (24, 26), (74, 60), (88, 33), (53, 92), (35, 91), (81, 78), (10, 74), (56, 64), (8, 13), (37, 94), (70, 44), (80, 20), (87, 81), (12, 63), (66, 41), (82, 17), (75, 3), (11, 5), (47, 87), (15, 92), (82, 33), (9, 39), (34, 12), (29, 67), (46, 76), (45, 99), (0, 49), (13, 95), (43, 37), (74, 59), (94, 34), (53, 47), (30, 31), (85, 2), (52, 20), (64, 79), (52, 65), (28, 24), (35, 37), (44, 34), (68, 75), (52, 32), (79, 70), (14, 40), (37, 33), (97, 84), (64, 56), (48, 22), (14, 68), (17, 38), (67, 23), (83, 18), (50, 58), (47, 41), (58, 0), (61, 25), (99, 89), (89, 35), (74, 56), (96, 31), (24, 28), (76, 32), (4, 64), (36, 88), (3, 93), (97, 83), (55, 41), (19, 64), (90, 10), (72, 10), (72, 18), (97, 71), (72, 38), (72, 40), (97, 43), (45, 57), (12, 10), (15, 17), (70, 81), (7, 96), (80, 99), (27, 85), (16, 91), (86, 12), (26, 41), (81, 34), (99, 42), (76, 46), (69, 12), (68, 82), (71, 58), (36, 20), (43, 0), (1, 20), (2, 47), (6, 54), (55, 1), (52, 12), (50, 87), (32, 99), (33, 34), (49, 18), (35, 81), (10, 87), (46, 86), (32, 31), (97, 50), (52, 62), (68, 87), (15, 40), (56, 94), (49, 65), (14, 49), (80, 14), (84, 20), (66, 7), (74, 92), (43, 69), (42, 58), (36, 85), (39, 61), (83, 33), (93, 6), (46, 30), (38, 49), (79, 17), (58, 8), (13, 43), (18, 82), (79, 88), (58, 43), (10, 94), (58, 95), (32, 78), (60, 52), (54, 1), (13, 64), (67, 58), (83, 71), (16, 17), (24, 52), (96, 17), (13, 26), (9, 39), (7, 75), (74, 86), (26, 54), (88, 16), (46, 74), (64, 7), (17, 67), (74, 0), (62, 96), (83, 17), (50, 23), (91, 51), (65, 23), (74, 48), (26, 54), (59, 21), (81, 97), (50, 45), (53, 92), (0, 85), (14, 87), (98, 83), (36, 57), (68, 89), (38, 2), (86, 53), (43, 22), (92, 23), (20, 11), (98, 61), (86, 78), (45, 99), (31, 52), (78, 25), (67, 34), (97, 6), (20, 21), (8, 99), (47, 70), (21, 80), (95, 32), (55, 52), (37, 88), (20, 59), (35, 2), (95, 42), (58, 57), (96, 76), (79, 47), (57, 61), (98, 75), (53, 88), (29, 34), (41, 1), (59, 45), (71, 44), (96, 21), (44, 28), (4, 50), (50, 2), (46, 99), (52, 73), (66, 81), (38, 88), (10, 22), (3, 68), (61, 15), (39, 46), (36, 63), (13, 73), (16, 52), (5, 61), (46, 29), (70, 73), (99, 81), (11, 28), (22, 49), (56, 39), (71, 54), (72, 77), (45, 56), (56, 92), (87, 2), (2, 46), (77, 7), (95, 14), (50, 34), (71, 91), (17, 0), (96, 73), (54, 29), (26, 49), (78, 0), (92, 59), (7, 80), (35, 12), (53, 25), (99, 67), (74, 55), (97, 66), (0, 88), (44, 98), (40, 59), (61, 8), (2, 12), (74, 92), (81, 6), (34, 84), (19, 26), (95, 18), (95, 45), (79, 86), (79, 18), (26, 54), (13, 60), (37, 42), (10, 46), (19, 63), (10, 90), (76, 94), (16, 40), (1, 95), (89, 60), (67, 55), (48, 33), (74, 60), (26, 56), (65, 44), (24, 55), (13, 8), (9, 40), (73, 3), (87, 7), (94, 51), (39, 17), (22, 81), (66, 90), (60, 58), (5, 17), (3, 53), (64, 28), (70, 6), (63, 51), (30, 40), (34, 15), (82, 38), (42, 84), (64, 96), (5, 91), (41, 69), (6, 83), (74, 54), (67, 95), (19, 52), (6, 81), (47, 89), (2, 23), (25, 55), (64, 51), (72, 29), (65, 78), (83, 16), (99, 58), (46, 22), (82, 44), (26, 19), (41, 67), (18, 82), (61, 38), (33, 81), (34, 57), (38, 77), (97, 46), (94, 22), (42, 19), (9, 24), (78, 47), (29, 80), (24, 92), (93, 44), (89, 11), (51, 12), (17, 40), (58, 71), (49, 94), (65, 15), (79, 4), (22, 19), (20, 38), (90, 55), (71, 7), (76, 79), (49, 23), (41, 93), (89, 0), (6, 5), (59, 35), (75, 24), (27, 44), (15, 96), (8, 13), (40, 26), (90, 69), (4, 39), (1, 30), (44, 66), (4, 1), (51, 40), (16, 54), (48, 5), (78, 66), (11, 18), (72, 33), (99, 79), (75, 60), (28, 96), (19, 37), (39, 63), (17, 87), (46, 80), (36, 54), (2, 83), (32, 16), (91, 34), (37, 16), (35, 27), (10, 44), (54, 1), (29, 71), (50, 80), (93, 64), (4, 11), (5, 84), (18, 81), (37, 45), (49, 32), (8, 86), (52, 98), (67, 16), (75, 55), (40, 74), (97, 37), (67, 5), (25, 54), (51, 97), (93, 18), (95, 13), (80, 57), (5, 82), (6, 80), (90, 81), (7, 83), (12, 53), (63, 34), (64, 45), (84, 16), (14, 71), (48, 71), (9, 52), (83, 37), (24, 87), (19, 15), (6, 5), (15, 67), (10, 64), (56, 55), (11, 3), (3, 12), (97, 24), (96, 1), (16, 50), (43, 84), (50, 24), (99, 6), (5, 28), (9, 20), (74, 23), (22, 45), (57, 37), (78, 30), (76, 98), (98, 62), (44, 23), (59, 37), (57, 42), (68, 99), (13, 49), (26, 39), (44, 49), (61, 97), (52, 50), (10, 84), (24, 21), (32, 6), (66, 2), (53, 12), (93, 91), (96, 32), (42, 95), (92, 52), (34, 74), (63, 28), (86, 89), (57, 51), (84, 51), (50, 11), (84, 90), (14, 53), (58, 31), (58, 8), (80, 32), (42, 92), (59, 78), (7, 77), (55, 83), (26, 96), (30, 16), (51, 47), (26, 35), (10, 89), (49, 17), (55, 47), (9, 6), (31, 59), (55, 60), (82, 65), (34, 62), (50, 62), (83, 35), (35, 85), (39, 62), (8, 68), (62, 50), (21, 29), (58, 12), (0, 41), (23, 62), (35, 96), (34, 8), (20, 84), (82, 25), (15, 2), (1, 33), (73, 60), (28, 30), (65, 55), (53, 83), (2, 90), (84, 55), (36, 29), (32, 9), (33, 9), (39, 28), (82, 85), (43, 58), (93, 24), (2, 88), (82, 18), (59, 61), (55, 23), (56, 42), (2, 29), (35, 20), (71, 97), (81, 65), (10, 48), (74, 48), (14, 58), (66, 58), (67, 70), (94, 79), (55, 84), (49, 0), (18, 44), (2, 11), (54, 29), (88, 22), (39, 72), (98, 6), (68, 79), (42, 97), (72, 10), (86, 89), (40, 19), (11, 43), (39, 36), (56, 48), (40, 85), (21, 66), (7, 27), (74, 89), (12, 45), (38, 93), (19, 13), (71, 45), (80, 87), (39, 65), (52, 65), (64, 26), (12, 30), (81, 98), (40, 37), (92, 65), (16, 79), (62, 13), (77, 89), (87, 63), (21, 11), (46, 49), (56, 5), (50, 79), (45, 28), (49, 14), (2, 65), (7, 8), (36, 86), (22, 39), (81, 32), (20, 78), (88, 4), (42, 86), (73, 33), (34, 64), (4, 92), (3, 95), (71, 47), (85, 21), (97, 15), (74, 32), (71, 46), (63, 21), (91, 74), (25, 36), (39, 97), (43, 85), (76, 73), (15, 20), (62, 70), (40, 41), (14, 32), (2, 0), (66, 27), (69, 1), (44, 96), (44, 48), (53, 75), (71, 97), (5, 80), (41, 58), (77, 35), (42, 59), (56, 2), (79, 68), (2, 98), (71, 70), (3, 86), (15, 95), (84, 98), (1, 52), (11, 60), (80, 52), (25, 29), (13, 87), (65, 89), (21, 61), (71, 48), (28, 67), (32, 19), (41, 17), (40, 37), (17, 80), (87, 70), (8, 92), (5, 95), (61, 12), (98, 61), (28, 70), (76, 82), (5, 24), (4, 23), (42, 49), (70, 97), (52, 14), (83, 16), (31, 22), (44, 91), (2, 70), (19, 74), (39, 22), (84, 90), (38, 83), (56, 10), (33, 50), (56, 39), (0, 11), (69, 74), (34, 71), (93, 30), (40, 9), (61, 17), (83, 62), (53, 37), (13, 4), (99, 26), (37, 69), (94, 23), (33, 28), (91, 63), (99, 81), (52, 23), (82, 28), (43, 33), (96, 5), (6, 82), (61, 83), (82, 22), (47, 39), (78, 62), (60, 5), (53, 19), (64, 63), (61, 12), (58, 90), (93, 62), (83, 39), (29, 23), (35, 82), (14, 82), (41, 35), (62, 49), (65, 80), (86, 3), (17, 21), (74, 18), (36, 21), (76, 15), (67, 32), (59, 2), (74, 63), (18, 39), (44, 41), (21, 57), (81, 15), (48, 51), (69, 85), (26, 13), (73, 69), (53, 41), (87, 85), (74, 27), (25, 13), (27, 64), (39, 38), (57, 86), (97, 54), (64, 75), (67, 96), (38, 18), (64, 10), (52, 53), (26, 0), (28, 95), (85, 97), (2, 3), (15, 5), (59, 88), (10, 71), (7, 2), (76, 69), (31, 99), (86, 28), (63, 0), (12, 19), (62, 59), (94, 75), (43, 27), (91, 76), (64, 2), (72, 41), (79, 73), (46, 9), (25, 42), (0, 51), (26, 62), (74, 42), (81, 37), (62, 39), (80, 88), (29, 84), (72, 62), (69, 38), (79, 82), (83, 90), (43, 41), (38, 26), (19, 4), (35, 22), (45, 67), (71, 62), (8, 36), (67, 94), (43, 32), (91, 1), (2, 57), (12, 64), (93, 20), (22, 24), (8, 32), (71, 15), (70, 76), (46, 61), (23, 11), (9, 92), (5, 26), (70, 30), (61, 72), (37, 3), (1, 56), (4, 38), (60, 52), (39, 15), (29, 39), (0, 27), (7, 0), (38, 36), (16, 82), (69, 39), (61, 78), (68, 17), (87, 68), (75, 78), (24, 1), (40, 46), (45, 19), (85, 4), (62, 25), (67, 21), (33, 30), (11, 0), (26, 51), (81, 77), (19, 31), (54, 70), (24, 63), (27, 78), (81, 63), (16, 69), (17, 73), (85, 34), (93, 81), (13, 32), (36, 63)]

#list_unweighted_edges = [(54, 100), (96, 14), (39, 93), (91, 53), (21, 78), (95, 72), (25, 65), (22, 32), (33, 2), (23, 68), (55, 69), (81, 97), (46, 54), (55, 67), (70, 20), (26, 38), (28, 48), (17, 35), (74, 96), (61, 53), (83, 48), (4, 5), (77, 51), (35, 15), (30, 57), (83, 90), (77, 44), (67, 39), (68, 52), (57, 52), (23, 54), (47, 61), (61, 33), (36, 49), (32, 55), (14, 41), (87, 63), (2, 79), (81, 89), (43, 57), (78, 29), (22, 87), (12, 62), (29, 73), (23, 66), (71, 99), (38, 33), (85, 13), (86, 88), (19, 46), (31, 45), (71, 45), (41, 12), (62, 51), (40, 8), (0, 61), (22, 18), (0, 34), (96, 11), (77, 10), (70, 78), (63, 75), (3, 95), (84, 16), (32, 86), (16, 85), (24, 90), (11, 83), (73, 33), (48, 2), (47, 98), (31, 34), (64, 52), (87, 58), (6, 96), (66, 40), (50, 14), (12, 10), (1, 84), (27, 31), (3, 70), (63, 19), (8, 61), (90, 9), (26, 49), (48, 6), (70, 17), (38, 50), (22, 41), (86, 24), (61, 59), (82, 48), (92, 40), (79, 68), (58, 89), (83, 32), (62, 72), (29, 12), (79, 71), (21, 76), (30, 29), (92, 96), (98, 51), (8, 27), (100, 20), (7, 43), (0, 9), (92, 90), (65, 64), (65, 50), (75, 72), (42, 40), (73, 63), (80, 33), (56, 78), (8, 27), (98, 12), (69, 65), (0, 65), (93, 46), (78, 24), (59, 31), (61, 89), (71, 60), (90, 76), (62, 29), (49, 77), (99, 95), (70, 39), (93, 100), (86, 68), (70, 44), (31, 59), (51, 77), (27, 34), (46, 69), (60, 6), (99, 62), (65, 60), (20, 61), (84, 92), (4, 43), (92, 39), (3, 61), (24, 23), (86, 72), (71, 8), (57, 65), (17, 94), (4, 40), (56, 88), (44, 43), (47, 92), (52, 8), (49, 79), (92, 60), (60, 84), (25, 89), (27, 63), (72, 69), (85, 52), (73, 25), (85, 20), (0, 2), (27, 4), (61, 59), (32, 2), (3, 21), (17, 2), (64, 48), (54, 80), (4, 21), (47, 80), (88, 22), (78, 17), (29, 66), (32, 81), (27, 89), (97, 76), (20, 28), (80, 48), (45, 24), (49, 76), (55, 79), (27, 62), (53, 28), (33, 35), (99, 43), (37, 100), (33, 54), (82, 75), (58, 8), (15, 90), (71, 63), (24, 55), (54, 97), (97, 2), (91, 84), (84, 2), (59, 72), (38, 90), (72, 69), (88, 49), (6, 31), (44, 54), (31, 61), (88, 94), (66, 28), (18, 76), (98, 51), (94, 62), (32, 6), (78, 65), (79, 31), (60, 13), (10, 28), (25, 74), (66, 39), (44, 25), (33, 25), (17, 24), (46, 39), (43, 20), (13, 57), (47, 87), (13, 28), (59, 18), (36, 91), (28, 97), (26, 27), (30, 21), (58, 52), (71, 25), (85, 61), (59, 8), (85, 9), (42, 84), (94, 5), (71, 80), (6, 84), (96, 72), (52, 81), (28, 5), (8, 18), (59, 52), (73, 77), (76, 34), (41, 89), (39, 97)]
#xy2 = [(727, 333), (410, 921), (369, 283), (943, 142), (423, 646), (153, 477), (649, 828), (911, 989), (972, 422), (35, 419), (648, 836), (17, 688), (281, 402), (344, 909), (815, 675), (371, 908), (748, 991), (838, 45), (462, 505), (508, 474), (565, 617), (2, 979), (392, 991), (398, 265), (789, 35), (449, 952), (88, 281), (563, 839), (128, 725), (639, 35), (545, 329), (259, 294), (379, 907), (830, 466), (620, 290), (789, 579), (778, 453), (667, 663), (665, 199), (844, 732), (105, 884), (396, 411), (351, 452), (488, 584), (677, 94), (743, 659), (752, 203), (108, 318), (941, 691), (981, 702), (100, 701), (783, 822), (250, 788), (96, 902), (540, 471), (449, 473), (671, 295), (870, 246), (588, 102), (703, 121), (402, 637), (185, 645), (808, 10), (668, 617), (467, 852), (280, 39), (563, 377), (675, 334), (429, 177), (494, 637), (430, 831), (57, 726), (509, 729), (376, 311), (429, 833), (395, 417), (628, 792), (512, 259), (845, 729), (456, 110), (277, 501), (211, 996), (297, 689), (160, 87), (590, 605), (498, 557), (971, 211), (562, 326), (315, 963), (316, 471), (390, 316), (365, 755), (573, 631), (881, 532), (969, 218), (220, 388), (517, 500), (869, 670), (490, 575), (331, 992), (500, 500)]

#list_unweighted_edges = [(4, 0), (80, 26), (73, 11), (20, 62), (19, 32), (87, 34), (19, 26), (61, 80), (11, 79), (0, 31), (82, 43), (68, 19), (35, 1), (9, 61), (17, 87), (86, 79), (4, 89), (64, 6), (45, 20), (73, 8), (4, 93), (30, 24), (19, 73), (65, 41), (50, 42), (68, 26), (47, 79), (99, 74), (43, 54), (89, 44), (39, 68), (36, 42), (71, 90), (26, 50), (66, 64), (51, 22), (62, 88), (5, 40), (85, 33), (18, 8), (86, 77), (5, 98), (95, 59), (52, 73), (74, 85), (29, 68), (43, 37), (3, 14), (36, 14), (78, 98), (75, 52), (55, 11), (24, 86), (5, 73), (49, 1), (70, 53), (76, 64), (90, 30), (88, 65), (43, 68), (25, 64), (48, 97), (49, 61), (28, 81), (20, 25), (55, 82), (7, 85), (44, 74), (17, 36), (71, 1), (39, 59), (24, 52), (29, 78), (73, 91), (71, 85), (21, 43), (34, 25), (1, 57), (98, 55), (51, 93), (54, 2), (26, 6), (54, 83), (98, 37), (23, 81), (12, 0), (0, 11), (30, 81), (16, 82), (17, 8), (93, 68), (72, 24), (58, 28), (93, 74), (14, 71), (45, 82), (16, 62), (35, 80), (33, 77), (31, 41), (83, 8), (39, 69), (0, 27), (38, 17), (4, 65), (31, 90), (2, 5), (94, 42), (81, 90), (31, 2), (94, 81), (97, 41), (100, 99), (42, 88), (91, 44), (49, 18), (18, 51), (96, 40), (81, 75), (5, 23), (71, 9), (87, 6), (57, 71), (87, 81), (18, 41), (32, 62), (69, 22), (10, 68), (8, 24), (53, 35), (24, 66), (92, 35), (68, 94), (49, 98), (38, 74), (88, 96), (84, 80), (84, 18), (82, 13), (84, 32), (26, 8), (47, 84), (85, 27), (9, 8), (18, 81), (9, 97), (82, 36), (20, 82), (50, 15), (46, 5), (75, 57), (0, 57), (77, 26), (5, 70), (64, 43), (55, 51), (66, 97), (31, 43), (13, 97), (95, 62), (73, 4), (38, 46), (96, 57), (59, 43), (13, 91), (71, 73), (89, 29), (64, 68), (48, 69), (63, 56), (82, 12), (33, 24), (47, 85), (23, 39), (58, 42), (52, 91), (50, 6), (0, 84), (36, 78), (9, 64), (91, 29), (99, 65), (52, 24), (33, 1), (52, 24), (18, 1), (40, 3), (45, 32), (2, 42), (52, 37), (85, 67), (18, 50), (64, 45), (21, 30), (3, 2), (31, 34), (57, 23), (75, 52), (49, 68), (17, 30), (20, 65), (11, 99), (54, 96), (12, 19), (40, 10), (70, 36), (96, 67), (76, 32), (13, 5), (89, 12), (15, 62), (0, 86), (75, 13), (63, 53), (34, 51), (70, 27), (82, 17), (15, 22), (19, 100), (75, 3), (5, 1), (89, 61), (44, 88), (94, 70), (52, 95), (53, 49), (75, 100), (79, 9), (43, 87), (86, 67), (60, 72), (52, 82), (7, 59), (78, 60), (26, 86), (25, 46), (25, 4), (97, 35), (73, 84), (57, 40), (42, 49), (27, 67), (51, 23), (4, 75), (67, 65), (26, 8), (42, 36), (88, 15), (96, 99), (38, 47), (85, 35), (13, 32), (86, 96), (44, 81), (91, 36), (60, 30), (88, 59), (88, 35), (53, 89), (8, 18), (61, 18), (3, 25), (90, 9), (54, 8), (100, 66), (67, 44), (69, 43), (54, 38), (65, 53), (8, 50), (38, 90), (98, 11), (67, 21), (36, 26), (64, 65), (79, 3), (41, 35), (30, 5), (8, 69), (16, 81), (0, 90), (95, 12), (63, 73), (1, 61), (1, 26), (33, 64), (30, 90), (29, 66), (100, 84), (19, 87), (52, 99), (81, 49), (83, 14), (55, 14), (27, 36), (70, 75), (100, 72), (6, 65), (19, 25), (7, 59), (62, 46), (15, 84), (30, 34), (82, 14), (90, 93), (97, 50), (86, 62), (69, 35), (100, 53), (66, 85), (80, 13), (88, 21), (10, 93), (13, 55), (88, 29), (48, 40), (58, 73), (0, 72), (12, 15), (96, 43), (98, 0), (91, 68), (38, 54), (50, 78), (85, 86), (13, 88), (46, 22), (25, 27), (47, 25), (10, 87), (100, 53), (38, 42), (54, 11), (92, 75), (96, 42), (5, 2), (61, 99), (36, 99), (50, 68), (24, 80), (15, 23), (52, 97), (33, 29), (0, 14), (18, 4), (45, 77), (94, 71), (66, 83), (42, 81), (91, 66), (13, 9), (79, 69), (52, 74), (47, 65), (41, 59), (14, 67), (32, 18), (14, 100), (11, 3), (1, 21), (29, 67), (91, 66), (8, 74), (71, 12), (28, 9), (41, 7), (45, 22), (93, 34), (48, 54), (89, 55), (80, 52), (16, 46), (20, 78), (59, 57), (32, 22), (42, 73), (14, 3), (0, 34), (9, 96), (47, 94), (30, 10), (89, 81), (37, 48), (96, 48), (88, 77), (59, 64), (71, 91), (98, 50), (79, 98), (55, 17), (45, 79), (97, 92), (37, 4), (84, 73), (53, 26), (56, 41), (20, 25), (9, 26), (56, 91), (74, 42), (24, 98), (0, 3), (16, 96), (65, 85), (61, 52), (95, 59), (66, 32), (0, 98), (2, 18), (4, 94), (57, 73), (6, 69), (82, 34), (11, 82), (27, 6), (94, 1), (2, 44), (1, 86), (56, 1), (43, 26), (89, 46), (63, 28), (31, 8), (16, 8), (36, 18), (16, 28), (74, 51), (3, 65), (90, 57), (3, 13), (50, 66), (83, 97), (7, 18), (26, 29), (51, 22), (87, 39), (59, 16), (42, 98), (69, 1), (79, 5), (95, 64), (34, 3), (60, 43), (84, 73), (97, 57), (55, 63), (22, 38), (66, 100), (15, 47), (26, 99), (76, 90), (33, 34), (22, 16), (29, 82), (14, 80), (47, 70), (3, 90), (43, 97), (99, 13), (83, 26), (56, 8), (87, 100), (83, 24), (97, 37), (34, 39), (35, 75), (92, 37), (67, 65), (21, 65), (59, 87), (32, 5), (39, 45), (17, 0), (88, 16), (83, 25), (80, 52), (76, 48), (76, 15), (93, 6), (49, 16), (3, 45), (66, 42), (32, 43), (84, 11), (90, 35), (80, 34), (12, 25), (78, 72), (4, 57), (59, 96), (29, 12), (30, 47), (14, 86), (18, 42), (46, 79), (80, 31), (22, 2), (22, 12), (41, 26), (81, 65), (25, 81), (7, 49), (54, 11), (6, 43), (80, 15), (44, 20), (19, 32), (49, 19), (25, 85), (75, 38), (100, 42), (29, 77), (41, 11), (85, 24), (91, 62), (48, 90), (84, 41), (69, 37), (50, 3), (79, 13), (96, 69), (73, 86), (80, 88), (48, 71), (78, 90), (92, 94), (5, 13), (72, 46), (5, 87), (5, 56), (32, 46), (1, 4), (31, 20), (22, 94), (76, 43), (73, 66), (19, 97), (56, 77), (70, 41), (18, 25), (35, 100), (71, 6), (67, 38), (59, 74), (90, 51), (51, 22), (0, 73), (34, 98), (79, 6), (59, 88), (1, 19), (88, 21), (83, 2), (6, 45), (40, 96), (67, 77), (82, 29), (58, 31), (20, 71), (42, 10), (82, 62), (9, 86), (22, 43), (41, 61), (52, 100), (16, 96), (61, 70), (49, 60), (92, 8), (63, 38), (87, 17), (41, 16), (56, 57), (69, 0), (77, 55), (20, 90), (82, 43), (1, 40), (39, 28), (76, 89), (35, 27), (4, 72), (99, 71), (93, 100), (16, 84), (59, 8), (2, 35), (0, 54), (25, 24), (67, 40), (44, 45), (83, 16), (51, 71), (13, 97), (68, 23), (68, 59), (86, 46), (36, 63), (75, 44), (74, 13), (17, 81), (86, 3), (16, 95), (89, 11), (87, 16), (45, 86), (98, 40), (79, 61), (40, 98), (96, 69), (2, 10), (89, 44), (92, 25), (14, 67), (9, 24), (21, 36), (79, 38), (97, 31), (5, 43), (25, 40), (32, 97), (27, 71), (4, 62), (63, 70), (41, 20), (96, 26), (60, 54), (90, 2), (50, 46), (7, 60), (50, 32), (85, 23), (60, 78), (6, 43), (7, 26), (24, 74), (100, 46), (25, 58), (97, 0), (21, 20), (59, 67), (40, 36), (88, 51), (14, 20), (93, 66), (0, 85), (16, 68), (72, 16), (43, 53), (54, 85), (96, 83), (14, 36), (75, 68), (89, 75), (22, 43), (44, 34), (14, 94), (94, 3), (72, 36), (93, 16), (47, 26), (97, 47), (17, 84), (68, 6), (9, 69), (25, 81), (80, 13), (100, 83), (7, 5), (60, 27), (28, 10), (72, 46), (49, 14), (64, 25), (89, 22), (66, 78), (84, 47), (84, 93), (94, 75), (1, 63), (12, 85), (98, 71), (4, 76), (14, 9), (42, 86), (9, 1), (30, 80), (26, 40), (67, 96), (37, 1), (40, 39), (41, 8), (26, 4), (78, 31), (61, 11), (74, 93), (5, 24), (80, 58), (28, 77), (54, 4), (29, 69), (18, 57), (64, 55), (6, 3), (49, 53), (9, 94), (2, 51), (49, 25), (58, 14), (71, 95), (38, 82), (59, 78), (83, 72), (87, 9), (77, 20), (81, 15), (66, 86), (10, 15), (97, 57), (84, 95), (48, 59), (35, 94), (6, 74), (91, 41), (80, 51), (70, 39), (68, 50), (8, 0), (12, 43), (2, 19), (67, 87), (31, 41), (25, 82), (63, 94), (35, 65), (48, 35), (73, 96), (96, 43), (27, 76), (17, 70), (42, 14), (67, 62), (0, 21), (53, 64), (14, 34), (49, 65), (37, 13), (53, 22), (18, 29), (1, 16), (73, 94), (65, 77), (42, 44), (52, 74), (87, 42), (100, 38), (2, 11), (13, 69), (30, 85), (98, 12), (59, 84), (93, 14), (70, 53), (56, 13), (36, 33), (10, 99), (50, 60), (99, 57), (20, 41), (23, 51), (79, 59), (73, 24), (45, 0), (92, 79), (19, 57), (64, 99), (27, 77), (78, 6), (91, 57), (49, 26), (37, 12), (7, 8), (98, 35), (88, 46), (38, 86), (31, 50), (86, 31), (95, 24), (76, 31), (69, 5), (67, 89), (17, 5), (92, 83), (65, 19), (54, 85), (93, 1), (10, 67), (93, 33), (72, 47), (68, 87), (8, 95), (95, 56), (98, 80), (44, 71), (76, 91), (99, 22), (84, 50), (11, 32), (64, 35), (43, 37), (1, 65), (0, 52), (92, 6), (69, 75), (65, 50), (25, 95), (16, 39), (44, 13), (45, 27), (70, 36), (81, 39), (14, 86), (85, 0), (68, 87), (9, 20), (48, 100), (41, 67), (94, 79), (72, 84), (44, 41), (83, 80), (58, 11), (48, 27), (88, 27), (87, 67), (83, 95), (50, 49), (81, 8), (34, 63), (1, 25), (55, 87), (97, 82), (92, 52), (75, 53), (78, 72), (45, 94), (64, 98), (26, 17), (69, 73), (68, 2), (100, 54), (99, 35), (36, 62), (72, 61), (41, 75), (53, 25), (36, 34), (60, 83), (43, 83), (4, 94), (42, 71), (39, 98), (22, 92), (71, 98), (26, 1), (90, 36), (16, 26), (18, 29), (70, 61), (68, 33), (31, 7), (67, 54), (86, 13), (28, 99), (16, 67), (85, 89), (19, 46), (33, 43), (75, 44), (83, 90), (96, 10), (3, 28), (29, 95), (46, 57), (13, 88), (26, 76), (77, 19), (17, 94), (9, 53), (62, 85), (24, 0), (43, 69), (10, 63), (78, 0), (38, 83), (53, 92), (70, 90), (87, 8), (98, 75), (30, 80), (100, 69), (11, 72), (1, 75), (22, 54), (50, 38), (13, 96), (34, 20), (73, 80), (93, 48), (47, 33), (27, 8), (7, 39), (74, 26), (79, 15), (93, 56), (62, 83), (78, 23), (8, 30), (18, 53), (84, 94), (29, 11), (93, 26), (6, 9), (78, 96), (53, 0), (100, 82), (30, 94), (6, 89), (16, 79), (26, 27), (36, 35), (24, 42), (9, 98), (70, 23), (99, 21), (35, 41), (5, 29), (63, 45), (3, 88), (8, 37), (13, 63), (84, 88), (92, 50), (91, 50), (61, 23), (12, 70), (54, 39), (40, 78), (99, 45), (83, 59), (71, 55), (8, 85), (70, 89), (29, 56), (89, 80), (44, 40), (36, 50), (80, 34), (50, 18), (13, 66), (29, 45), (34, 13), (61, 74), (95, 19), (24, 84), (12, 62), (69, 47), (84, 82), (44, 59), (90, 22), (25, 43), (33, 35), (78, 82), (87, 5), (16, 60), (80, 27), (71, 34), (92, 37), (63, 13), (26, 22), (29, 8), (54, 94), (5, 62), (65, 49), (99, 51), (72, 85), (38, 45), (44, 83), (0, 69), (85, 86), (45, 89)]
#xy2 = [(38, 16), (274, 804), (292, 860), (889, 703), (674, 597), (517, 535), (305, 134), (114, 315), (649, 638), (515, 700), (732, 124), (87, 318), (298, 25), (327, 588), (365, 609), (267, 140), (281, 276), (38, 22), (298, 199), (268, 218), (943, 677), (608, 989), (432, 850), (429, 152), (222, 688), (321, 256), (27, 262), (44, 51), (892, 167), (874, 121), (305, 748), (120, 790), (902, 984), (893, 891), (396, 667), (476, 294), (379, 441), (195, 466), (44, 671), (767, 455), (516, 367), (177, 652), (643, 392), (534, 556), (503, 680), (367, 556), (881, 861), (437, 100), (65, 712), (433, 685), (140, 223), (874, 33), (110, 833), (702, 58), (491, 987), (13, 970), (608, 201), (729, 290), (94, 994), (682, 519), (913, 207), (257, 562), (582, 503), (990, 759), (614, 716), (518, 13), (516, 598), (532, 573), (503, 56), (177, 725), (443, 732), (59, 685), (977, 188), (453, 276), (369, 385), (849, 270), (611, 645), (814, 991), (415, 58), (852, 289), (398, 308), (220, 558), (754, 186), (883, 71), (688, 978), (3, 707), (874, 368), (27, 676), (71, 838), (371, 11), (860, 121), (208, 651), (502, 631), (972, 412), (760, 454), (364, 930), (561, 159), (67, 380), (706, 373), (380, 26), (500, 500)]

with open('pos.txt', 'r') as filehandle:
    pos_list = json.load(filehandle)

x_y = []
for ps in pos_list:
    x_y.append(tuple(ps))

with open('edges.txt', 'r') as filehandle:
    edges_list = json.load(filehandle)

list_unweighted_edges = []
for ed in edges_list:
    list_unweighted_edges.append(tuple(ed))


xy = {}
for xx in range(len(x_y)):
    xy[xx] = x_y[xx]

d_o = math.ceil(math.sqrt(e_fs/e_mp))

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

    Trx_dis = []
    for u, v in links:
        distance = math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow((position_array[u][1] - position_array[v][1]), 2))
        nor_distance = math.ceil(distance)
        if nor_distance <= txr:
            Trx_dis.append(distance)
            G.add_edge(u, v, weight=distance)

    com_range = max(Trx_dis)

    #print('cm_range:', com_range)

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


    return G, hop_counts, node_neigh, all_q_vals, e_vals, all_path_q_vals, all_path_rwds
    #return G, node_neigh, all_q_vals, e_vals, all_path_q_vals, all_path_rwds


Av_mean_Q = []
Av_E_consumed = []
Av_delay = []
No_Alive_Node = []
round = []

graph, h_counts, node_neighbors, q_values, e_values, path_q_values, all_path_rwds = build_graph(xy, list_unweighted_edges)
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
            path = str(start)  # first node
            temp_qval = dict()
            initial_delay = 0
            tx_energy = 0
            rx_energy = 0


            while True:
                for neigh in node_neighbors[start]:
                    dis_start_sink = math.ceil(math.sqrt(math.pow((xy[start][0] - xy[sink_node][0]), 2) + math.pow((xy[start][1] - xy[sink_node][1]), 2)))
                    dis_neigh_sink = math.ceil(math.sqrt(math.pow((xy[neigh][0] - xy[sink_node][0]), 2) + math.pow((xy[neigh][1] - xy[sink_node][1]), 2)))
                    #if dis_start_sink >= dis_neigh_sink and h_counts[start] >= h_counts[neigh]:
                    dis_start_neigh = math.ceil(math.sqrt(math.pow((xy[start][0] - xy[neigh][0]), 2) + math.pow((xy[start][1] - xy[neigh][1]), 2)))
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

                mean_Qvals = sum([q_values[node][k] for k in q_values[node]]) / (len(q_values[node]) * max([q_values[node][k] for k in q_values[node]]))
                dis = math.sqrt(math.pow((xy[start][0] - xy[next_hop][0]), 2) + math.pow((xy[start][1] - xy[next_hop][1]), 2))
                nor_dis = math.ceil(dis)
                if nor_dis <= d_o:
                    etx = electronic_energy * data_packet_size + e_fs * data_packet_size * math.pow(nor_dis, 2)
                else:
                    etx = electronic_energy * data_packet_size + e_mp * data_packet_size * math.pow(nor_dis, 4)
                erx = electronic_energy * data_packet_size
                e_values[start] = e_values[start] - etx                      # update the start node energy
                e_values[next_hop] = e_values[next_hop] - erx                # update the next hop energy
                tx_energy += etx
                rx_energy += erx
                initial_delay += dis

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


        try:
            graph, h_counts, node_neighbors, q_values, e_values, path_q_values, all_path_rwds = build_graph(xy, list_unweighted_edges)
            #graph, node_neighbors, q_values, e_values, path_q_values, all_path_rwds = build_graph(xy,list_unweighted_edges)


            e_values = update_evals
            #print('hopcounts:', h_counts)
            #print('Updated Evals:', e_values)
            #print('updated_node_neighbours:', node_neighbors)

        except (ValueError, IndexError, KeyError):
            print('lifetime:', rdn)

            break

    profit = True

    for nd in graph.nodes:
        if node_neighbors[nd] == [] or len(graph.nodes) == 1:
            profit = False

    if not profit:
        print('lifetime:', rdn)
        break


print("--- %s seconds ---" % (time.time() - start_time))

with open('rlbrnan.txt', 'w') as f:
    f.write(json.dumps(No_Alive_Node))

# Now read the file back into a Python list object
with open('rlbrnan.txt', 'r') as f:
    No_Alive_Node = json.loads(f.read())


plt.plot(round, No_Alive_Node)
plt.xlabel('Round')
plt.ylabel('NAN')
plt.title('Number of Alive Node')
plt.show()

'''
plt.plot(round, Av_mean_Q)
plt.xlabel('Round')
plt.ylabel('Average Q-Value')
plt.title('Q-Value Convergence ')
plt.show()

plt.plot(round, Av_delay)
plt.xlabel('Round')
plt.ylabel('Delay (s)')
plt.title('Delay for each round')
plt.show()

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
'''





