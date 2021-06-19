import numpy as np
import networkx as nx
import random
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from collections import Counter
import time

from AllMst import Yamada

G = nx.Graph()
transmission_range = 50

list_unweighted_edges = [(54, 100), (96, 14), (39, 93), (91, 53), (21, 78), (95, 72), (25, 65), (22, 32), (33, 2), (23, 68), (55, 69), (81, 97), (46, 54), (55, 67), (70, 20), (26, 38), (28, 48), (17, 35), (74, 96), (61, 53), (83, 48), (4, 5), (77, 51), (35, 15), (30, 57), (83, 90), (77, 44), (67, 39), (68, 52), (57, 52), (23, 54), (47, 61), (61, 33), (36, 49), (32, 55), (14, 41), (87, 63), (2, 79), (81, 89), (43, 57), (78, 29), (22, 87), (12, 62), (29, 73), (23, 66), (71, 99), (38, 33), (85, 13), (86, 88), (19, 46), (31, 45), (71, 45), (41, 12), (62, 51), (40, 8), (0, 61), (22, 18), (0, 34), (96, 11), (77, 10), (70, 78), (63, 75), (3, 95), (84, 16), (32, 86), (16, 85), (24, 90), (11, 83), (73, 33), (48, 2), (47, 98), (31, 34), (64, 52), (87, 58), (6, 96), (66, 40), (50, 14), (12, 10), (1, 84), (27, 31), (3, 70), (63, 19), (8, 61), (90, 9), (26, 49), (48, 6), (70, 17), (38, 50), (22, 41), (86, 24), (61, 59), (82, 48), (92, 40), (79, 68), (58, 89), (83, 32), (62, 72), (29, 12), (79, 71), (21, 76), (30, 29), (92, 96), (98, 51), (8, 27), (100, 20), (7, 43), (0, 9), (92, 90), (65, 64), (65, 50), (75, 72), (42, 40), (73, 63), (80, 33), (56, 78), (8, 27), (98, 12), (69, 65), (0, 65), (93, 46), (78, 24), (59, 31), (61, 89), (71, 60), (90, 76), (62, 29), (49, 77), (99, 95), (70, 39), (93, 100), (86, 68), (70, 44), (31, 59), (51, 77), (27, 34), (46, 69), (60, 6), (99, 62), (65, 60), (20, 61), (84, 92), (4, 43), (92, 39), (3, 61), (24, 23), (86, 72), (71, 8), (57, 65), (17, 94), (4, 40), (56, 88), (44, 43), (47, 92), (52, 8), (49, 79), (92, 60), (60, 84), (25, 89), (27, 63), (72, 69), (85, 52), (73, 25), (85, 20), (0, 2), (27, 4), (61, 59), (32, 2), (3, 21), (17, 2), (64, 48), (54, 80), (4, 21), (47, 80), (88, 22), (78, 17), (29, 66), (32, 81), (27, 89), (97, 76), (20, 28), (80, 48), (45, 24), (49, 76), (55, 79), (27, 62), (53, 28), (33, 35), (99, 43), (37, 100), (33, 54), (82, 75), (58, 8), (15, 90), (71, 63), (24, 55), (54, 97), (97, 2), (91, 84), (84, 2), (59, 72), (38, 90), (72, 69), (88, 49), (6, 31), (44, 54), (31, 61), (88, 94), (66, 28), (18, 76), (98, 51), (94, 62), (32, 6), (78, 65), (79, 31), (60, 13), (10, 28), (25, 74), (66, 39), (44, 25), (33, 25), (17, 24), (46, 39), (43, 20), (13, 57), (47, 87), (13, 28), (59, 18), (36, 91), (28, 97), (26, 27), (30, 21), (58, 52), (71, 25), (85, 61), (59, 8), (85, 9), (42, 84), (94, 5), (71, 80), (6, 84), (96, 72), (52, 81), (28, 5), (8, 18), (59, 52), (73, 77), (76, 34), (41, 89), (39, 97)]
xy = [(727, 333), (410, 921), (369, 283), (943, 142), (423, 646), (153, 477), (649, 828), (911, 989), (972, 422), (35, 419), (648, 836), (17, 688), (281, 402), (344, 909), (815, 675), (371, 908), (748, 991), (838, 45), (462, 505), (508, 474), (565, 617), (2, 979), (392, 991), (398, 265), (789, 35), (449, 952), (88, 281), (563, 839), (128, 725), (639, 35), (545, 329), (259, 294), (379, 907), (830, 466), (620, 290), (789, 579), (778, 453), (667, 663), (665, 199), (844, 732), (105, 884), (396, 411), (351, 452), (488, 584), (677, 94), (743, 659), (752, 203), (108, 318), (941, 691), (981, 702), (100, 701), (783, 822), (250, 788), (96, 902), (540, 471), (449, 473), (671, 295), (870, 246), (588, 102), (703, 121), (402, 637), (185, 645), (808, 10), (668, 617), (467, 852), (280, 39), (563, 377), (675, 334), (429, 177), (494, 637), (430, 831), (57, 726), (509, 729), (376, 311), (429, 833), (395, 417), (628, 792), (512, 259), (845, 729), (456, 110), (277, 501), (211, 996), (297, 689), (160, 87), (590, 605), (498, 557), (971, 211), (562, 326), (315, 963), (316, 471), (390, 316), (365, 755), (573, 631), (881, 532), (969, 218), (220, 388), (517, 500), (869, 670), (490, 575), (331, 992), (500, 500)]

#xy = [(1, 2), (7, 2), (4, 0), (4, 6)]
#xy = [(1, 3), (2.5, 5), (2.5, 1), (4.5, 5), (4.5, 1), (6, 3)]
#xy = [(14, 82), (10, 19), (80, 34), (54, 8), (66, 40), (1, 12), (24, 69), (56, 78), (57, 76), (38, 91), (1, 77), (77, 35), (96, 89), (0, 64), (23, 72), (49, 52), (79, 39), (39, 48), (56, 45), (63, 3), (15, 13), (80, 99), (57, 86), (9, 54), (97, 25), (17, 11), (70, 38), (92, 80), (94, 90), (5, 36), (9, 89), (18, 91), (80, 17), (41, 25), (66, 78), (21, 66), (90, 4), (64, 71), (8, 61), (89, 84), (70, 10), (83, 84), (62, 41), (22, 71), (9, 70), (23, 91), (56, 54), (72, 49), (80, 98), (75, 32), (46, 70), (65, 99), (91, 96), (85, 100), (82, 87), (92, 87), (13, 45), (28, 18), (25, 64), (41, 29), (93, 32), (58, 73), (45, 84), (4, 59), (31, 52), (40, 28), (51, 79), (2, 60), (71, 100), (17, 37), (21, 35), (31, 32), (71, 76), (89, 47), (50, 42), (40, 23), (92, 21), (53, 21), (76, 53), (95, 88), (72, 91), (93, 66), (19, 26), (83, 85), (0, 62), (84, 2), (4, 39), (41, 44), (70, 81), (19, 12), (94, 90), (57, 61), (99, 2), (94, 69), (46, 97), (22, 19), (38, 20), (90, 73), (48, 21), (50, 50)]
#xy = [(36, 408), (661, 871), (57, 281), (198, 408), (203, 883), (214, 12), (274, 624), (533, 57), (480, 580), (999, 738), (129, 789), (144, 111), (764, 128), (879, 730), (586, 630), (453, 583), (65, 11), (109, 3), (69, 615), (47, 894), (45, 844), (412, 137), (137, 427), (301, 477), (71, 202), (748, 179), (541, 407), (259, 626), (546, 930), (91, 148), (905, 502), (474, 282), (42, 372), (202, 413), (640, 59), (972, 946), (102, 625), (382, 239), (565, 823), (983, 355), (300, 332), (891, 619), (418, 221), (278, 48), (845, 388), (734, 842), (961, 117), (66, 681), (713, 712), (363, 732), (145, 484), (522, 409), (487, 69), (403, 778), (305, 185), (633, 642), (995, 465), (825, 706), (833, 956), (341, 571), (735, 334), (664, 691), (624, 984), (567, 709), (580, 409), (671, 854), (153, 705), (189, 676), (454, 946), (608, 901), (569, 888), (921, 31), (33, 886), (72, 170), (421, 260), (845, 616), (692, 231), (33, 181), (132, 188), (314, 324), (331, 374), (238, 55), (604, 732), (584, 173), (723, 229), (740, 510), (828, 374), (285, 979), (724, 217), (678, 446), (134, 576), (600, 667), (941, 184), (761, 429), (461, 570), (598, 805), (280, 512), (93, 590), (692, 569), (347, 490), (500, 500)]


for i in range(len(xy)):
    G.add_node(i, pos=xy[i])

# Adding unweighted edges to the Graph
#list_unweighted_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
#list_unweighted_edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]
#list_unweighted_edges = [(48, 50), (33, 29), (49, 68), (66, 44), (6, 44), (8, 18), (27, 50), (63, 62), (36, 63), (32, 52), (31, 2), (12, 70), (54, 83), (85, 33), (38, 78), (27, 26), (35, 41), (25, 62), (60, 77), (65, 13), (65, 8), (45, 34), (10, 28), (84, 51), (16, 15), (40, 50), (75, 74), (11, 75), (56, 76), (39, 62), (32, 90), (54, 97), (61, 64), (20, 56), (4, 48), (49, 5), (14, 98), (39, 23), (73, 76), (76, 47), (65, 61), (91, 6), (99, 14), (75, 85), (8, 53), (19, 97), (98, 18), (19, 23), (15, 3), (81, 77), (89, 36), (49, 68), (75, 22), (77, 54), (11, 25), (4, 84), (45, 19), (32, 23), (70, 59), (17, 37), (67, 30), (71, 78), (24, 46), (37, 80), (93, 71), (16, 28), (2, 73), (57, 27), (52, 49), (40, 98), (6, 11), (53, 50), (3, 16), (16, 59), (64, 25), (82, 14), (25, 27), (9, 82), (46, 36), (60, 52), (3, 86), (63, 66), (46, 89), (3, 33), (70, 90), (72, 34), (98, 50), (39, 31), (49, 81), (76, 19), (99, 76), (78, 8), (2, 49), (80, 14), (61, 0), (51, 31), (82, 61), (42, 63), (88, 97), (1, 68), (81, 47), (54, 11), (26, 2), (0, 40), (64, 25), (93, 61), (40, 8), (17, 84), (91, 55), (32, 2), (61, 47), (1, 2), (24, 4), (26, 31), (63, 69), (2, 26), (78, 29), (47, 14), (58, 4), (77, 40), (55, 30), (16, 52), (21, 13), (37, 59), (93, 86), (28, 84), (46, 35), (48, 42), (36, 4), (56, 44), (90, 1), (8, 25), (82, 15), (39, 99), (15, 89), (43, 40), (11, 98), (42, 81), (74, 16), (22, 2), (18, 65), (39, 53), (33, 34), (16, 11), (67, 22), (81, 74), (69, 6), (20, 54), (85, 89), (67, 13), (46, 0), (81, 78), (55, 63), (41, 4), (34, 76), (93, 80), (26, 62), (95, 15), (27, 75), (28, 79), (49, 80), (80, 59), (14, 75), (5, 57), (39, 4), (23, 62), (91, 79), (45, 11), (77, 32), (4, 18), (57, 89), (9, 97), (93, 60), (78, 67), (23, 36), (10, 12), (34, 69), (22, 54), (90, 46), (11, 10), (88, 12), (33, 60), (76, 60), (1, 81), (85, 9), (19, 1), (67, 93), (20, 16), (28, 30), (7, 39), (97, 86), (4, 98), (11, 3), (71, 60), (0, 64), (58, 45), (62, 38), (24, 26), (74, 60), (88, 33), (53, 92), (35, 91), (81, 78), (10, 74), (56, 64), (8, 13), (37, 94), (70, 44), (80, 20), (87, 81), (12, 63), (66, 41), (82, 17), (75, 3), (11, 5), (47, 87), (15, 92), (82, 33), (9, 39), (34, 12), (29, 67), (46, 76), (45, 99), (0, 49), (13, 95), (43, 37), (74, 59), (94, 34), (53, 47), (30, 31), (85, 2), (52, 20), (64, 79), (52, 65), (28, 24), (35, 37), (44, 34), (68, 75), (52, 32), (79, 70), (14, 40), (37, 33), (97, 84), (64, 56), (48, 22), (14, 68), (17, 38), (67, 23), (83, 18), (50, 58), (47, 41), (58, 0), (61, 25), (99, 89), (89, 35), (74, 56), (96, 31), (24, 28), (76, 32), (4, 64), (36, 88), (3, 93), (97, 83), (55, 41), (19, 64), (90, 10), (72, 10), (72, 18), (97, 71), (72, 38), (72, 40), (97, 43), (45, 57), (12, 10), (15, 17), (70, 81), (7, 96), (80, 99), (27, 85), (16, 91), (86, 12), (26, 41), (81, 34), (99, 42), (76, 46), (69, 12), (68, 82), (71, 58), (36, 20), (43, 0), (1, 20), (2, 47), (6, 54), (55, 1), (52, 12), (50, 87), (32, 99), (33, 34), (49, 18), (35, 81), (10, 87), (46, 86), (32, 31), (97, 50), (52, 62), (68, 87), (15, 40), (56, 94), (49, 65), (14, 49), (80, 14), (84, 20), (66, 7), (74, 92), (43, 69), (42, 58), (36, 85), (39, 61), (83, 33), (93, 6), (46, 30), (38, 49), (79, 17), (58, 8), (13, 43), (18, 82), (79, 88), (58, 43), (10, 94), (58, 95), (32, 78), (60, 52), (54, 1), (13, 64), (67, 58), (83, 71), (16, 17), (24, 52), (96, 17), (13, 26), (9, 39), (7, 75), (74, 86), (26, 54), (88, 16), (46, 74), (64, 7), (17, 67), (74, 0), (62, 96), (83, 17), (50, 23), (91, 51), (65, 23), (74, 48), (26, 54), (59, 21), (81, 97), (50, 45), (53, 92), (0, 85), (14, 87), (98, 83), (36, 57), (68, 89), (38, 2), (86, 53), (43, 22), (92, 23), (20, 11), (98, 61), (86, 78), (45, 99), (31, 52), (78, 25), (67, 34), (97, 6), (20, 21), (8, 99), (47, 70), (21, 80), (95, 32), (55, 52), (37, 88), (20, 59), (35, 2), (95, 42), (58, 57), (96, 76), (79, 47), (57, 61), (98, 75), (53, 88), (29, 34), (41, 1), (59, 45), (71, 44), (96, 21), (44, 28), (4, 50), (50, 2), (46, 99), (52, 73), (66, 81), (38, 88), (10, 22), (3, 68), (61, 15), (39, 46), (36, 63), (13, 73), (16, 52), (5, 61), (46, 29), (70, 73), (99, 81), (11, 28), (22, 49), (56, 39), (71, 54), (72, 77), (45, 56), (56, 92), (87, 2), (2, 46), (77, 7), (95, 14), (50, 34), (71, 91), (17, 0), (96, 73), (54, 29), (26, 49), (78, 0), (92, 59), (7, 80), (35, 12), (53, 25), (99, 67), (74, 55), (97, 66), (0, 88), (44, 98), (40, 59), (61, 8), (2, 12), (74, 92), (81, 6), (34, 84), (19, 26), (95, 18), (95, 45), (79, 86), (79, 18), (26, 54), (13, 60), (37, 42), (10, 46), (19, 63), (10, 90), (76, 94), (16, 40), (1, 95), (89, 60), (67, 55), (48, 33), (74, 60), (26, 56), (65, 44), (24, 55), (13, 8), (9, 40), (73, 3), (87, 7), (94, 51), (39, 17), (22, 81), (66, 90), (60, 58), (5, 17), (3, 53), (64, 28), (70, 6), (63, 51), (30, 40), (34, 15), (82, 38), (42, 84), (64, 96), (5, 91), (41, 69), (6, 83), (74, 54), (67, 95), (19, 52), (6, 81), (47, 89), (2, 23), (25, 55), (64, 51), (72, 29), (65, 78), (83, 16), (99, 58), (46, 22), (82, 44), (26, 19), (41, 67), (18, 82), (61, 38), (33, 81), (34, 57), (38, 77), (97, 46), (94, 22), (42, 19), (9, 24), (78, 47), (29, 80), (24, 92), (93, 44), (89, 11), (51, 12), (17, 40), (58, 71), (49, 94), (65, 15), (79, 4), (22, 19), (20, 38), (90, 55), (71, 7), (76, 79), (49, 23), (41, 93), (89, 0), (6, 5), (59, 35), (75, 24), (27, 44), (15, 96), (8, 13), (40, 26), (90, 69), (4, 39), (1, 30), (44, 66), (4, 1), (51, 40), (16, 54), (48, 5), (78, 66), (11, 18), (72, 33), (99, 79), (75, 60), (28, 96), (19, 37), (39, 63), (17, 87), (46, 80), (36, 54), (2, 83), (32, 16), (91, 34), (37, 16), (35, 27), (10, 44), (54, 1), (29, 71), (50, 80), (93, 64), (4, 11), (5, 84), (18, 81), (37, 45), (49, 32), (8, 86), (52, 98), (67, 16), (75, 55), (40, 74), (97, 37), (67, 5), (25, 54), (51, 97), (93, 18), (95, 13), (80, 57), (5, 82), (6, 80), (90, 81), (7, 83), (12, 53), (63, 34), (64, 45), (84, 16), (14, 71), (48, 71), (9, 52), (83, 37), (24, 87), (19, 15), (6, 5), (15, 67), (10, 64), (56, 55), (11, 3), (3, 12), (97, 24), (96, 1), (16, 50), (43, 84), (50, 24), (99, 6), (5, 28), (9, 20), (74, 23), (22, 45), (57, 37), (78, 30), (76, 98), (98, 62), (44, 23), (59, 37), (57, 42), (68, 99), (13, 49), (26, 39), (44, 49), (61, 97), (52, 50), (10, 84), (24, 21), (32, 6), (66, 2), (53, 12), (93, 91), (96, 32), (42, 95), (92, 52), (34, 74), (63, 28), (86, 89), (57, 51), (84, 51), (50, 11), (84, 90), (14, 53), (58, 31), (58, 8), (80, 32), (42, 92), (59, 78), (7, 77), (55, 83), (26, 96), (30, 16), (51, 47), (26, 35), (10, 89), (49, 17), (55, 47), (9, 6), (31, 59), (55, 60), (82, 65), (34, 62), (50, 62), (83, 35), (35, 85), (39, 62), (8, 68), (62, 50), (21, 29), (58, 12), (0, 41), (23, 62), (35, 96), (34, 8), (20, 84), (82, 25), (15, 2), (1, 33), (73, 60), (28, 30), (65, 55), (53, 83), (2, 90), (84, 55), (36, 29), (32, 9), (33, 9), (39, 28), (82, 85), (43, 58), (93, 24), (2, 88), (82, 18), (59, 61), (55, 23), (56, 42), (2, 29), (35, 20), (71, 97), (81, 65), (10, 48), (74, 48), (14, 58), (66, 58), (67, 70), (94, 79), (55, 84), (49, 0), (18, 44), (2, 11), (54, 29), (88, 22), (39, 72), (98, 6), (68, 79), (42, 97), (72, 10), (86, 89), (40, 19), (11, 43), (39, 36), (56, 48), (40, 85), (21, 66), (7, 27), (74, 89), (12, 45), (38, 93), (19, 13), (71, 45), (80, 87), (39, 65), (52, 65), (64, 26), (12, 30), (81, 98), (40, 37), (92, 65), (16, 79), (62, 13), (77, 89), (87, 63), (21, 11), (46, 49), (56, 5), (50, 79), (45, 28), (49, 14), (2, 65), (7, 8), (36, 86), (22, 39), (81, 32), (20, 78), (88, 4), (42, 86), (73, 33), (34, 64), (4, 92), (3, 95), (71, 47), (85, 21), (97, 15), (74, 32), (71, 46), (63, 21), (91, 74), (25, 36), (39, 97), (43, 85), (76, 73), (15, 20), (62, 70), (40, 41), (14, 32), (2, 0), (66, 27), (69, 1), (44, 96), (44, 48), (53, 75), (71, 97), (5, 80), (41, 58), (77, 35), (42, 59), (56, 2), (79, 68), (2, 98), (71, 70), (3, 86), (15, 95), (84, 98), (1, 52), (11, 60), (80, 52), (25, 29), (13, 87), (65, 89), (21, 61), (71, 48), (28, 67), (32, 19), (41, 17), (40, 37), (17, 80), (87, 70), (8, 92), (5, 95), (61, 12), (98, 61), (28, 70), (76, 82), (5, 24), (4, 23), (42, 49), (70, 97), (52, 14), (83, 16), (31, 22), (44, 91), (2, 70), (19, 74), (39, 22), (84, 90), (38, 83), (56, 10), (33, 50), (56, 39), (0, 11), (69, 74), (34, 71), (93, 30), (40, 9), (61, 17), (83, 62), (53, 37), (13, 4), (99, 26), (37, 69), (94, 23), (33, 28), (91, 63), (99, 81), (52, 23), (82, 28), (43, 33), (96, 5), (6, 82), (61, 83), (82, 22), (47, 39), (78, 62), (60, 5), (53, 19), (64, 63), (61, 12), (58, 90), (93, 62), (83, 39), (29, 23), (35, 82), (14, 82), (41, 35), (62, 49), (65, 80), (86, 3), (17, 21), (74, 18), (36, 21), (76, 15), (67, 32), (59, 2), (74, 63), (18, 39), (44, 41), (21, 57), (81, 15), (48, 51), (69, 85), (26, 13), (73, 69), (53, 41), (87, 85), (74, 27), (25, 13), (27, 64), (39, 38), (57, 86), (97, 54), (64, 75), (67, 96), (38, 18), (64, 10), (52, 53), (26, 0), (28, 95), (85, 97), (2, 3), (15, 5), (59, 88), (10, 71), (7, 2), (76, 69), (31, 99), (86, 28), (63, 0), (12, 19), (62, 59), (94, 75), (43, 27), (91, 76), (64, 2), (72, 41), (79, 73), (46, 9), (25, 42), (0, 51), (26, 62), (74, 42), (81, 37), (62, 39), (80, 88), (29, 84), (72, 62), (69, 38), (79, 82), (83, 90), (43, 41), (38, 26), (19, 4), (35, 22), (45, 67), (71, 62), (8, 36), (67, 94), (43, 32), (91, 1), (2, 57), (12, 64), (93, 20), (22, 24), (8, 32), (71, 15), (70, 76), (46, 61), (23, 11), (9, 92), (5, 26), (70, 30), (61, 72), (37, 3), (1, 56), (4, 38), (60, 52), (39, 15), (29, 39), (0, 27), (7, 0), (38, 36), (16, 82), (69, 39), (61, 78), (68, 17), (87, 68), (75, 78), (24, 1), (40, 46), (45, 19), (85, 4), (62, 25), (67, 21), (33, 30), (11, 0), (26, 51), (81, 77), (19, 31), (54, 70), (24, 63), (27, 78), (81, 63), (16, 69), (17, 73), (85, 34), (93, 81), (13, 32), (36, 63)]
#list_unweighted_edges =[(66, 97), (78, 57), (51, 89), (31, 90), (44, 95), (62, 2), (8, 47), (96, 49), (40, 59), (84, 14), (41, 73), (62, 97), (22, 68), (93, 25), (56, 20), (13, 49), (69, 14), (76, 64), (23, 37), (22, 60), (55, 96), (89, 39), (85, 87), (29, 87), (80, 76), (60, 61), (43, 46), (0, 78), (67, 16), (21, 28), (39, 14), (61, 51), (51, 9), (74, 28), (69, 88), (15, 75), (51, 20), (32, 80), (69, 26), (43, 58), (58, 98), (16, 14), (31, 94), (62, 36), (81, 67), (26, 79), (75, 73), (35, 39), (3, 18), (64, 5), (21, 64), (74, 2), (58, 69), (11, 84), (34, 32), (83, 15), (2, 62), (5, 91), (66, 53), (95, 47), (37, 94), (83, 36), (18, 73), (37, 10), (63, 42), (35, 46), (55, 61), (63, 36), (56, 77), (72, 54), (84, 34), (84, 18), (29, 80), (31, 70), (72, 6), (18, 48), (39, 62), (62, 38), (17, 70), (77, 57), (84, 41), (35, 82), (38, 35), (23, 43), (51, 1), (11, 18), (27, 82), (20, 72), (48, 89), (18, 40), (47, 97), (58, 92), (84, 13), (58, 1), (26, 41), (56, 21), (44, 63), (32, 14), (8, 41), (95, 71), (33, 23), (87, 31), (77, 4), (32, 93), (10, 29), (9, 66), (32, 50), (19, 75), (36, 85), (74, 66), (44, 22), (59, 17), (12, 29), (77, 54), (33, 42), (62, 27), (91, 34), (99, 4), (54, 33), (53, 43), (37, 75), (62, 55), (97, 69), (26, 33), (80, 23), (67, 27), (96, 83), (89, 18), (48, 19), (12, 89), (3, 41), (51, 100), (61, 72), (62, 82), (42, 48), (9, 10), (25, 20), (13, 96), (20, 79), (38, 36), (56, 80), (62, 18), (45, 23), (57, 31), (75, 68), (82, 5), (5, 89), (90, 13), (13, 71), (73, 82), (24, 51), (84, 3), (100, 51), (83, 23), (2, 99), (13, 49), (21, 66), (18, 75), (93, 52), (14, 49), (20, 13), (65, 73), (20, 46), (33, 57), (32, 56), (9, 21), (24, 22), (21, 11), (67, 54), (72, 27), (24, 78), (59, 74), (61, 3), (8, 80), (53, 27), (35, 65), (7, 76), (3, 95), (29, 21), (3, 73), (23, 2), (52, 56), (74, 91), (27, 23), (92, 28), (60, 100), (57, 10), (47, 94), (95, 50), (44, 82), (66, 1), (75, 97), (35, 90), (43, 77), (95, 21), (66, 58), (97, 72), (50, 6), (82, 96), (48, 6), (22, 19), (47, 99), (28, 100), (81, 91), (72, 84), (1, 76), (9, 58), (56, 13), (67, 1), (37, 79), (96, 76), (94, 0), (45, 63), (34, 19), (88, 24), (39, 27), (59, 63), (99, 54), (20, 40), (12, 14), (1, 8), (14, 26), (70, 98), (15, 14), (44, 32), (24, 97), (56, 70), (53, 21), (50, 63), (2, 54), (17, 60), (7, 85), (33, 59), (40, 6), (38, 9), (15, 49), (77, 26), (45, 54), (82, 60), (74, 15), (64, 43), (12, 51), (57, 95), (50, 98), (66, 65), (91, 54), (98, 96), (16, 91), (40, 48), (46, 75), (59, 91), (86, 42), (39, 10), (68, 18), (18, 35), (24, 33), (7, 79), (57, 26), (97, 76), (81, 10), (98, 69), (34, 63), (62, 9), (55, 52), (38, 12), (8, 72), (19, 12), (46, 4), (8, 21), (78, 54), (66, 91), (86, 32), (34, 63), (42, 7), (58, 75), (57, 19), (18, 63), (91, 83), (61, 10), (71, 17), (33, 66), (100, 60), (2, 27), (91, 38), (73, 48), (91, 52), (83, 69), (18, 41), (59, 26), (83, 12), (54, 2), (30, 27), (40, 70), (29, 38), (27, 15), (84, 68), (24, 57), (69, 39), (58, 35), (8, 63), (45, 63), (45, 80), (97, 53), (18, 60), (57, 17), (93, 95), (79, 56), (33, 97), (61, 99), (46, 45), (65, 45), (25, 34), (94, 15), (45, 65), (79, 70), (99, 49), (20, 6), (38, 69), (88, 100), (100, 60), (78, 88), (85, 38), (5, 9), (60, 37), (87, 79), (4, 87), (3, 67), (26, 53), (98, 5), (92, 98), (97, 67), (55, 23), (15, 61), (3, 2), (62, 28), (97, 50), (54, 24), (60, 69), (8, 37), (97, 16), (37, 38), (87, 94), (58, 49), (16, 67), (69, 25), (74, 43), (9, 89), (84, 44), (75, 46), (32, 28), (69, 54), (72, 0), (31, 54), (45, 8), (50, 90), (43, 88), (24, 92), (61, 3), (36, 71), (45, 80), (2, 0), (56, 71), (57, 56), (32, 23), (93, 55), (59, 25), (62, 16), (50, 41), (78, 37), (93, 92), (2, 37), (12, 99), (54, 16), (87, 60), (34, 57), (42, 75), (64, 23), (88, 7), (93, 83), (86, 6), (87, 30), (48, 40), (33, 99), (63, 51), (68, 4), (43, 46), (91, 95), (98, 32), (18, 22), (36, 93), (40, 38), (78, 54), (46, 91), (76, 58), (25, 6), (54, 21), (74, 95), (92, 35), (95, 78), (19, 9), (5, 61), (49, 13), (85, 51), (76, 81), (50, 33), (36, 89), (86, 39), (0, 20), (81, 87), (35, 96), (13, 55), (77, 74), (72, 43), (26, 5), (96, 43), (22, 11), (4, 32), (51, 35), (83, 47), (79, 4), (48, 57), (22, 75), (26, 46), (62, 28), (96, 75), (98, 37), (96, 52), (7, 62), (56, 31), (72, 10), (62, 86), (87, 50), (0, 12), (83, 24), (18, 13), (81, 95), (9, 35), (12, 10), (25, 39), (83, 70), (12, 2), (66, 5), (100, 21), (22, 67), (29, 2), (12, 94), (74, 61), (3, 8), (5, 91), (67, 77), (30, 0), (37, 79), (44, 12), (41, 34), (24, 28), (91, 44), (92, 62), (36, 28), (23, 93), (37, 42), (71, 16), (89, 96), (90, 8), (83, 72), (72, 85), (72, 3), (15, 53), (35, 73), (20, 50), (80, 29), (10, 44), (59, 95), (21, 89), (88, 57), (98, 41), (12, 98), (41, 54), (80, 51), (16, 86), (27, 25), (88, 11), (32, 80), (79, 48), (90, 32), (38, 48), (24, 9), (19, 66), (50, 27), (39, 8), (65, 34), (11, 51), (84, 72), (91, 18), (41, 92)]



position_array = []
for node in sorted(G):
    position_array.append(G.nodes[node]['pos'])
distances = squareform(pdist(np.array(position_array)))
#print('dis_max:', distances)
for u, v in list_unweighted_edges:
    #G.add_edge(u, v, weight = 1)
    #G.add_edge(u, v, weight=np.round(distances[u][v], decimals=1))
    distance = math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow(
        (position_array[u][1] - position_array[v][1]), 2))
    nor_distance = math.ceil(distance/transmission_range)
    G.add_edge(u, v, weight=nor_distance)

# initialization of network parameters

discount_factor = 0
learning_rate = 0.7
initial_energy = 20  # Joules
data_packet_size = 512  # bits
control_packet_size = 96 # bits
electronic_energy = 50e-9  # Joules/bit 5
e_fs = 10e-12  # Joules/bit/(meter)**2
e_mp = 0.0013e-12 #Joules/bit/(meter)**4
node_energy_limit = 10  # Joules
learning_period = 10  # seconds

d = [[0 for i in range(len(G))] for j in range(len(G))]
Etx = [[0 for i in range(len(G))] for j in range(len(G))]
Erx = [[0 for i in range(len(G))] for j in range(len(G))]
CEtx = [[0 for i in range(len(G))] for j in range(len(G))]
CErx = [[0 for i in range(len(G))] for j in range(len(G))]
initial_E_vals = [initial_energy for i in range(len(G))]
ref_E_vals = [initial_energy for i in range(len(G))]
epsilon = 0.1
episodes = 5000000

sink_energy = 500000
sink_node = 100

initial_E_vals[sink_node] = sink_energy
ref_E_vals[sink_node] = sink_energy
total_initial_energy = sum(initial_E_vals)

d_o = math.sqrt(e_fs/e_mp)

for i in range(len(G)):
    for j in range(len(G)):
        if i != j:
            d[i][j] = math.ceil((math.sqrt(math.pow((position_array[i][0] - position_array[j][0]), 2) + math.pow(
                (position_array[i][1] - position_array[j][1]), 2)))/transmission_range)

            if d[i][j] <= d_o:
                Etx[i][j] = electronic_energy * data_packet_size + e_fs * data_packet_size * math.pow((d[i][j]), 2)
            else:
                Etx[i][j] = electronic_energy * data_packet_size + e_mp * data_packet_size * math.pow((d[i][j]), 4)
            Erx[i][j] = electronic_energy * data_packet_size
            if d[i][j] <= d_o:
                CEtx[i][j] = electronic_energy * control_packet_size + e_fs * control_packet_size * math.pow((d[i][j]), 2)
            else:
                CEtx[i][j] = electronic_energy * control_packet_size + e_mp * control_packet_size * math.pow((d[i][j]), 4)
            Erx[i][j] = electronic_energy * control_packet_size


d_max = []
for ix in d:
    d_max.append(max(ix))

#print('d_max:', d_max)
#print('d_max_max:', max(d_max))
#print('dist_thre:', d_o)
#print('distance:', d)
#print('Energy:', Etx)

Y = Yamada(graph=G, n_trees = np.inf)
all_STs = Y.spanning_trees()

# Ranking nodes in terms of hop count to sink for each MST
STs_hop_count = []
ST_paths = []
for T in all_STs:
    hop_counts = {}
    ST_path = {}
    for n in T.nodes:
        for path in nx.all_simple_paths(T, source=n, target=sink_node):
            hop_counts[n] = len(path) - 1
            ST_path[n] = path
    hop_counts[sink_node] = 0  # hop count of sink
    STs_hop_count.append(hop_counts)
    ST_paths.append(ST_path)

#print('All paths:', ST_paths)
#print('length all ST:', len(ST_paths))


Q_matrix = np.zeros((len(ST_paths), len(ST_paths)))
initial_state = random.choice(range(0, len(ST_paths), 1))


Q_value = []
Action = []
Min_value = []
Episode = []
E_consumed = []

start_time = time.time()

for epi in range(episodes):
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

    ETX = []
    ERX = []
    for l in range(learning_period):

        tx_energy = 0
        rx_energy = 0

        for node in chosen_ST:
            counter = 0
            while counter < len(chosen_ST[node]) - 1:
                init_node = chosen_ST[node][counter]
                next_node = chosen_ST[node][counter + 1]
                initial_E_vals[init_node] = initial_E_vals[init_node] - Etx[init_node][next_node]  # update the start node energy
                initial_E_vals[next_node] = initial_E_vals[next_node] - Erx[init_node][next_node]  # update the next hop energy
                tx_energy += Etx[init_node][next_node]
                rx_energy += Erx[init_node][next_node]
                counter += 1
            ETX.append(tx_energy)
            ERX.append(rx_energy)


    Energy_Consumption = [ref_E_vals[i] - initial_E_vals[i] for i in G.nodes if i != sink_node]
    reward = max(Energy_Consumption)
    Min_value.append(reward)

    ctx_energy = 0
    crx_energy = 0
    CETX = []
    CERX = []
    for node in chosen_ST:
        counter = 0
        while counter < len(chosen_ST[node]) - 1:
            init_node = chosen_ST[node][counter]
            next_node = chosen_ST[node][counter + 1]
            initial_E_vals[init_node] = initial_E_vals[init_node] - CEtx[init_node][next_node]  # update the start node energy
            initial_E_vals[next_node] = initial_E_vals[next_node] - CErx[init_node][next_node]  # update the next hop energy
            ctx_energy += CEtx[init_node][next_node]
            crx_energy += CErx[init_node][next_node]
            counter += 1
        CETX.append(ctx_energy)
        CERX.append(crx_energy)

    # Maximum possible Q value in next step (for new state)
    max_future_q = np.max(Q_matrix[action, :])

    # Current Q value (for current state and performed action)
    current_q = Q_matrix[current_state, action]

    # And here's our equation for a new Q value for current state and action
    new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)

    Q_value.append(new_q)

    # Update Q table with new Q value
    Q_matrix[current_state, action] = new_q

    E_consumed.append(sum(ETX) + sum(ERX) + sum(CETX) + sum(CERX))
    Episode.append(epi)

    cost = True
    for index, item in enumerate(initial_E_vals):
        if item <= node_energy_limit:
            print('E_vals:', initial_E_vals)
            print('Index:', index)
            cost = False
            print("Energy cannot be negative!")
            print("The final round is", epi)
            print('Lifetime in secs:', epi*learning_period)
            print('Total Energy Consumed:', total_initial_energy - sum(initial_E_vals))

    if not cost:
        break
    #ref_E_vals = initial_E_vals
    for i in range(len(ref_E_vals)):
        ref_E_vals[i] = initial_E_vals[i]


print("--- %s seconds ---" % (time.time() - start_time))
my_data = Counter(Action)
print('RT_UT:', my_data.most_common())  # Returns all unique items and their counts

'''
plt.plot(Episode, Q_value)
plt.xlabel('Round')
plt.ylabel('Discrete Action')
# plt.title('Selected Action for each round')
plt.show()

plt.plot(Episode, E_consumed)
plt.xlabel('Rounds')
plt.ylabel('Total Nodes Energy Consumption (Joules)')
# plt.title('Energy Consumption for each round')
plt.show()
'''

