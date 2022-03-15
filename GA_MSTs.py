import networkx as nx
import matplotlib.pyplot as plt
from pqdict import PQDict
import math
import random

g = nx.Graph()

xy = [(1, 3), (2.5, 5), (2.5, 1), (4.5, 5), (4.5, 1), (6, 3)]
list_unweighted_edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]

#xy = [(727, 333), (410, 921), (369, 283), (943, 142), (423, 646), (153, 477), (649, 828), (911, 989), (972, 422), (35, 419), (648, 836), (17, 688), (281, 402), (344, 909), (815, 675), (371, 908), (748, 991), (838, 45), (462, 505), (508, 474), (565, 617), (2, 979), (392, 991), (398, 265), (789, 35), (449, 952), (88, 281), (563, 839), (128, 725), (639, 35), (545, 329), (259, 294), (379, 907), (830, 466), (620, 290), (789, 579), (778, 453), (667, 663), (665, 199), (844, 732), (105, 884), (396, 411), (351, 452), (488, 584), (677, 94), (743, 659), (752, 203), (108, 318), (941, 691), (981, 702), (100, 701), (783, 822), (250, 788), (96, 902), (540, 471), (449, 473), (671, 295), (870, 246), (588, 102), (703, 121), (402, 637), (185, 645), (808, 10), (668, 617), (467, 852), (280, 39), (563, 377), (675, 334), (429, 177), (494, 637), (430, 831), (57, 726), (509, 729), (376, 311), (429, 833), (395, 417), (628, 792), (512, 259), (845, 729), (456, 110), (277, 501), (211, 996), (297, 689), (160, 87), (590, 605), (498, 557), (971, 211), (562, 326), (315, 963), (316, 471), (390, 316), (365, 755), (573, 631), (881, 532), (969, 218), (220, 388), (517, 500), (869, 670), (490, 575), (331, 992), (500, 500)]
#list_unweighted_edges = [(54, 100), (96, 14), (39, 93), (91, 53), (21, 78), (95, 72), (25, 65), (22, 32), (33, 2), (23, 68), (55, 69), (81, 97), (46, 54), (55, 67), (70, 20), (26, 38), (28, 48), (17, 35), (74, 96), (61, 53), (83, 48), (4, 5), (77, 51), (35, 15), (30, 57), (83, 90), (77, 44), (67, 39), (68, 52), (57, 52), (23, 54), (47, 61), (61, 33), (36, 49), (32, 55), (14, 41), (87, 63), (2, 79), (81, 89), (43, 57), (78, 29), (22, 87), (12, 62), (29, 73), (23, 66), (71, 99), (38, 33), (85, 13), (86, 88), (19, 46), (31, 45), (71, 45), (41, 12), (62, 51), (40, 8), (0, 61), (22, 18), (0, 34), (96, 11), (77, 10), (70, 78), (63, 75), (3, 95), (84, 16), (32, 86), (16, 85), (24, 90), (11, 83), (73, 33), (48, 2), (47, 98), (31, 34), (64, 52), (87, 58), (6, 96), (66, 40), (50, 14), (12, 10), (1, 84), (27, 31), (3, 70), (63, 19), (8, 61), (90, 9), (26, 49), (48, 6), (70, 17), (38, 50), (22, 41), (86, 24), (61, 59), (82, 48), (92, 40), (79, 68), (58, 89), (83, 32), (62, 72), (29, 12), (79, 71), (21, 76), (30, 29), (92, 96), (98, 51), (8, 27), (100, 20), (7, 43), (0, 9), (92, 90), (65, 64), (65, 50), (75, 72), (42, 40), (73, 63), (80, 33), (56, 78), (8, 27), (98, 12), (69, 65), (0, 65), (93, 46), (78, 24), (59, 31), (61, 89), (71, 60), (90, 76), (62, 29), (49, 77), (99, 95), (70, 39), (93, 100), (86, 68), (70, 44), (31, 59), (51, 77), (27, 34), (46, 69), (60, 6), (99, 62), (65, 60), (20, 61), (84, 92), (4, 43), (92, 39), (3, 61), (24, 23), (86, 72), (71, 8), (57, 65), (17, 94), (4, 40), (56, 88), (44, 43), (47, 92), (52, 8), (49, 79), (92, 60), (60, 84), (25, 89), (27, 63), (72, 69), (85, 52), (73, 25), (85, 20), (0, 2), (27, 4), (61, 59), (32, 2), (3, 21), (17, 2), (64, 48), (54, 80), (4, 21), (47, 80), (88, 22), (78, 17), (29, 66), (32, 81), (27, 89), (97, 76), (20, 28), (80, 48), (45, 24), (49, 76), (55, 79), (27, 62), (53, 28), (33, 35), (99, 43), (37, 100), (33, 54), (82, 75), (58, 8), (15, 90), (71, 63), (24, 55), (54, 97), (97, 2), (91, 84), (84, 2), (59, 72), (38, 90), (72, 69), (88, 49), (6, 31), (44, 54), (31, 61), (88, 94), (66, 28), (18, 76), (98, 51), (94, 62), (32, 6), (78, 65), (79, 31), (60, 13), (10, 28), (25, 74), (66, 39), (44, 25), (33, 25), (17, 24), (46, 39), (43, 20), (13, 57), (47, 87), (13, 28), (59, 18), (36, 91), (28, 97), (26, 27), (30, 21), (58, 52), (71, 25), (85, 61), (59, 8), (85, 9), (42, 84), (94, 5), (71, 80), (6, 84), (96, 72), (52, 81), (28, 5), (8, 18), (59, 52), (73, 77), (76, 34), (41, 89), (39, 97)]


for i in range(len(xy)):
	g.add_node(i, pos=xy[i])

position_array = []
for node in sorted(g):
    position_array.append(g.nodes[node]['pos'])
for u, v in list_unweighted_edges:
    distance = math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow(
        (position_array[u][1] - position_array[v][1]), 2))

    g.add_edge(u, v, weight = distance)


node_pos = nx.get_node_attributes(g, 'pos')
edge_weight = nx.get_edge_attributes(g, 'weight')

node_col = ['green']
edge_col = ['black']

# Draw the nodes
nx.draw_networkx(g, node_pos, node_color=node_col, node_size=200)
# Draw the node labels
nx.draw_networkx_labels(g, node_pos, node_color=node_col)
# Draw the edges
nx.draw_networkx_edges(g, node_pos, edge_color=edge_col)
nx.draw_networkx_edge_labels(g, node_pos, edge_color=edge_col, edge_labels=edge_weight)
c_t = nx.minimum_edge_cut(g, s=3, t=1)
print('cut_set:', list(c_t))
plt.axis('on')
plt.show()

'''

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

    pos_array = []
    for node in sorted(h):
        pos_array.append(h.nodes[node]['pos'])
    # print(T_edges)
    for (x, y) in mst:
        distance = math.sqrt(math.pow((position_array[x][0] - position_array[y][0]), 2) + math.pow(
            (position_array[x][1] - position_array[y][1]), 2))

        h.add_edge(x, y, weight=distance)

    return h

MST = []

MST_edges = []

for i in range(len(g.nodes)):
    y = prim(g,i)
    MST.append(y)
    #print(list(y.edges))
    if list(y.edges) not in MST_edges:
        MST_edges.append(list(y.edges))

#print(MST_edges)
#print(len(MST_edges))

unique_MSTs = []
for M_edges in MST_edges:
    t = nx.Graph()
    for p in range(len(xy)):
        t.add_node(p, pos=xy[p])
    p_array = []
    for node in sorted(t):
        p_array.append(t.nodes[node]['pos'])
    # print(T_edges)
    for (u, v) in M_edges:
        dis = math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow(
            (position_array[u][1] - position_array[v][1]), 2))

        t.add_edge(u, v, weight=dis)
    unique_MSTs.append(t)

#print('unique_MSTs:', unique_MSTs)
#print('No of unique_MSTs:', len(unique_MSTs))


cr = 25   # crossover rate
mr = 20   # mutation rate
ng = 1   # numner of generations

for idx in range(ng):
    ind_pop = []
    
    nc = int(100/cr)
    for n_c in range(nc):
        two_ind = random.sample(unique_MSTs, 2)

        gr_ed = []
        for gr in two_ind:
            gr_ed.append(list(gr.edges))
        #print(gr_ed)
        sub_graph_edges = list(set(gr_ed[0] + gr_ed[1]))
        #print(sub_graph_edges)
        sub_graph = nx.Graph()
        for sp in range(len(xy)):
            sub_graph.add_node(sp, pos=xy[sp])
        sg_array = []
        for node in sorted(sub_graph):
            sg_array.append(sub_graph.nodes[node]['pos'])
        # print(T_edges)
        for (su, sv) in sub_graph_edges:
            dis = math.sqrt(math.pow((position_array[su][0] - position_array[sv][0]), 2) + math.pow(
                (position_array[su][1] - position_array[sv][1]), 2))

            sub_graph.add_edge(su, sv, weight=dis)

        new_mst = nx.minimum_spanning_tree(sub_graph)
        #print(list(new_mst.edges))
        if list(new_mst.edges) not in ind_pop:
            ind_pop.append(list(new_mst.edges))

    #print(ind_pop)
    #print(len(ind_pop))
    
    nm = int(100 / mr)
    for n_m in range(nm):
        one_ind = random.sample(unique_MSTs, 1)
        sgr_ed = []
        for sgr in one_ind:
            sgr_ed.append(list(sgr.edges))
        #print(sgr_ed)
        one_edge = random.sample(sgr_ed[0], 1)
        #print('edge to delete:', one_edge)
        su_graph_edges = [value for value in list_unweighted_edges if value not in one_edge]
        #print('sub_graph_edges:', su_graph_edges)


'''