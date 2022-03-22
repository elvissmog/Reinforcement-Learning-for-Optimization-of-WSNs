import networkx as nx
from pqdict import PQDict
import math
import random
import matplotlib.pyplot as plt

g = nx.Graph()

#xy = [(1, 3), (2.5, 5), (2.5, 1), (4.5, 5), (4.5, 1), (6, 3)]
#list_unweighted_edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]

xy = [(727, 333), (410, 921), (369, 283), (943, 142), (423, 646), (153, 477), (649, 828), (911, 989), (972, 422), (35, 419), (648, 836), (17, 688), (281, 402), (344, 909), (815, 675), (371, 908), (748, 991), (838, 45), (462, 505), (508, 474), (565, 617), (2, 979), (392, 991), (398, 265), (789, 35), (449, 952), (88, 281), (563, 839), (128, 725), (639, 35), (545, 329), (259, 294), (379, 907), (830, 466), (620, 290), (789, 579), (778, 453), (667, 663), (665, 199), (844, 732), (105, 884), (396, 411), (351, 452), (488, 584), (677, 94), (743, 659), (752, 203), (108, 318), (941, 691), (981, 702), (100, 701), (783, 822), (250, 788), (96, 902), (540, 471), (449, 473), (671, 295), (870, 246), (588, 102), (703, 121), (402, 637), (185, 645), (808, 10), (668, 617), (467, 852), (280, 39), (563, 377), (675, 334), (429, 177), (494, 637), (430, 831), (57, 726), (509, 729), (376, 311), (429, 833), (395, 417), (628, 792), (512, 259), (845, 729), (456, 110), (277, 501), (211, 996), (297, 689), (160, 87), (590, 605), (498, 557), (971, 211), (562, 326), (315, 963), (316, 471), (390, 316), (365, 755), (573, 631), (881, 532), (969, 218), (220, 388), (517, 500), (869, 670), (490, 575), (331, 992), (500, 500)]
list_unweighted_edges = [(54, 100), (96, 14), (39, 93), (91, 53), (21, 78), (95, 72), (25, 65), (22, 32), (33, 2), (23, 68), (55, 69), (81, 97), (46, 54), (55, 67), (70, 20), (26, 38), (28, 48), (17, 35), (74, 96), (61, 53), (83, 48), (4, 5), (77, 51), (35, 15), (30, 57), (83, 90), (77, 44), (67, 39), (68, 52), (57, 52), (23, 54), (47, 61), (61, 33), (36, 49), (32, 55), (14, 41), (87, 63), (2, 79), (81, 89), (43, 57), (78, 29), (22, 87), (12, 62), (29, 73), (23, 66), (71, 99), (38, 33), (85, 13), (86, 88), (19, 46), (31, 45), (71, 45), (41, 12), (62, 51), (40, 8), (0, 61), (22, 18), (0, 34), (96, 11), (77, 10), (70, 78), (63, 75), (3, 95), (84, 16), (32, 86), (16, 85), (24, 90), (11, 83), (73, 33), (48, 2), (47, 98), (31, 34), (64, 52), (87, 58), (6, 96), (66, 40), (50, 14), (12, 10), (1, 84), (27, 31), (3, 70), (63, 19), (8, 61), (90, 9), (26, 49), (48, 6), (70, 17), (38, 50), (22, 41), (86, 24), (61, 59), (82, 48), (92, 40), (79, 68), (58, 89), (83, 32), (62, 72), (29, 12), (79, 71), (21, 76), (30, 29), (92, 96), (98, 51), (8, 27), (100, 20), (7, 43), (0, 9), (92, 90), (65, 64), (65, 50), (75, 72), (42, 40), (73, 63), (80, 33), (56, 78), (8, 27), (98, 12), (69, 65), (0, 65), (93, 46), (78, 24), (59, 31), (61, 89), (71, 60), (90, 76), (62, 29), (49, 77), (99, 95), (70, 39), (93, 100), (86, 68), (70, 44), (31, 59), (51, 77), (27, 34), (46, 69), (60, 6), (99, 62), (65, 60), (20, 61), (84, 92), (4, 43), (92, 39), (3, 61), (24, 23), (86, 72), (71, 8), (57, 65), (17, 94), (4, 40), (56, 88), (44, 43), (47, 92), (52, 8), (49, 79), (92, 60), (60, 84), (25, 89), (27, 63), (72, 69), (85, 52), (73, 25), (85, 20), (0, 2), (27, 4), (61, 59), (32, 2), (3, 21), (17, 2), (64, 48), (54, 80), (4, 21), (47, 80), (88, 22), (78, 17), (29, 66), (32, 81), (27, 89), (97, 76), (20, 28), (80, 48), (45, 24), (49, 76), (55, 79), (27, 62), (53, 28), (33, 35), (99, 43), (37, 100), (33, 54), (82, 75), (58, 8), (15, 90), (71, 63), (24, 55), (54, 97), (97, 2), (91, 84), (84, 2), (59, 72), (38, 90), (72, 69), (88, 49), (6, 31), (44, 54), (31, 61), (88, 94), (66, 28), (18, 76), (98, 51), (94, 62), (32, 6), (78, 65), (79, 31), (60, 13), (10, 28), (25, 74), (66, 39), (44, 25), (33, 25), (17, 24), (46, 39), (43, 20), (13, 57), (47, 87), (13, 28), (59, 18), (36, 91), (28, 97), (26, 27), (30, 21), (58, 52), (71, 25), (85, 61), (59, 8), (85, 9), (42, 84), (94, 5), (71, 80), (6, 84), (96, 72), (52, 81), (28, 5), (8, 18), (59, 52), (73, 77), (76, 34), (41, 89), (39, 97)]


for i in range(len(xy)):
	g.add_node(i, pos=xy[i])

position_array = []
for node in sorted(g):
    position_array.append(g.nodes[node]['pos'])
for u, v in list_unweighted_edges:
    distance = math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow(
        (position_array[u][1] - position_array[v][1]), 2))

    g.add_edge(u, v, weight = math.ceil(distance/10))

mst = nx.minimum_spanning_tree(g)
mst_edge = mst.edges()
# Calculating the minimum spanning tree cost of g
mst_edge_cost = []
for tu, tv in mst_edge:
    mst_edge_dis = math.sqrt(math.pow((position_array[tu][0] - position_array[tv][0]), 2) + math.pow(
        (position_array[tu][1] - position_array[tv][1]), 2))
    mst_edge_cost.append(math.ceil(mst_edge_dis/10))

mst_cost = sum(mst_edge_cost)

print('mst_cost:', mst_cost)

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

        h.add_edge(x, y, weight=math.ceil(distance/10))

    return h

MST = []

# Extracting the edges of unique MST from Prims algorithm and storing in MST_edges
MST_edges = []

for i in range(len(g.nodes)):
    y = prim(g,i)
    MST.append(y)
    #print(list(y.edges))
    if set(y.edges()) not in MST_edges:
        MST_edges.append(set(y.edges()))

print('Initial population:', MST_edges)
print('No of Initial population:', len(MST_edges))

# Converting the unique MST edges into graph and storing in unique_MSTs

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

        t.add_edge(u, v, weight=math.ceil(dis/10))
    unique_MSTs.append(t)

#print('unique_MSTs:', unique_MSTs)
#print('No of unique_MSTs:', len(unique_MSTs))


cr = 20   # crossover rate
mr = 20   # mutation rate
ng = 2   # number of generations

num_msts = []
rounds = []
fitness = []

# Generating new MSTs from the unique_MSTs using genetic algorithm

for idx in range(ng):
    ind_pop = []                       # initializing an empty intermidiate population to store unique children
    
    nc = int(100/cr)
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
        sg_array = []
        for node in sorted(sub_graph):
            sg_array.append(sub_graph.nodes[node]['pos'])
        # print(T_edges)
        for (su, sv) in sub_graph_edges:
            dis = math.sqrt(math.pow((position_array[su][0] - position_array[sv][0]), 2) + math.pow(
                (position_array[su][1] - position_array[sv][1]), 2))

            sub_graph.add_edge(su, sv, weight=math.ceil(dis/10))

        new_mst = nx.minimum_spanning_tree(sub_graph)
        #print(list(new_mst.edges))

        if set(new_mst.edges()) not in ind_pop:
            ind_pop.append(set(new_mst.edges()))

    #print('ind_pop:', ind_pop)
    #print(len(ind_pop))
    
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
        su_array = []
        for node in sorted(su_graph):
            su_array.append(su_graph.nodes[node]['pos'])

        for (us, vs) in su_graph_edges:
            dis = math.sqrt(math.pow((position_array[us][0] - position_array[vs][0]), 2) + math.pow(
                (position_array[us][1] - position_array[vs][1]), 2))

            su_graph.add_edge(us, vs, weight=math.ceil(dis)/10)

        cut_set = [value for value in su_graph_edges if value not in sgr_ed]
        #print('cut_set:', cut_set)
        add_edge = random.sample(cut_set, 1)
        #print('add_edge:', add_edge)
        sgr_ed = sgr_ed + add_edge
        #print('sgr ed:', sgr_ed)
        mu_graph = nx.Graph()
        for mu in range(len(xy)):
            su_graph.add_node(mu, pos=xy[mu])
        mu_array = []
        for node in sorted(mu_graph):
            mu_array.append(mu_graph.nodes[node]['pos'])

        for (ms, vu) in sgr_ed:
            dis = math.sqrt(math.pow((position_array[ms][0] - position_array[vu][0]), 2) + math.pow(
                (position_array[ms][1] - position_array[vu][1]), 2))

            mu_graph.add_edge(ms, vu, weight=math.ceil(dis/10))

        if set(mu_graph.edges()) not in ind_pop:
            ind_pop.append(set(mu_graph.edges()))

    #print('ind pop:', ind_pop)
    #print('len(ind_pop):', len(ind_pop))

    # Calculate the fitness of each individual in the intermidate population.
    # Converting each individual in the intermidate population to graph and store in ind_pop_graph
    ind_pop_graphs = []
    for ipg_edges in ind_pop:
        ipg = nx.Graph()
        for pg in range(len(xy)):
            ipg.add_node(pg, pos=xy[pg])
        pg_array = []
        for node in sorted(ipg):
            pg_array.append(ipg.nodes[node]['pos'])
        # print(T_edges)
        for (pu, pv) in ipg_edges:
            dis = math.sqrt(math.pow((position_array[pu][0] - position_array[pv][0]), 2) + math.pow(
                (position_array[pu][1] - position_array[pv][1]), 2))

            ipg.add_edge(pu, pv, weight=math.ceil(dis/10))
        ind_pop_graphs.append(ipg)

    #print('ind_pop_graphs:', ind_pop_graphs)

    # Calculating the spanning tree cost of each individual in Population.
    sum_mst_cost = []
    for pop in ind_pop_graphs:
        pop_edges = set(pop.edges())  # Extracting the spanning tree graph edges
        #print(pop_edges)
        pop_Spanning_Tree_Edge_Distances = []
        for up, vp in pop_edges:
            distance_edge = math.sqrt(math.pow((position_array[up][0] - position_array[vp][0]), 2) + math.pow(
                (position_array[up][1] - position_array[vp][1]), 2))
            pop_Spanning_Tree_Edge_Distances.append(math.ceil(distance_edge/10))
        pop_Tree_Cost = sum(pop_Spanning_Tree_Edge_Distances)
        sum_mst_cost.append(pop_Tree_Cost)
        #print("pop spanning tree cost is:", pop_Tree_Cost)
        if pop_Tree_Cost <= mst_cost:
            if pop_edges not in MST_edges:
                MST_edges.append(pop_edges)

    print('MST_edges after fitness:', MST_edges)
    print('len_MST_edges after fitness:', len(MST_edges))

    fitness.append(((len(sum_mst_cost))*mst_cost)/sum(sum_mst_cost))
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

            us.add_edge(su, sv, weight=math.ceil(dis/10))
        unique_sol.append(us)

    final_pop = []
    for uni_pop in unique_sol:
        if set(uni_pop.edges()) not in final_pop:
            final_pop.append(set(uni_pop.edges()))

    print('length_unique_sol:', len(unique_sol))
    print('Final population:', final_pop)
    print('No of final population:', len(final_pop))
    MST_edges = final_pop
    unique_MSTs = unique_sol
    num_msts.append(len(unique_MSTs))
    rounds.append(idx)

'''
plt.plot(rounds, num_msts)
plt.xlabel('Number of Generations')
plt.ylabel('Number of MSTs')
plt.show()


plt.plot(rounds, fitness)
plt.xlabel('Number of Generations')
plt.ylabel('fitness')
plt.show()
'''