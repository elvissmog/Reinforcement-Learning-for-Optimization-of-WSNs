import networkx as nx
from pqdict import PQDict
import math
import random
import matplotlib.pyplot as plt
import time
import json

g = nx.Graph()

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

txr = 1

def is_tree_of_graph(child, parent):
    """
    Determine if a potential child graph is a tree of a parent graph.

    Args:
        child (nx.Graph): potential child graph of `parent`.
        parent (nx.Graph): proposed parent graph of `child`.

    Returns:
        (boolean): whether `child` is a tree with all of its edges found in
            `parent`.
    """
    parent_edges = parent.edges()
    for child_edge in child.edges():
        if child_edge not in parent_edges:
            return False
    return nx.is_tree(child)


for i in range(len(xy)):
	g.add_node(i, pos=xy[i])

position_array = []
for node in sorted(g):
    position_array.append(g.nodes[node]['pos'])
for u, v in list_unweighted_edges:
    distance = math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow(
        (position_array[u][1] - position_array[v][1]), 2))

    g.add_edge(u, v, weight = math.ceil(distance/txr))

mst = nx.minimum_spanning_tree(g)
mst_edge = mst.edges()
# Calculating the minimum spanning tree cost of g
mst_edge_cost = []
for tu, tv in mst_edge:
    mst_edge_dis = math.sqrt(math.pow((position_array[tu][0] - position_array[tv][0]), 2) + math.pow(
        (position_array[tu][1] - position_array[tv][1]), 2))
    mst_edge_cost.append(math.ceil(mst_edge_dis/txr))

mst_cost = sum(mst_edge_cost)

#print('mst_cost:', mst_cost)

start_time = time.time()
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

        h.add_edge(x, y, weight=math.ceil(distance/txr))

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

        t.add_edge(u, v, weight=math.ceil(dis/txr))
    unique_MSTs.append(t)


cr = 10   # crossover rate
mr = 10   # mutation rate
ng = 100   # number of generations

num_msts = []
rounds = []
fitness = []
nor_fitness = []

# Generating new MSTs from the unique_MSTs using genetic algorithm

for idx in range(ng):
    ind_pop = []                       # initializing an empty intermidiate population to store unique children
    sum_fitness = []
    nc = int(100/cr)

    # Crossover Operator
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

            sub_graph.add_edge(su, sv, weight=math.ceil(dis/txr))

        new_mst = nx.minimum_spanning_tree(sub_graph)
        #print(list(new_mst.edges))

        if set(new_mst.edges()) not in ind_pop:
            ind_pop.append(set(new_mst.edges()))

    # Mutation Operator
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

            su_graph.add_edge(us, vs, weight=math.ceil(dis)/txr)

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

            mu_graph.add_edge(ms, vu, weight=math.ceil(dis/txr))

        if is_tree_of_graph(mu_graph, g):
            if set(mu_graph.edges()) not in ind_pop:
                ind_pop.append(set(mu_graph.edges()))


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

            ipg.add_edge(pu, pv, weight=math.ceil(dis/txr))
        ind_pop_graphs.append(ipg)

    #print('ind_pop_graphs:', ind_pop_graphs)

    # Calculating the spanning tree cost of each individual in Population.
    sum_mst_cost = []
    for pop in ind_pop_graphs:
        pop_edges = set(pop.edges())  # Extracting the spanning tree graph edges
        #print(pop_edges)
        pop_Spanning_Tree_Edge_Distances = []
        for up, vp in pop_edges:
            dist_edge = math.sqrt(math.pow((position_array[up][0] - position_array[vp][0]), 2) + math.pow(
                (position_array[up][1] - position_array[vp][1]), 2))
            pop_Spanning_Tree_Edge_Distances.append(math.ceil(dist_edge/txr))
        pop_Tree_Cost = sum(pop_Spanning_Tree_Edge_Distances)
        sum_mst_cost.append(pop_Tree_Cost)
        #print("pop spanning tree cost is:", pop_Tree_Cost)
        if pop_Tree_Cost <= mst_cost:
            if pop_edges not in MST_edges:
                MST_edges.append(pop_edges)


    sum_fitness.append(sum(sum_mst_cost))
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

            us.add_edge(su, sv, weight=math.ceil(dis/txr))
        unique_sol.append(us)

    final_pop = []
    for uni_pop in unique_sol:
        if set(uni_pop.edges()) not in final_pop:
            final_pop.append(set(uni_pop.edges()))


    MST_edges = final_pop
    unique_MSTs = unique_sol
    num_msts.append(len(unique_MSTs))
    rounds.append(idx)
    fitness.append(sum(sum_fitness))

print("--- %s seconds ---" % (time.time() - start_time))

'''
for num in num_msts:
    nor_fitness.append(num/max(num_msts))
'''
print('length_unique_sol:', num_msts[-1])

with open('test.txt', 'w') as f:
    f.write(json.dumps(num_msts))

# Now read the file back into a Python list object
with open('test.txt', 'r') as f:
    num_msts = json.loads(f.read())

plt.plot(rounds, num_msts)
plt.xlabel('Number of Generations')
plt.ylabel('Number of MSTs')
plt.show()

'''
plt.plot(rounds, nor_fitness)
plt.xlabel('Number of Generations')
plt.ylabel('Average Normalized Fitness')
plt.show()
'''
