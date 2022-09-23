import networkx as nx
from pqdict import PQDict
import math
import random
import matplotlib.pyplot as plt
import time
import json

start_time = time.time()
g = nx.Graph()

txr = 120

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


def is_tree_of_graph(child, parent):
    parent_edges = parent.edges()
    for child_edge in child.edges():
        if child_edge not in parent_edges:
            return False
    return nx.is_tree(child)

for i in range(len(xy)):
	g.add_node(i, pos=xy[i])

p = []
for node in sorted(g):
    p.append(g.nodes[node]['pos'])

Trx_dis = []
for u, v in list_unweighted_edges:
    distance = math.sqrt(math.pow((p[u][0] - p[v][0]), 2) + math.pow((p[u][1] - p[v][1]), 2))

    if distance <= txr:
        Trx_dis.append(distance)
        g.add_edge(u, v, weight = 1)

com_range = max(Trx_dis)

print('cm_range:', com_range)

mst = nx.minimum_spanning_tree(g)
mst_edge = mst.edges()
# Calculating the minimum spanning tree cost of g
mst_edge_cost = []
for tu, tv in mst_edge:
    mst_edge_dis = 1
    mst_edge_cost.append(mst_edge_dis)

mst_cost = sum(mst_edge_cost)

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

    for (x, y) in mst:
        distance = 1
        h.add_edge(x, y, weight = distance)

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

    for (u, v) in M_edges:
        dis = 1

        t.add_edge(u, v, weight = dis)
    unique_MSTs.append(t)

print('No of initial Population:', len(unique_MSTs))

print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()

cr = 10   # crossover rate
mr = 10   # mutation rate
ng = 1000   # number of generations

num_msts = []
rounds = []
#fitness = []
#nor_fitness = []

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

        sub_graph_edges = list(set(gr_ed[0] + gr_ed[1]))

        sub_graph = nx.Graph()
        for sp in range(len(xy)):
            sub_graph.add_node(sp, pos=xy[sp])

        for (su, sv) in sub_graph_edges:
            dis = 1

            sub_graph.add_edge(su, sv, weight = dis)

        new_mst = nx.minimum_spanning_tree(sub_graph)


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
        one_edge = random.sample(sgr_ed, 1)
        sgr_ed.remove(one_edge[0])
        su_graph_edges = [value for value in list_unweighted_edges if value not in one_edge]
        su_graph = nx.Graph()
        for su in range(len(xy)):
            su_graph.add_node(su, pos=xy[su])

        for (us, vs) in su_graph_edges:
            dis = 1

            su_graph.add_edge(us, vs, weight = dis)

        cut_set = [value for value in su_graph_edges if value not in sgr_ed]
        add_edge = random.sample(cut_set, 1)
        sgr_ed = sgr_ed + add_edge
        mu_graph = nx.Graph()
        for mu in range(len(xy)):
            su_graph.add_node(mu, pos=xy[mu])

        for (ms, vu) in sgr_ed:
            dis = 1

            mu_graph.add_edge(ms, vu, weight = dis)

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

        for (pu, pv) in ipg_edges:
            dis = 1

            ipg.add_edge(pu, pv, weight = dis)
        ind_pop_graphs.append(ipg)



    # Calculating the spanning tree cost of each individual in Population.
    sum_mst_cost = []
    for pop in ind_pop_graphs:
        pop_edges = set(pop.edges())  # Extracting the spanning tree graph edges
        pop_Spanning_Tree_Edge_Distances = []
        for up, vp in pop_edges:
            dist_edge = 1
            pop_Spanning_Tree_Edge_Distances.append(dist_edge)
        pop_Tree_Cost = sum(pop_Spanning_Tree_Edge_Distances)
        sum_mst_cost.append(pop_Tree_Cost)
        if pop_Tree_Cost <= mst_cost:
            if pop_edges not in MST_edges:
                MST_edges.append(pop_edges)


    #sum_fitness.append(sum(sum_mst_cost))
    unique_sol = []

    for p_edges in MST_edges:
        us = nx.Graph()
        for up in range(len(xy)):
            us.add_node(up, pos=xy[up])
        for (su, sv) in p_edges:
            dis = 1

            us.add_edge(su, sv, weight = dis)
        unique_sol.append(us)

    final_pop = []
    for uni_pop in unique_sol:
        if set(uni_pop.edges()) not in final_pop:
            final_pop.append(set(uni_pop.edges()))

    cost = True
    if len(unique_sol) == len(unique_MSTs):
        cost = False
    if not cost:
        break

    MST_edges = final_pop
    unique_MSTs = unique_sol
    num_msts.append(len(unique_MSTs))
    rounds.append(idx)
    #fitness.append(sum(sum_fitness))

#print("--- %s seconds ---" % (time.time() - start_time))

'''
for num in num_msts:
    nor_fitness.append(num/max(num_msts))

print('length_unique_sol:', num_msts[-1])
'''

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
