import networkx as nx
from collections import OrderedDict
from sortedcontainers import SortedSet
import numpy as np
import json
import seaborn as sns
import math
import time

sns.set()

txr = 1


# methods to ensure proper graph structure for both Yamada and Substitute classes
def is_weighted(graph):

    for edge in graph.edges():
        edge_data = graph.get_edge_data(*edge)
        # print(edge_data)
        try:
            edge_data['weight']
        except KeyError:
            return False
    return True


def has_self_cycles(graph):

    edges = graph.edges()
    # print(edges)
    for node in graph.nodes():
        if (node, node) in edges:
            return True
    return False


def check_input_graph(graph):
    """
    Ensure a graph is weighted, has no self-cycles, and is connected.
    """
    if not nx.is_connected(graph):
        raise ValueError("Input graph must be a connected.")
    if has_self_cycles(graph):
        raise ValueError("Input graph must have no self-cycles.")
    if not is_weighted(graph):
        raise ValueError("Input graph must have weighted edges.")


def is_tree_of_graph(child, parent):

    parent_edges = parent.edges()
    for child_edge in child.edges():
        if child_edge not in parent_edges:
            return False
    return nx.is_tree(child)


def check_input_tree(tree, parent_graph):

    check_input_graph(tree)
    if not is_tree_of_graph(tree, parent_graph):
        raise ValueError("Input tree is not a spanning tree.")


class Yamada(object):


    def __init__(self, graph, n_trees=np.inf):

        self.instantiate_graph(graph)
        self.trees = []  # minimum spanning trees of graph
        self.n_trees = n_trees

    def instantiate_graph(self, graph):

        check_input_graph(graph)
        self.graph = graph

    def replace_edge(self, tree, old_edge, new_edge):

        new_tree = tree.copy()
        if new_edge in self.graph.edges():
            new_tree.remove_edge(*old_edge)
            weight = self.graph[new_edge[0]][new_edge[1]]['weight']
            new_tree.add_edge(*new_edge, weight=weight)
        else:
            raise ValueError("{} is not contained in parent graph" \
                             .format(new_edge))
        return new_tree

    def spanning_trees(self):

        tree = nx.minimum_spanning_tree(self.graph)
        self.trees.append(tree)
        # if self.n_trees == 1:
        #     return self.trees
        mst_edge_sets = self.new_spanning_trees(tree, set(), set())
        while len(mst_edge_sets) > 0 and len(self.trees) < self.n_trees:
            # container for generated edge sets
            new_edge_sets = []
            for each in mst_edge_sets:
                # ensure number of trees does not exceed threshold
                if len(self.trees) < self.n_trees:
                    # generate new spanning trees and their associated edge sets
                    edge_set = self.new_spanning_trees(each['tree'],
                                                       each['fixed'],
                                                       each['restricted'])
                    # append every newly discovered tree
                    for every in edge_set:
                        new_edge_sets.append(every)

            # re-assign edge sets for looping
            mst_edge_sets = new_edge_sets

        return self.trees

    def new_spanning_trees(self, tree, fixed_edges, restricted_edges):

        # find substitute edges -> step 1 in All_MST2 from Yamada et al. 2010
        step_1 = Substitute(self.graph, tree, fixed_edges, restricted_edges)
        s_edges = step_1.substitute()
        edge_sets = []
        if s_edges is not None and len(self.trees) < self.n_trees:
            for i, edge in enumerate(s_edges):
                if s_edges[edge] is not None:
                    # create new minimum spanning tree with substitute edge
                    new_edge = s_edges[edge]
                    tree_i = self.replace_edge(tree, edge, new_edge)

                    # add new tree to list of minimum spanning trees
                    self.trees.append(tree_i)

                    # update F and R edge sets
                    fixed_i = fixed_edges.union(list(s_edges.keys())[0:i])
                    restricted_i = restricted_edges.union([edge])
                    edge_sets.append({'tree': tree_i,
                                      'fixed': fixed_i,
                                      'restricted': restricted_i})

                    # break tree generation if the number of MSTs exceeds limit
                    if len(self.trees) == self.n_trees:
                        return edge_sets

        return edge_sets


class Substitute(object):


    def __init__(self, graph, tree, fixed_edges, restricted_edges):

        check_input_graph(graph)
        self.graph = graph
        check_input_tree(tree, graph)
        self.tree = tree
        self.fixed_edges = fixed_edges
        self.restricted_edges = restricted_edges
        self.source_node = list(graph.nodes)[-1]
        self.instantiate_substitute_variables()  # step 1 in Substitute

    def instantiate_substitute_variables(self):

        self.directed = self.tree.to_directed()  # directed graph for postorder
        self.postorder_nodes, self.descendants = self.postorder_tree()
        # set Q in original paper
        self.quasi_cuts = SortedSet(key=lambda x: (x[0], x[1], x[2]))

    @staticmethod
    def check_edge_set_membership(edge, edge_set):

        return edge in edge_set or edge[::-1] in edge_set

    def find_incident_edges(self, node):

        incident_set = set()
        for neighbor in nx.neighbors(self.graph, node):
            edge = (node, neighbor)
            restricted = self.check_edge_set_membership(edge,
                                                        self.restricted_edges)
            if not restricted and edge not in self.tree.edges():
                w_edge = (self.graph.get_edge_data(*edge)['weight'], *edge)
                incident_set.add(w_edge)

        return incident_set

    def postorder_tree(self):

        nodes = nx.dfs_postorder_nodes(self.directed, self.source_node)
        postorder_nodes = OrderedDict()

        # map nodes to their postorder position and remove child edges
        child_edges = []
        for i, node in enumerate(nodes):
            postorder_nodes[node] = i + 1
            # remove directed edges not already logged in dictionary
            # --> higher postorder, won't be descendant
            for neighbor in nx.neighbors(self.directed, node):
                if neighbor not in postorder_nodes:
                    # neighbor has higher postorder, remove node -> neighbor
                    # edge, but keep neighbor -> node edge
                    child_edges.append((node, neighbor))
        self.directed.remove_edges_from(child_edges)

        # map nodes to their postordered descendants
        descendants = {}
        for each in postorder_nodes:
            descendants[each] = [postorder_nodes[each]]
            for child in nx.descendants(self.directed, each):
                descendants[each].append(postorder_nodes[child])
            descendants[each].sort()

        return (postorder_nodes, descendants)

    def postordered_edges(self):
        """Return postorded, weighted edges."""
        edges = []
        for u, v in self.tree.edges():
            # ensure edges are orders (u, v) such that u has the lowest
            # postorder
            n1_idx = np.argmin((self.postorder_nodes[u],
                                self.postorder_nodes[v]))
            n2_idx = np.argmax((self.postorder_nodes[u],
                                self.postorder_nodes[v]))
            (n1, n2) = (u, v)[n1_idx], (u, v)[n2_idx]
            w = self.graph.get_edge_data(*(u, v))['weight']
            edges.append((w, n1, n2))

        # order edge list by post order of first node, and then post order of
        # second node
        edges = sorted(edges, key=lambda x: (self.postorder_nodes[x[1]],
                                             self.postorder_nodes[x[2]]))
        return edges

    def equal_weight_descendant(self, weighted_edge):

        weight, node = weighted_edge[0:2]
        for cut_edge in self.quasi_cuts:
            related = self.postorder_nodes[cut_edge[1]] in self.descendants[node]
            if related and cut_edge[0] == weight:
                return (cut_edge)
        return (None)

    def _create_substitute_dict(self, ordered_edges):

        substitute_dict = OrderedDict()
        for e in ordered_edges:
            substitute_dict[e[1:]] = None
        return substitute_dict

    def substitute(self):

        # step 1
        substitute_dict = None

        # step 2
        ordered_edges = self.postordered_edges()
        for n_edge in ordered_edges:
            incident_edges = self.find_incident_edges(n_edge[1])

            # step 2.1
            for i_edge in incident_edges:
                reversed_edge = (i_edge[0], *i_edge[1:][::-1])

                # step 2.1.a
                if self.postorder_nodes[i_edge[2]] < self.descendants[i_edge[1]][0]:
                    if reversed_edge in self.quasi_cuts:
                        self.quasi_cuts.remove(reversed_edge)
                    self.quasi_cuts.add(i_edge)

                # step 2.1.b
                if self.postorder_nodes[i_edge[2]] in self.descendants[i_edge[1]]:
                    if reversed_edge in self.quasi_cuts:
                        self.quasi_cuts.remove(reversed_edge)

                # step 2.1.c
                if self.postorder_nodes[i_edge[2]] > self.descendants[i_edge[1]][-1]:
                    self.quasi_cuts.add(i_edge)

            # step 2.2
            if not self.check_edge_set_membership(n_edge[1:3], self.fixed_edges):

                # step 2.2.a
                cut_edge = self.equal_weight_descendant(n_edge)
                while cut_edge is not None:

                    # step 2.2.b
                    if self.postorder_nodes[cut_edge[2]] in \
                            self.descendants[n_edge[1]]:
                        self.quasi_cuts.remove(cut_edge)

                        # back to step 2.2.a
                        cut_edge = self.equal_weight_descendant(n_edge)

                    # step 2.2.c
                    else:
                        if substitute_dict is None:
                            substitute_dict = self._create_substitute_dict(
                                ordered_edges)

                        substitute_dict[n_edge[1:]] = cut_edge[1:]
                        cut_edge = None

        return (substitute_dict)


# Initializing the graph object
start_time = time.time()
if __name__ == "__main__":

    graph = nx.Graph()

    with open('pos1.txt', 'r') as filehandle:
        pos_list = json.load(filehandle)

    xy = []
    for ps in pos_list:
        xy.append(tuple(ps))

    with open('edges1.txt', 'r') as filehandle:
        edges_list = json.load(filehandle)

    list_unweighted_edges = []
    for ed in edges_list:
        list_unweighted_edges.append(tuple(ed))

    #print('No of edges:', len(list_unweighted_edges))

    for i in range(len(xy)):
        graph.add_node(i, pos=xy[i])

    # Extracting the (x,y) coordinate of each node to enable the calculation of the Euclidean distances of the graph edges
    position_array = []
    for node in sorted(graph):
        position_array.append(graph.nodes[node]['pos'])
    # distances = squareform(pdist(np.array(position_array)))
    for u, v in list_unweighted_edges:
        # graph.add_edge(u, v, weight=np.round(distances[u][v], decimals=1))
        distance = math.sqrt(math.pow((position_array[u][0] - position_array[v][0]), 2) + math.pow(
            (position_array[u][1] - position_array[v][1]), 2))
        # nor_distance = math.ceil(distance/10)
        graph.add_edge(u, v, weight=math.ceil(distance / txr))


    Y = Yamada(graph=graph, n_trees=np.inf)

    MST = Y.spanning_trees()
    print('No of MSTs:', len(MST))
print("--- %s seconds ---" % (time.time() - start_time))

