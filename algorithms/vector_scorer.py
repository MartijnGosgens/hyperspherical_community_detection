from .scorers import Scorer
from .clusteringactions import RelabelNode, Aggregate, Flatten
import networkx as nx
import itertools as it
from collections import defaultdict
from copy import deepcopy
import numpy as np


def euclidean_distance(x_vb, x_vv, x_v, x_b, N):
    return (x_vv + x_b - 2 * x_vb) ** 0.5


# Assuming each entry of v in [0,1]
def jaccard_distance(x_vb, x_vv, x_v, x_b, N):
    return 1 - x_vb / ( x_v + x_b - x_vb )


def correlation_distance(x_vb, x_vv, x_v, x_b, N):
    if x_b==0 or x_v==0:
        return 0.5
    return np.arccos(
        (x_vb - x_v * x_b / N) / np.sqrt(
            (x_vv - x_v*x_v/N) * (x_b - x_b*x_b/N)
        )
    ) / np.pi


def angular_distance(x_vb, x_vv, x_v, x_b, N):
    if x_b==0 or x_vv==0:
        return 0.5
    return np.arccos(x_vb / np.sqrt(x_vv * x_b)) / np.pi


def central_cos(x_vb, x_vv, x_v, x_b, N):
    return (N/4+x_vb-x_v/2-x_b/2)/(0.5 * N**0.5 * (N/4-x_v+x_vv)**0.5)


def central_angular_distance(x_vb, x_vv, x_v, x_b, N):
    return np.arccos(central_cos(x_vb, x_vv, x_v, x_b, N)) / np.pi


class VectorScorer(Scorer):
    '''
        constants is a dictionary from attribute-names to their constants.
        G must be undirected
        Each edge of G should have an attribute 's'
        Each node of G should have an attribute for each key of constants
    '''
    def __init__(self, G, C0=None, constants={}, c0=0,distance=euclidean_distance):
        super().__init__(G=G, C0=C0)
        self.G_start = deepcopy(self.G)
        self.constants = constants.copy()
        if 'size' not in self.constants:
            # We take the constant term into account by giving each node a weight 'size'
            # that is equal to the number of nodes in the (super)node.
            self.constants['size'] = c0
            nx.set_node_attributes(self.G, 1, 'size')
        self.N = len(G)*(len(G)-1)/2
        # A dictionary from community to sums of attributes
        self.community_aggregates = {}
        self.initialize_aggregates()
        # Note that x_v and x_vv are constant
        self.x_v = self.compute_x_v(G=self.G, constants=self.constants)
        self.x_vv = self.compute_x_vv(G=self.G, constants=self.constants)
        self.x_b = self.compute_x_b()
        self.x_vv_res = sum([
            c * sum([
                self.G.nodes[i][agg]**2
                for i in self.G.nodes
            ])
            for agg, c in self.constants.items()
        ])
        self.x_vb = self.compute_x_vb()
        self.distance = distance

    def get_distance(self,distance=None):
        if distance is None:
            distance = self.distance
        return distance(
            x_vb=self.x_vb,
            x_vv=self.x_vv,
            x_v=self.x_v,
            x_b=self.x_b,
            N=self.N
        )

    def candidate_granularity(self,angular=False):
        if angular:
            return central_angular_distance(0, self.x_b, self.x_b, 0, self.N)
        else:
            return euclidean_distance(0, self.x_b, self.x_b, 0, self.N)

    def get_entry(self,i,j):
        return (
            self.G_start[i][j]['s'] if (i,j) in self.G_start.edges else 0
        ) + sum([
            c * self.G_start.nodes[i][attr] * self.G_start.nodes[j][attr]
            for attr,c in self.constants.items()
        ])

    def sum_inside(self,c):
        return sum(
            self.G[i][j]['s']
            for i in c
            for j in set(c).intersection(self.G[i])
        )/2

    # Assuming c_start.intersection(c_end)=={}
    def sum_between(self, c_start, c_end):
        c_small = c_start if len(c_start)<len(c_end) else c_end
        c_large = c_end if len(c_start)<len(c_end) else c_start
        return sum([
            self.G[i][j]['s']
            for i in c_small
            for j in set(c_large).intersection(self.G[i])
        ])

    def initialize_aggregates(self):
        self.community_aggregates = {
            l: {
                's': self.sum_inside(c),
                **{
                    attr: sum(
                        self.G.nodes[i][attr]
                        for i in c
                    )
                    for attr in self.constants.keys()
                }
            }
            for l, c in self.HC.clusters.items()
        }

    def aggregate(self):
        labels = list(self.HC.clusters.keys())
        label2c = {l: i for i,l in enumerate(labels)}
        edges = defaultdict(int)
        for i,j in self.G.edges:
            c_i,c_j = (label2c[self.HC[x]] for x in (i,j))
            edges[(c_i,c_j) if c_i<c_j else (c_j,c_i)] += self.G[i][j]['s']
        G = nx.Graph()
        G.add_nodes_from(labels)
        # Note that we include self-loops
        G.add_weighted_edges_from([
            (labels[c_i],labels[c_j],s)
            for (c_i,c_j),s in edges.items()
        ], weight='s')
        nx.set_node_attributes(G, {
            i: {
                attr: aggregates[attr]
                for attr in self.constants.keys()
                # self.community_aggregates contains the aggregate 's', which we don't need here
            }
            for i,aggregates in self.community_aggregates.items()
        })
        self.G = G

    # Assuming node is not yet in HC[newlabel]
    def update_aggregates_before_enter(self, newlabel, node):
        if newlabel not in self.HC.clusters.keys():
            # print(node,'wants to move to the new label',newlabel)
            # Then it's a move to a new label
            return ({
                's': self.sum_inside({node}),
                **{
                    attr: self.G.nodes[node][attr]
                    for attr in self.constants.keys()
                }
            },0,0)

        if node in self.HC.clusters[newlabel]:
            print("update_aggregates_before_enter error: node", node, "is in community", newlabel)

        aggs_before = self.community_aggregates[newlabel]
        newc = self.HC.clusters[newlabel]
        s_increase = self.sum_between({node}, newc)
        x_vb_change = s_increase + sum([
            c * aggs_before[attr] * self.G.nodes[node][attr]
            for attr,c in self.constants.items()
        ])
        x_b_change = aggs_before['size'] * self.G.nodes[node]['size']
        return ({
            's': aggs_before['s'] + s_increase + self.sum_inside({node}),
            **{
                attr: aggs_before[attr] + self.G.nodes[node][attr]
                for attr in self.constants.keys()
            }
        }, x_b_change, x_vb_change)

    # Assuming node in HC.clusters[oldlabel]
    def update_aggregates_before_leave(self, oldlabel, node):
        if not node in self.HC.clusters[oldlabel]:
            print("update_aggregates_before_leave error: node", node, "is not in community",
                  oldlabel)
        aggs_before = self.community_aggregates[oldlabel]
        oldc = self.HC.clusters[oldlabel] - {node}
        s_decrease = self.sum_between({node}, oldc)
        aggs_after = {
            's': aggs_before['s'] - s_decrease - self.sum_inside({node}),
            **{
                attr: aggs_before[attr] - self.G.nodes[node][attr]
                for attr in self.constants.keys()
            }
        }
        x_vb_change = - s_decrease - sum([
            c * aggs_after[attr] * self.G.nodes[node][attr]
            for attr,c in self.constants.items()
        ])
        x_b_change = - aggs_after['size'] * self.G.nodes[node]['size']
        return (aggs_after, x_b_change, x_vb_change)

    # The nodes who need to be revisited by MoveNodes after node has been moved to newlabel
    def revisit_nodes_after_relabel(self, relabel_action):
        node = relabel_action.node
        new_c = self.HC.clusters[relabel_action.newlabel]
        positive_neighbors = {i for i in self.G.neighbors(node) if self.G[node][i]['s']>0}
        # After this action, the situation might have changed for old_c, new_c and node's neighborhood
        # However, positive neighbors of node that are also in new_c will have become more stable, as well
        # as members of old_c that are not connected to node. Therefore, we need to revisit the symmetric
        # difference between new_c and positive_neighbors.
        return new_c.symmetric_difference(positive_neighbors)

    def relabel_value(self,newlabel,node):
        oldlabel = self.HC[node]
        _,x_b_enter_change,x_vb_enter_change = self.update_aggregates_before_enter(newlabel,node)
        _,x_b_leave_change,x_vb_leave_change = self.update_aggregates_before_leave(oldlabel,node)
        return -self.distance(
            x_b=self.x_b + x_b_enter_change + x_b_leave_change,
            x_vb=self.x_vb + x_vb_enter_change + x_vb_leave_change,
            x_v=self.x_v,
            x_vv=self.x_vv,
            N=self.N
        )

    def action_value(self, action):
        # Only do something for RelabelNode
        if type(action) == RelabelNode:
            return self.relabel_value(action.newlabel, action.node)
        return self.value()

    def perform_action(self,action):
        if type(action) == Aggregate:
            self.aggregate()
        elif type(action) == Flatten:
            # Use the flat version of G
            self.G = self.G_start
        elif type(action) == RelabelNode:
            newlabel = action.newlabel
            node = action.node
            updated_newl, x_b_change, x_vb_change = self.update_aggregates_before_enter(newlabel, node)
            self.community_aggregates[newlabel] = updated_newl
            self.x_b += x_b_change
            self.x_vb += x_vb_change
            oldlabel = self.HC[node]
            if len(self.HC.clusters[oldlabel]) == 1:
                del self.community_aggregates[oldlabel]
            else:
                updated_oldl, x_b_change, x_vb_change = self.update_aggregates_before_leave(oldlabel, node)
                self.community_aggregates[oldlabel] = updated_oldl
                self.x_b += x_b_change
                self.x_vb += x_vb_change
        elif type(action) != RelabelNode:
            print(action,"of type",type(action),"is not allowed for a",type(self))
        super().perform_action(action)

    def value(self):
        return -self.distance(
            x_vb=self.x_vb,
            x_vv=self.x_vv,
            x_v=self.x_v,
            x_b=self.x_b,
            N=self.N)

    def compute_x_b(self):
        return sum([
            agg['size'] * (agg['size']-1)/2
            for agg in self.community_aggregates.values()
        ])

    def compute_x_vb(self):
        x_vb = sum([
            aggs['s'] + sum([
                (c/2) * aggs[name]**2
                for name, c in self.constants.items()
            ])
            for aggs in self.community_aggregates.values()
        ])
        return x_vb - self.x_vv_res/2

    @staticmethod
    def compute_x_v(G, constants={}):
        w = G.nodes
        # Add edge term
        x_v = sum([G[i][j]['s'] for i,j in G.edges])
        # Add product terms
        x_v += sum([
            (c/2)*(sum([
                w[i][attr]
                for i in G.nodes
            ])**2 - sum([
                w[i][attr]**2
                for i in G.nodes
            ]))
            for attr,c in constants.items()
        ])
        return x_v

    @staticmethod
    def compute_x_vv(G, constants={}):
        x_vv = 0
        w = G.nodes
        # Edge terms
        for i,j in G.edges:
            s = G[i][j]['s']
            x_vv += s * (s + 2*sum([
                c * w[i][attr] * w[j][attr]
                for attr, c in constants.items()
            ]))
        # Product terms
        x_vv += sum([
            (c1*c2/2) * (sum([
                w[i][attr1] * w[i][attr2]
                for i in G.nodes
            ])**2 - sum([
                (w[i][attr1]*w[i][attr2])**2
                for i in G.nodes
            ]))
            for (attr1, c1), (attr2, c2) in it.product(constants.items(), constants.items())
        ])
        return x_vv

    @staticmethod
    def FromPairVector(pairvector, C0=None, distance=euclidean_distance):
        from .pair_vector import ConstantDict
        G_new = nx.Graph()
        G_new.add_nodes_from(pairvector.vertices)
        G_new.add_weighted_edges_from([
            (i, j, pairvector.c_sparse * s) for (i, j), s in pairvector.sparse.items()
        ], weight='s')
        factor_names = ['factor' + str(i) for i, _ in enumerate(pairvector.factors)]
        constants = dict(zip(factor_names, pairvector.constants))
        for f, name in zip(pairvector.factors, factor_names):
            nx.set_node_attributes(G_new, f.constant if isinstance(f, ConstantDict) else f, name)
        return VectorScorer(G_new,C0=C0,constants=constants,distance=distance)



