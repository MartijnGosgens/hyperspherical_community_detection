from ..Clustering import HierarchicalClustering
import networkx as nx
from .clusteringactions import Flatten,Aggregate,RelabelNode,Done,Restart,SkipAction,Refine,Recluster
import itertools as it
import math

'''
    Often, it is inefficient to recompute Q(G,C') each time the clustering is
    changed. Often, this can be done more efficiently by keeping track of a number
    of variables. The following class can be subclassed to keep track of such variables.
    At any point, value() should return Q(G,HC) and HC should only be modified
    through perform_action(action). The function action_value(action) should return
    Q(G,action.perform(HC)) without actually modifying HC.
'''
class Scorer:
    def __init__(self, G, C0=None, quality_function=None):
        self.G=G
        # Initialize with singleton clustering by default
        if C0 is None:
            self.HC0 = HierarchicalClustering.SingletonClustering(G.nodes)
        elif isinstance(C0,HierarchicalClustering):
            self.HC0 = C0.copy()
        else:
            self.HC0=HierarchicalClustering(C0)
        self.HC = self.HC0.copy()

        # Quality function, if supplied
        self.Q = quality_function

        # Keep list of actions such that HC can be reproduced from HC0 by these actions.
        self.performed_actions = []

    # Should return Q(G,HC) in a fast/cheap way
    def value(self):
        return self.Q(self.G,self.candidate())

    # Should return Q(G,action(HC)) without modifying HC.
    def action_value(self,action):
        HC=self.HC.copy()
        action.perform(HC)
        return self.Q(self.G,HC.flatClustering())

    # Updates clustering by performing the action
    def perform_action(self,action):
        self.HC=action.perform(self.HC)
        self.performed_actions.append(action)

    # Returns the (current) candidate as a flat clustering
    def candidate(self):
        return self.HC.flatClustering()

    # Returns the candidate that is obtained by applying performed_actions[:step]
    # to HC0.
    def candidate_at_step(self,step):
        hc = self.HC0.copy()
        for action in self.performed_actions[:step]:
            hc = action.perform(hc)
        return hc.flatClustering()

    def print_performed_actions(self):
        print(*self.performed_actions,sep='\n')

'''
    A scorer that allows to efficiently optimize a quality function when the
    action value of relabeling can be computed cheaply from the aggregated graph.
    To optimize a quality function, create a subclass and implement
    relabel_value and recalculate_value.
'''
class LouvainScorer(Scorer):
    @staticmethod
    def aggregate_graph(G,C,agg_attrs=[],agg_edge_attrs=[]):
        newG = nx.DiGraph()
        newG.add_nodes_from(C.clusters.keys())
        attrs_dict = {
            node: {
                **{
                    attr: sum([
                        G.nodes[subnode][attr]
                        for subnode in subnodes
                    ])
                    for attr in agg_attrs
                }, **{
                    attr: sum([
                        G[i][j][attr]
                        for i,j in G.subgraph(subnodes).edges
                    ]) + sum([
                        G.nodes[subnode][attr]
                        for subnode in subnodes
                        if attr in G.nodes[subnode]
                    ])
                    for attr in agg_edge_attrs
                }
            }
            for node,subnodes in C.clusters.items()
        }
        nx.set_node_attributes(newG,attrs_dict)
        newEdges={}
        for i,j in G.edges:
            if not (C[i],C[j]) in newEdges:
                newEdges[C[i],C[j]] = 0
            newEdges[C[i],C[j]] += G[i][j]['weight']
        newG.add_weighted_edges_from([
            e+(w,)
            for e,w in newEdges.items()
        ])
        return newG

    def __init__(self, G, C0=None, weight_attr=None,agg_attrs=[],agg_edge_attrs=[]):
        super().__init__(G,C0)
        self.agg_attrs=agg_attrs
        self.agg_edge_attrs = agg_edge_attrs

        # We convert G to a weighted directed graph. Weighted for aggregation
        # purposes, directed for simplicity. The edge attribute will then be 'weight'
        self.G = LouvainScorer.to_directed(G,weight_attr=weight_attr)
        self.G_start = self.G.copy()

        # Total weight of the graph
        self.in_degree = lambda x: (sum(d for _,d in self.G.in_degree(x, weight='weight')) if hasattr(x, '__iter__') else self.G.in_degree(x,weight='weight'))
        self.out_degree = lambda x: (sum(d for _,d in self.G.out_degree(x, weight='weight')) if hasattr(x, '__iter__') else self.G.out_degree(x,weight='weight'))
        self.total = self.in_degree(self.G.nodes)

        self.current_value = self.recalculate_value()

    def to_directed(G,weight_attr=None):
        newG = nx.DiGraph()
        newG.add_nodes_from(G.nodes)
        # Copy all node attributes from G
        nx.set_node_attributes(newG,G.nodes)
        # Copy all edge attributes. this will also copy 'weight' but this'll be overwritten
        nx.set_edge_attributes(newG,G.edges)

        if nx.is_directed(G):
            newG.add_weighted_edges_from([
                (i,j,G[i][j][weight_attr] if weight_attr else 1)
                for i,j in G.edges
            ])
        else:
            # We want the degree to correspond to the weight of outgoing edges,
            # so we add all edges twice
            newG.add_weighted_edges_from([
                (i,j,G[i][j][weight_attr] if weight_attr else 1)
                for i,j in G.edges
            ])
            newG.add_weighted_edges_from([
                (j,i,G[i][j][weight_attr] if weight_attr else 1)
                for i,j in G.edges
            ])
        return newG

    def relabel_value(self,newlabel,node):
        pass

    # Only called at initialization. Leaving this at 0 should not influence the
    # optimization while it will result in wrong values of current_value and value()
    def recalculate_value(self):
        return 0

    def value(self):
        return self.current_value

    def action_value(self,action):
        # Only do something for RelabelNode
        if type(action)==RelabelNode:
            return self.relabel_value(action.newlabel, action.node)
        return self.current_value

    # This is called right before the Aggregate action is performed. Can be
    # implemented to keep track of variables
    def will_aggregate(self):
        pass

    def will_refine(self):
        pass

    def will_recluster(self):
        pass

    def will_relabel(self,newlabel,node):
        pass

    def will_flatten(self):
        pass

    # Whether node i is well-connected to the set of nodes S.
    def well_connected(self,i,S):
        return True

    # The nodes who need to be revisited by MoveNodes after node has been moved to newlabel
    def revisit_nodes_after_relabel(self,relabel_action):
        node = relabel_action.node
        new_c = self.HC.clusters[relabel_action.newlabel]
        neighbors = set(self.G.predecessors(node)).union(self.G.successors(node))
        # After this action, the situation might have changed for old_c, new_c and node's neighborhood
        # However, neighbors of node that are also in new_c will have become more stable, as well
        # as members of old_c that are not connected to node. Therefore, we need to revisit the symmetric
        # difference between new_c and neighbors.
        return new_c.symmetric_difference(neighbors)

    # Updates clustering by performing the action
    def perform_action(self,action):
        if type(action) == Aggregate:
            # Aggregate the graph
            self.G = self.aggregate_graph(self.G,self.HC,agg_attrs=self.agg_attrs,agg_edge_attrs=self.agg_edge_attrs)
            self.will_aggregate()
        elif type(action) == Flatten:
            # Use the flat version of G
            self.G = self.G_start
            self.will_flatten()
        elif type(action) == RelabelNode:
            # Use the flat version of G
            self.current_value = self.relabel_value(action.newlabel,action.node)
            self.will_relabel(action.newlabel,action.node)
        elif type(action) == Refine:
            self.movenodes_value = self.current_value
            self.will_refine()
        elif type(action) == Recluster:
            # Assume refine has been called before
            self.current_value = self.movenodes_value
            self.will_recluster()
        elif type(action) != RelabelNode:
            print(action,"of type",type(action),"is not allowed for a",type(self))
        super().perform_action(action)
        # After the action
        if type(action) in (Refine,Recluster):
            self.current_value = self.recalculate_value()

'''
    Given a list of node attributes agg_attrs that will be aggregated (by sum) per community,
    sum_c function_of_aggregate_values_of(c)
'''
class AggregationScorer(LouvainScorer):
    def initialize_aggregates(self):
        self.community_aggregates = {
            l: {
                'weight': self.weight(c,c,symmetric=False),
                **{
                    attr: sum(
                        self.G.nodes[i][attr]
                        for i in c
                    )
                    for attr in self.agg_attrs
                }
            }
            for l,c in self.HC.clusters.items()
        }

    def recalculate_value(self):
        # Recompute the aggregates
        self.initialize_aggregates()

        return sum(
            self.community_contribution(**aggs)
            for aggs in self.community_aggregates.values()
        )

    def relabel_value(self,newlabel,node,verbose=False):
        oldlabel = self.HC[node]

        if verbose:
            print('move',node,'to',newlabel,'origin:',self.community_contribution(
                    **self.community_aggregates[oldlabel]
                ),'->',self.community_contribution(
                    **self.update_aggregates_before_leave(oldlabel,node)
                ),'destination:',self.community_contribution(
                    **self.community_aggregates[newlabel]
                ),'->',self.community_contribution(
                    **self.update_aggregates_before_enter(newlabel,node)
            ))

        newlabel_contribution_before = self.community_contribution(
            **self.community_aggregates[newlabel]
        ) if newlabel in self.community_aggregates else 0

        # Subtract previous contributions of newlabel and oldlabel and add updated ones
        return self.current_value - newlabel_contribution_before - self.community_contribution(
            **self.community_aggregates[oldlabel]
        ) + self.community_contribution(
            **self.update_aggregates_before_leave(oldlabel,node)
        ) + self.community_contribution(
            **self.update_aggregates_before_enter(newlabel,node)
        )

    def will_relabel(self,newlabel,node):
        self.community_aggregates[newlabel] = self.update_aggregates_before_enter(newlabel,node)
        oldlabel = self.HC[node]
        if len(self.HC.clusters[oldlabel])==1:
            del self.community_aggregates[oldlabel]
        else:
            self.community_aggregates[oldlabel] = self.update_aggregates_before_leave(oldlabel,node)

    def will_refine(self):
        self.movenodes_aggregates = self.community_aggregates

    def will_recluster(self):
        self.community_aggregates = self.movenodes_aggregates

    # Assuming node is not yet in HC[newlabel]
    def update_aggregates_before_enter(self,newlabel,node):
        if not newlabel in self.HC.clusters.keys():
            #print(node,'wants to move to the new label',newlabel)
            # Then it's a move to a new label
            return {
                'weight': self.weight({node},{node},symmetric=False),
                **{
                    attr: self.G.nodes[node][attr]
                    for attr in self.agg_attrs
                }
            }

        if node in self.HC.clusters[newlabel]:
            print("AggregationScorer.update_aggregates_before_enter error: node",node,"is in community",newlabel)

        aggs_before = self.community_aggregates[newlabel]
        newc = self.HC.clusters[newlabel]
        return {
            'weight': aggs_before['weight'] + self.weight({node},newc,symmetric=True) + self.weight({node},{node},symmetric=False),
            **{
                attr: aggs_before[attr] + self.G.nodes[node][attr]
                for attr in self.agg_attrs
            }
        }

    def aggregate_degrees(G,agg_attrs=[],weight_attr='weight'):
        G = G.copy()
        agg_attrs = agg_attrs.copy()
        # Add in/out degree as attributes
        in_degree = G.degree
        out_degree = G.degree
        if G.is_directed():
            in_degree = G.in_degree
            out_degree = G.out_degree
        nx.set_node_attributes(G,{
            i: {
                'in_degree': in_degree(i,weight=weight_attr),
                'out_degree': out_degree(i,weight=weight_attr)
            }
            for i in G.nodes
        })
        if not 'in_degree' in agg_attrs:
            agg_attrs.append('in_degree')
        if not 'out_degree' in agg_attrs:
            agg_attrs.append('out_degree')
        return (G,agg_attrs)

    # Assuming node in HC.clusters[oldlabel]
    def update_aggregates_before_leave(self,oldlabel,node):
        if not node in self.HC.clusters[oldlabel]:
            print("AggregationScorer.update_aggregates_before_leave error: node",node,"is not in community",oldlabel)
        aggs_before = self.community_aggregates[oldlabel]
        oldc = self.HC.clusters[oldlabel]-{node}

        return {
            'weight': aggs_before['weight'] - self.weight({node},oldc,symmetric=True) - self.weight({node},{node},symmetric=False),
            **{
                attr: aggs_before[attr] - self.G.nodes[node][attr]
                for attr in self.agg_attrs
            }
        }

    # Override this method to return the contribution of the community based on the aggregates.
    # The dictionary kwargs contains the aggregates
    def community_contribution(self,weight,**kwargs):
        return weight

    def weight(self,c_start,c_end,symmetric=False,weight_attr='weight'):
        # Note that even if weight_attr!='weight', then still the constructor
        # of LouvainScorer gives self.G edge-weights with label 'weight'
        weight = sum(
            self.G[i][j][weight_attr]
            for i in c_start
            for j in set(c_end).intersection(self.G[i])
        )
        if symmetric:
            weight += sum(
                self.G[i][j][weight_attr]
                for i in c_end
                for j in set(c_start).intersection(self.G[i])
            )
        return weight

class ProductFormScorer(AggregationScorer):
    def __init__(self, G, node_in_weights, node_out_weights, C0=None, weight_attr=None):
        newG = G.copy()
        nx.set_node_attributes(newG,node_in_weights,'in_weight')
        nx.set_node_attributes(newG,node_out_weights,'out_weight')
        agg_attrs = ['in_weight','out_weight']
        super().__init__(newG,C0=C0,weight_attr=weight_attr,agg_attrs=agg_attrs)

    def community_contribution(self,weight,in_weight,out_weight):
        return weight - in_weight * out_weight

    # If withoutself and HC[i]==label, then we compute the penalty between {i} and label-{i}
    def node_community_penalty(self,i,label,withoutself=False):
        selfpenalty = 0
        if withoutself and self.HC[i]==label:
            selfpenalty = -2*self.G.nodes[i]['in_weight']*self.G.nodes[i]['out_weight']
        return selfpenalty + (
            self.community_aggregates[label]['in_weight'] * self.G.nodes[i]['out_weight']
            + self.community_aggregates[label]['out_weight'] * self.G.nodes[i]['in_weight']
        )

    def node_community_contribution(self,i,label):
        weight = sum([
            self.G[i][j]['weight'] for j in self.G.successors(i)
        ]) + sum([
            self.G[j][i]['weight'] for j in self.G.predecessors(i)
        ])
        return weight - self.node_community_penalty(i,label)

    def pair_contribution(self,i,j):
        return (
            self.G[i][j]['weight'] if j in self.G[i] else 0
        ) + (
            self.G[j][i]['weight'] if i in self.G[j] else 0
        ) - (
            self.G.nodes[i]['in_weight'] * self.G.nodes[j]['out_weight']
            + self.G.nodes[i]['out_weight'] * self.G.nodes[j]['in_weight']
        )

class ModularityScorer(ProductFormScorer):
    def __init__(self, G, C0=None, res=1, weight_attr=None):
        in_weight,out_weight = ModularityScorer.in_out_weights(G,res=res,weight_attr=weight_attr)
        super().__init__(G, in_weight, out_weight, C0=C0, weight_attr=weight_attr)

    def in_out_weights(G,res=1,weight_attr=None):
        in_degree = G.degree
        out_degree = G.degree
        if G.is_directed():
            in_degree = G.in_degree
            out_degree = G.out_degree
        total = sum([d for _,d in in_degree(G.nodes, weight=weight_attr)])
        scaling = math.sqrt(res / total)
        in_weight = {
            i: scaling * in_degree(i, weight=weight_attr)
            for i in G.nodes
        }
        out_weight = {
            i: scaling * out_degree(i, weight=weight_attr)
            for i in G.nodes
        }
        return (in_weight,out_weight)

    def value(self,normalized=False):
        if normalized:
            return self.current_value/self.total
        return self.current_value

class CPM(ProductFormScorer):
    def __init__(self, G, C0=None, res=1, weight_attr=None,scale_by_density=True):
        constant_weight = math.sqrt(res)
        if scale_by_density:
            constant_weight *= math.sqrt(
                sum([d for _,d in G.degree(G.nodes, weight=weight_attr)]) / (len(G)*(len(G)-1))
            )
        in_weight = {
            i: constant_weight
            for i in G.nodes
        }
        out_weight = {
            i: constant_weight
            for i in G.nodes
        }
        self.res = res
        # Save in- and out-degrees as aggregates.
        G, agg_attrs = AggregationScorer.aggregate_degrees(G,[])
        super().__init__(G, in_weight, out_weight, C0=C0, weight_attr=weight_attr)

    def value(self,normalized=False):
        if normalized:
            return self.current_value/self.total
        return self.current_value

    @staticmethod
    def res_ppm_mle(p_in,p_out):
        import numpy as np
        w_in,w_out = (-np.log(1-p) for p in (p_in,p_out))
        return (w_in-w_out)/np.log(w_in/w_out)