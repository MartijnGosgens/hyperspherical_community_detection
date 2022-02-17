from .clusteringactions import SkipAction,RelabelNode,Aggregate,Flatten
from .optimization import OptimizationAlgorithm,GreedyOptimization
from ..Clustering import HierarchicalClustering

from collections import deque
from enum import Enum
import networkx as nx
import itertools as it
import numpy as np


class Louvain(GreedyOptimization):
    '''
    I like to view Louvain as a state diagram with the states represented by
    the enum LouvainState. The actions cause the following transitions:
        RelabelNode: MOVENODES_UNIMPROVED/MOVENODES_IMPROVED -> MOVENODES_IMPROVED
        Aggregate:   PASS_HAS_IMPROVED -> MOVENODES_UNIMPROVED
        Flatten: PASS_WAS_STABLE -> ITERATION_ENDED
        SkipAction:
            if the queue is empty:
                MOVENODES_UNIMPROVED -> PASS_WAS_STABLE
                MOVENODES_IMPROVED -> PASS_HAS_IMPROVED
            else we remain in state MOVENODES_UNIMPROVED/MOVENODES_IMPROVED
    Louvain can be applied iteratively by transitioning from ITERATION_ENDED to
    MOVENODES_UNIMPROVED (whenever the iteration performed RelabelNode at least once).
    After each occurrence of RelabelNode or SkipAction, current_node is taken from
    the queue.
    '''
    class LouvainState(Enum):
        MOVENODES_UNIMPROVED=0
        MOVENODES_IMPROVED=1
        PASS_WAS_STABLE=2
        PASS_HAS_IMPROVED=3
        ITERATION_ENDED=4

    def __init__(self,scorer,iterative=False,random=False,seed=None):
        super().__init__(scorer)
        # Keep track of the action_index at the start of the iteration so that
        # At the end of the iteration, we can see whether we can terminate.
        self.action_index_start = len(scorer.performed_actions)

        self.iterative = iterative
        if random:
            self.random = np.random.RandomState(seed=seed)
        else:
            self.random = False

        self.state = Louvain.LouvainState.MOVENODES_UNIMPROVED
        self.start_pass()

    def start_pass(self):
        nodes = self.scorer.HC.keys()
        # Randomize order
        if self.random:
            nodes = list(nodes)
            self.random.shuffle(nodes)
        self.queue = deque(nodes)
        self.stable_nodes = []
        self.current_node = self.queue.popleft()

    def iteration_ended(self):
        if not self.iterative:
            return # We are done, the algorithm has terminated
        # Check if RelabelNode was performed since the first aggregation of this iteration
        new_actions = self.scorer.performed_actions[self.action_index_start:]
        aggregated = False
        for action in new_actions:
            if type(action)==Aggregate:
                aggregated = True
            if aggregated and type(action)==RelabelNode:
                self.action_index_start = len(self.scorer.performed_actions)
                self.state = Louvain.LouvainState.MOVENODES_UNIMPROVED
                self.start_pass()
                return
        # Else, The algorithm has terminated

    def candidate_actions(self):
        if self.state == Louvain.LouvainState.ITERATION_ENDED:
            # Terminate
            return set()
        if self.state == Louvain.LouvainState.PASS_WAS_STABLE:
            return {Flatten()}
        if self.state == Louvain.LouvainState.PASS_HAS_IMPROVED:
            return {Aggregate()}

        # Else we return the RelabelNode candidates
        labels = {self.scorer.HC.newlabel()}
        # Don't allow to move to new community if it is in a singleton
        if len(self.scorer.HC.clusters[self.scorer.HC[self.current_node]])==1:
            labels=set()
        # Check if the graph is aggregated
        if self.current_node in self.scorer.G.nodes:
            # If so, return the labels of the neighbors of i
            neighbors = set(self.scorer.G.neighbors(self.current_node))
            if self.scorer.G is nx.DiGraph:
                neighbors |= self.scorer.G.predecessors(self.current_node)
            labels = labels.union({
                self.scorer.HC[j]
                for j in neighbors
            } - {self.scorer.HC[self.current_node]})
        else:
            # If not, we simply return all other labels
            labels = labels.union(self.scorer.HC.labels() - {self.scorer.HC[self.current_node]})
        # We add the SkipAction, so that there is always at least one action returned.
        # If only one action is returned (a forced action), it will be the SkipAction.
        return {
            RelabelNode(self.current_node, label)
            for label in labels
        }.union({SkipAction(self.current_node)})

    def select_action(self,candidates):
        candidates = {
            c: self.scorer.action_value(c)
            for c in candidates
        }
        supermove= super().select_action(candidates)
        return supermove

    def update_state(self,action):
        if type(action)==RelabelNode:
            self.state = Louvain.LouvainState.MOVENODES_IMPROVED
            # We add all nodes that were previously stable, since this change
            # may make them unstable again.
            # The speed could be significantly improved by efficiently identifying
            # which nodes definitely remained stable so that they do not need to
            # requeued.
            self.queue.extend(self.stable_nodes)
            self.stable_nodes = [self.current_node]
            # len(queue)==len(HC.keys())-len(stable_nodes)==len(HC.keys())-1
            # while an occurrance of RelabelNode implies len(HC.keys())>1.
            # Therefore the queue must be nonempty at this point
            self.current_node = self.queue.popleft()
        if type(action)==SkipAction:
            self.stable_nodes.append(self.current_node)
            if len(self.queue)>0:
                self.current_node = self.queue.popleft()
            elif self.state==Louvain.LouvainState.MOVENODES_IMPROVED:
                self.state = Louvain.LouvainState.PASS_HAS_IMPROVED
            else: # self.state==Louvain.LouvainState.MOVENODES_UNIMPROVED
                self.state = Louvain.LouvainState.PASS_WAS_STABLE

        if type(action)==Aggregate:
            self.state = Louvain.LouvainState.MOVENODES_UNIMPROVED
            self.start_pass()
        if type(action)==Flatten:
            self.state = Louvain.LouvainState.ITERATION_ENDED
            self.iteration_ended()
