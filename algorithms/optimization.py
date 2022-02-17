from ..Clustering import Clustering,HierarchicalClustering
from .clusteringactions import Done,SkipAction,Restart,RelabelNode


'''
    An optimization algorithm. The constructor can be used to pass hyperparameters
    and to determine the quality function that is optimized (by providing an
    appropriate Scorer object).

    Optimization occurs by iteratively picking an action from a set of candidate
    actions given by
        opt.candidate_actions()
    and then performing it.
    A iterator over the actions can be obtained by
        opt.action_sequence()
    Note that this iterator does not perform the action so that an iteration over
    this sequence will be an infinite loop unless the actions are applied to the
    scorer.
    To perform the next action, call
        opt.next()
    To perform all the actions of the sequence, call
        opt.optimize()
    By default, the first action will be selected. Therefore, the iteration will
    not terminate. This can be changed by overriding select_action, such as in
    GreedyOptimization below.
'''
class OptimizationAlgorithm:
    def __init__(self,scorer):
        self.scorer = scorer

    @classmethod
    def FromQuality(cls,quality_function,G,C0=None):
        from .scorers import Scorer
        if C0==None:
            C0 = HierarchicalClustering.SingletonClustering(G.nodes)
        return cls(Scorer(G,C0,quality_function=quality_function))

    def update_state(self,action):
        pass

    def optimize(self,verbose=False):
        for action in self.action_sequence():
            if verbose:
                print(action)
            if not type(action) in (Done,SkipAction,Restart):
                self.scorer.perform_action(action)
            self.update_state(action)

    def do_relabel_nodes_phase(self,verbose=False):
        for action in self.action_sequence():
            if type(action)==RelabelNode:
                self.scorer.perform_action(action)
            if verbose:
                print(action)
            self.update_state(action)
            if not type(action) in (RelabelNode,SkipAction):
                # Move nodes has ended
                return

    def next(self):
        if not hasattr(self,'action_iterator'):
            self.action_iterator = self.action_sequence()

        action = next(self.action_iterator)
        if not type(action) in (Done,SkipAction):
            self.scorer.perform_action(action)
        self.update_state(action)
        print(action)

    # By default, we'll just terminate immediately
    def candidate_actions(self):
        return set()

    def select_action(self, candidates):
        if len(candidates) == 0:
            # Iteration ends if there are no candidates left
            return Done()
        # Otherwise, just return the first action
        return candidates.pop()

    # Iterator that returns the next action at each point.
    # NOTE THAT IT DOES NOT PERFORM THE ACTION so that iterating over this
    # without performing results in an infinite loop
    def action_sequence(self):
        while True:
            action = self.select_action(self.candidate_actions())
            yield(action)
            if isinstance(action,Done):
                break # Optimization finished


class GreedyOptimization(OptimizationAlgorithm):
    def select_action(self, candidates):
        if len(candidates) == 0:
            # Iteration ends if there are no candidates left
            return Done()
        # If candidate is a dict, we assume its values are the scores.
        if isinstance(candidates,dict):
            best_action = max(candidates,key=candidates.get)
        # If there is only one candidate, then we consider it a 'forced move'
        # Otherwise, only perform an action if it increases the score
        best_action = max(candidates, key=self.scorer.action_value)
        if len(candidates) == 1 or self.scorer.action_value(best_action) > self.scorer.value():
            return best_action
        else:
            return SkipAction("tie")
