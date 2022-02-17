from ..Clustering import HierarchicalClustering


# A modification that can be applied to a clustering to result in a new clustering C'.
# Examples are: relabeling of a single node, merging of two clusters, splitting, etcetera.
class ClusteringAction:
    def __init__(self,perform_function=None):
        self.perform_function = perform_function

    # Performs the action on the (hierarchical) clustering HC and return the resulting clustering
    def perform(self,C):
        if self.perform_function!=None:
            return self.perform_function(C)
        # For reproducability purposes
    def __hash__(self):
        return str(self).__hash__()
    # For reproducability purposes
    def __eq__(self,obj):
        return type(self)==type(obj) and hash(self)==hash(obj)


# A non-action (doesn't change the clustering)
class SkipAction(ClusteringAction):
    def __init__(self,description=""):
        self.description = str(description)

    def perform(self,C):
        pass

    def __str__(self):
        if len(self.description)>0:
            return "skip {}".format(self.description)
        return "skip"
    # For reproducability purposes
    def __hash__(self):
        return (type(self),self.description).__hash__()


# A non-action (doesn't change the clustering)
class Done(ClusteringAction):
    def perform(self,C):
       pass

    def __str__(self):
       return "optimization finished"


# A non-action (doesn't change the clustering)
class Restart(ClusteringAction):
    def perform(self,C):
       pass

    def __str__(self):
       return "Start a new iteration"


class RelabelNode(ClusteringAction):
    def __init__(self, node, newlabel):
        self.node = node
        self.newlabel = newlabel

    def perform(self,HC):
        HC[self.node] = self.newlabel
        return HC

    def __str__(self):
        return "relabeling node {} to label {}".format(self.node,self.newlabel)
    def __hash__(self):
        return (type(self),self.node,self.newlabel).__hash__()


# Aggregate the clusters into new nodes by going up a level in the hierarchy
class Aggregate(ClusteringAction):
    print_str = "aggregating clusters"

    def perform(self,HC):
        self.print_str = "aggregating {} clusters".format(len(HC.clusters))
        return HC.nextlevel()

    def __str__(self):
        return self.print_str


# Flatten the Hierarchical clustering into a regular clustering
class Flatten(ClusteringAction):
    def perform(self,HC):
        return HierarchicalClustering(HC.flatClustering())

    def __str__(self):
        return "flattening clusters"


class Recluster(ClusteringAction):
    def __init__(self, mapping_HC):
        self.mapping_HC = mapping_HC

    def perform(self,HC):
        for i in HC.keys():
            HC[i] = self.mapping_HC[i]
        #
        return HC

    def __str__(self):
        mapping_str = "{"
        for i,c in self.mapping_HC.items():
            mapping_str += "{}: {},".format(i,c)
        mapping_str = mapping_str[:-1]+"}"
        return "Reclustering with "+mapping_str


class Refine(ClusteringAction):
    def perform(self,HC):
        for i in HC.keys():
            HC[i] = i
        return HC
    def __str__(self):
        return "Refining"
