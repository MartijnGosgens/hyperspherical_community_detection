import numpy as np
import itertools as iter

class Clustering(dict):
    def __init__(self, clustering_dict):
        if not isinstance(clustering_dict, dict):
            clustering_dict = dict(zip(range(len(clustering_dict)),clustering_dict))
        super().__init__(clustering_dict)
        self.clusters = {}
        for i, c in clustering_dict.items():
            if not c in self.clusters:
                self.clusters[c] = set()
            self.clusters[c].add(i)

    # Override
    def __setitem__(self, key, value):
        self.clusters[self[key]].remove(key)
        if len(self.clusters[self[key]]) == 0:
            del self.clusters[self[key]]
        if not value in self.clusters:
            self.clusters[value] = set()
        self.clusters[value].add(key)
        super().__setitem__(key, value)

    def __str__(self):
        return self.clusters.__str__()

    # Override
    def copy(self):
        return Clustering(super().copy())

    def labels(self):
        return self.clusters.keys()

    def swap(self, i, j):
        self[i], self[j] = self[j], self[i]

    def merge(self, c1, c2,newc=None):
        if newc==None:
            newc=c1
        for i in self.clusters[c1].union(self.clusters[c2]):
            self[i]=newc

    def intra_pairs(self):
        return sum([
            int(size*(size-1)/2)
            for size in self.sizes()
        ])

    def intra_pairs_iter(self):
        for c in self.clusters.values():
            for i,j in iter.combinations(c,2):
                yield min(i, j), max(i, j)

    def angular_coarseness(self,return_cos=False):
        N = len(self)*(len(self)-1)/2
        mC = self.intra_pairs()
        cos = 1-2*mC/N
        if return_cos:
            return cos
        return np.arccos(cos)

    def density(self):
        return self.intra_pairs() / ((len(self)*(len(self)-1))/2)

    def partition(self):
        return list(self.clusters.values())

    def sizes(self):
        return [
            len(cluster) for cluster in self.clusters.values()
        ]

    def Meet(A, B):
        meet = {}
        for i in A.keys():
            if not (A[i],B[i]) in meet:
                meet[A[i],B[i]] = set()
            meet[A[i],B[i]].add(i)
        return meet

    # Operator overload of A*B will return the meet of the clusterings.
    def __mul__(self,other):
        return Clustering.FromClusters(Clustering.Meet(self,other))

    def FromSizes(sizes):
        return Clustering(sum([
            [c]*size
            for c, size in enumerate(sizes)
        ],[]))

    def FromPartition(partition):
        clustering = {}
        for c,p in enumerate(partition):
            for i in p:
                clustering[i] = c
        return Clustering(clustering)

    def FromClusters(clusters):
        return Clustering({
            i: l
            for l,c in clusters.items()
            for i in c
        })

    def FromAnything(A):
        # Check if not already a clustering.
        if type(A) == Clustering:
            return A
        if isinstance(A, dict):
            return Clustering(A)
        # See whether it is iterable.
        if hasattr(A, '__iter__'):
            A = list(A)
            if type(A[0]) in {set, list}:
                # If the first item is a set or list, we consider it a partition.
                return Clustering.FromPartition(A)
            else:
                # Else we assume its a list of cluster labels
                return Clustering(A)

    def relabel_communities(self,mapping=None,random=False):
        if mapping==None:
            labels = list(range(len(self.clusters)))
            if random:
                np.random.shuffle(labels)
            mapping = dict(zip(self.clusters.keys(),labels))
        return Clustering({
            i: mapping[c]
            for i,c in self.items()
        })

    @staticmethod
    def BalancedSizes(n, k=None,intra_pairs=None):
        if not intra_pairs is None:
            # Choose k to best match intra_pairs
            k = int(n / (2*intra_pairs/n+1))
            intra_pairs_floor = sum([s*(s-1) for s in Clustering.BalancedSizes(n,k)])/2
            intra_pairs_ceil = sum([s*(s-1) for s in Clustering.BalancedSizes(n,k+1)])/2
            if abs(intra_pairs-intra_pairs_ceil)<abs(intra_pairs-intra_pairs_floor):
                k += 1
                
        smallSize = int(n/k)
        n_larger = n - k * smallSize
        return [smallSize + 1] * n_larger + [smallSize] * (k - n_larger)

    def BalancedClustering(n, k):
        return Clustering.FromSizes(Clustering.BalancedSizes(n, k))

    def random_same_sizes(self, rand=None,seed=None):
        if rand == None:
            rand = np.random
            if not seed is None:
                rand = rand.RandomState(seed)
        c = list(self.values()).copy()
        rand.shuffle(c)
        return Clustering(dict(zip(self.keys(),c)))

    def UniformRandom(n, k, rand=None):
        if rand == None:
            rand = np.random
        return Clustering(rand.randint(k, size=n))

    def newlabel(self):
        return 1+max([-1,*(l for l in self.labels() if isinstance(l,int))])

    def to_csv(self,file_name,index_label='item',cluster_label='cluster'):
        import pandas as pd
        pd.Series(self).to_csv(file_name,index_label=index_label,header=[cluster_label])

    # Returns the cluster that item belongs to. Setting exclusive=True will omit item.
    def cluster_of_item(self,item,exclusive=False):
        c = self.clusters[self[item]]
        if exclusive:
            return c-{item}
        return c

    @classmethod
    def FromCSV(cls,file_name,index_label='item',cluster_label='cluster'):
        import pandas as pd
        return Clustering(pd.read_csv(file_name,index_col=index_label)[cluster_label].to_dict())

    @classmethod
    def SingletonClustering(cls,nodes):
        return cls(dict(zip(nodes,range(len(nodes)))))

class HierarchicalClustering(Clustering):
    # Clustering but with additional list with sets of vertices that are represented by the aggregate-vertex.
    def __init__(self, clustering, previouslevel=None):
        super().__init__(clustering)
        if previouslevel:
            previouslevel = previouslevel.copy()
        self.previouslevel = previouslevel

    # Override
    def copy(self):
        c = super().copy()
        previous = self.previouslevel
        if previous != None:
            previous = previous.copy()
        return HierarchicalClustering(c,previous)

    def nextlevel(self):
        return HierarchicalClustering(dict(zip(self.labels(),self.labels())),self)

    def level(self, lvl):
        if self.previouslevel == None:
            return self
        if lvl <= 0:
            return self.flatClustering()
        return self.previouslevel.level(lvl-1)

    def flatClustering(self):
        if self.previouslevel == None:
            return self

        return Clustering({
            i: self[c]
            for i,c in self.previouslevel.flatClustering().items()
        })

    # Returns the Clusterings corresponding to each level of the hierarchy
    def getlevels(self, flat=True):
        levels = []
        lvl = self
        while not lvl==None:
            if flat:
                levels.append(lvl.flatClustering())
            else:
                levels.append(lvl)
            lvl = lvl.previouslevel
        return levels

    # Returns all intra_pairs that are not in the previous level
    def new_intra_pairs(self):
        if self.previouslevel == None:
            return set().union(*[{
                    (min(e),max(e))
                    for e in iter.combinations(p,2)
                }
                for p in self.partition()
            ])
        previouspartition = self.previouslevel.partition()
        return set().union(*[{
                (min(e),max(e))
                for e in iter.product(previouspartition[i],previouspartition[j])
            }
            for c in self.clusters.values()
            for i,j in iter.combinations(c,2)
        ])

    def flatitems(self, topitems):
        if self.previouslevel == None:
            return topitems
        return set().union(*[
            self.previouslevel.flatitems(self.previouslevel.clusters[i])
            for i in topitems
        ])
