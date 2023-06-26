import itertools as it
import numpy as np
from .vector_scorer import VectorScorer,euclidean_distance,correlation_distance
from . import louvain as louv

'''
    This file implements the calculus of the pair vectors. The PairVector class inherits from dict, so that the entry
    of a PairVector x corresponding to the vertex-pair (i,j) can be obtained by x[i,j].
    This dictionary does not explicitly compute/store all n*(n-1)/2 entries of the pair-vector, but stores an implicit
    representation which allows for efficient summation and computation of inner products between pair-vectors.
    
    The PairVector class overloads a number of operators: For PairVectors x,y and a scalar constant c, one can compute:
    * the scalar product c*x
    * scalar division x/c
    * the sum x+y
    * the difference x-y
    * the inner product x*y
    
    NOTE:
    The implementation differs from the description in the paper in the sense that here, we consider binary vectors with
    values 1 and 0 instead of +1 and -1. As a result of this, the center of the hypersphere is the vector where each
    entry equals 1/2, instead of 0. All results equivalently hold for this transformation of the hypersphere: the meridians
    coincide and when computing a Louvain projection of a vector on a meridian with a given latitude, we get the same
    clustering as we would get if we had implemented it with +1/-1 binary vectors.
    
    Let G be a networkx Graph and C be a Clustering. The following vectors and mappings are implemented:
    * connectivity(G): the edge-connectivity vector (e(G) in the paper), equal to 1 if there is an edge, 0 otherwise
    * wedges(G): the wedge-vector (w(G) in the paper), equals the number of common neighbors for each vertex-pair
    * query_ER(G,res): the Erdos-Renyi modularity vector with resolution parameter res
    * query_CM(G,res): the Configuration-model modularity vector with resolution parameter res
    * clustering_binary(C): the binary vector representing the clustering C (b(C) in the paper), 1 for each intra-cluster
    pair, 0 otherwise
    * ones: the all-one vector. Note that this is a constant and not a function, as it is implemented to work irrespectively
    of the vector dimension. That is, x+ones will work for any PairVector x.
    * zeroes: the all-zero vector. Again, this is a constant and does not depend on the vector dimension.
    
    We also implement a number of useful functions. For PairVectors x,y and a latitude lat. We have (among others):
    * the angular distance x.angular_distance(y)
    * the meridian angle / correlation distance x.meridian_angle(y)
    * the entry-sum x.sum() (note that this is equal to x*ones)
    * the Euclidean length x.length() (note that this is equal to sqrt(x*x))
    * the latitude x.latitude()
    * the hypersphere projection x.hypersphere_projection(): the unique vector that lies on the same parallel and
    meridian as x, while having Euclidean distance 0.25*sqrt(N) to the 0.5*ones, so that it lies on the hypersphere
    * the parallel projection x.latitude_on_meridian(lat): the unique vector (on the hypersphere) that lies on the same
    meridian as x and has latitude lat
    
    Finally, given a PairVector x, its Louvain projection can be computed by louvain_projection(x).
    By default, this minimizes the euclidean distance, which is equivalent to minimizing the angular distance. Whenever
    x is a modularity vector (query_ER or query_CM), this is equivalent to maximizing modularity. To instead minimize the
    correlation distance, call louvain_projection(x,distance=correlation_distance)
'''

class immutabledict(dict):
    def __hash__(self):
        return id(self)

    def _immutable(self, *args, **kws):
        raise TypeError('object is immutable')

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable
    pop = _immutable
    popitem = _immutable


class ImmutablePairDict(immutabledict):
    def __init__(self, x):
        super().__init__(x if isinstance(x, ImmutablePairDict) else ImmutablePairDict.transform(dict(x)))

    @staticmethod
    def transform(x):
        return {
            ImmutablePairDict._keytransform(key): value
            for key, value in x.items()
        }

    @staticmethod
    def _keytransform(key):
        # Assume item consists of two vertices
        i, j = key
        # Order
        if i > j:
            j, i = key
        return (i, j)


class PairVector(ImmutablePairDict):
    '''
        sparse_dict: Dictionary indexed by vertex-pairs. Though it is not required to actually be sparse,
        if the dict is not sparse, then this will affect the running times
        factors: A list of dictionaries indexed by vertices
        constants: list of scalar constants with the same length as factors
        vertices: The vertices over which the pairs are computed. If None is given, we take the keys of the first factors-entry.
        Otherwise, vertices will b set to None
    '''

    def __init__(self, sparse_dict={}, factors=[], constants=None, c_sparse=1, vertices=None):
        self.sparse = sparse_dict if isinstance(sparse_dict, ImmutablePairDict) else ImmutablePairDict(sparse_dict)
        self.c_sparse = c_sparse
        self.factors = factors.copy()
        if constants is None:
            constants = [1 for f in factors]
        self.constants = constants
        if vertices is None and len(self.factors) > 0 and not isinstance(self.factors[0], ConstantDict):
            vertices = list(self.factors[0].keys())
        self.vertices = vertices

    def sum(self,vertices=None):
        if vertices is None:
            vertices = self.vertices
        result = self.c_sparse * sum(self.sparse.values())
        for c, f in zip(self.constants, self.factors):
            result += (c / 2) * (
                    sum([f[i] for i in vertices]) ** 2
                    - sum([f[i]**2 for i in vertices])
            )
        return result

    def N(self,vertices=None):
        if vertices is None:
            vertices = self.vertices
        n=len(vertices)
        return n*(n-1)/2

    def length(self,vertices=None):
        if vertices is None:
            vertices = self.vertices
        return PairVector.innerproduct(self, self, vertices=vertices) ** 0.5

    def latitude(self,vertices=None):
        if vertices is None:
            vertices = self.vertices
        N=self.N(vertices=vertices)
        x1=self.sum(vertices=vertices)
        xx=PairVector.innerproduct(self, self, vertices=vertices)
        return np.arccos(
            (N/2-x1)/(
                (N/4+xx-x1)*N
            ) ** 0.5
        )

    def hypersphere_projection(self):
        # We compute 0.5*ones + (x-0.5*ones) * |0.5*ones| / |self-0.5*ones|
        # That is, ones*0.5*(1-|0.5*ones|/|self-0.5*ones|)+x*|0.5*ones|/|self-0.5*ones|
        N = self.N()
        all_one = constant_vector(1,self.vertices)
        scaling = 0.5*N**0.5 / (self*self+N/4-self.sum())**0.5
        return all_one*0.5*(1-scaling) + scaling*self

    def latitude_on_meridian(self, lat):
        # Returns a point on the same meridian as self, with latitude equal to lat.
        N = self.N()
        squared = self*self
        total = self.sum()
        all_one = constant_vector(1,self.vertices)
        return all_one * (
            (1 - np.cos(lat) - np.sin(lat) * total / (N * (squared - total ** 2 / N)) ** 0.5) / 2
        ) + self * (
            (N / (squared - total ** 2 / N)) ** 0.5 * np.sin(lat) / 2
        )

    def euclidean_distance(self,other):
        vertices = self.vertices if other.vertices is None else other.vertices
        self_sqrd = PairVector.innerproduct(self,self,vertices)
        other_sqrd = PairVector.innerproduct(other,other,vertices)
        inner = PairVector.innerproduct(self,other,vertices)
        return (self_sqrd + other_sqrd - 2*inner)**0.5

    def angular_distance(self, other):
        vertices = self.vertices if other.vertices is None else other.vertices
        N = self.N()
        self_sum = self.sum()
        other_sum = other.sum()
        self_sqrd = PairVector.innerproduct(self,self,vertices)
        other_sqrd = PairVector.innerproduct(other,other,vertices)
        inner = PairVector.innerproduct(self,other,vertices)
        return np.arccos((inner+N/4-(self_sum+other_sum)/2)/(
            (self_sqrd+N/4-self_sum)
            *(other_sqrd+N/4-other_sum)
        )**0.5)

    def meridian_angle(self, other):
        vertices = self.vertices if other.vertices is None else other.vertices
        self_lat = self.latitude(vertices=vertices)
        other_lat = other.latitude(vertices=vertices)
        if min(self_lat,other_lat)==0 and max(self_lat,other_lat)==np.pi:
            return np.pi
        if min(self_lat,other_lat)==0 and max(self_lat,other_lat)<np.pi or min(self_lat,other_lat)>0 and max(self_lat,other_lat)==np.pi:
            return np.pi/2
        gc_dist = self.angular_distance(other)
        # If both are on the same meridian
        if gc_dist==abs(self_lat-other_lat):
            return 0
        return np.arccos(
            (np.cos(gc_dist) - np.cos(self_lat) * np.cos(other_lat)) / (
                    np.sin(self_lat) * np.sin(other_lat)
            )
        )

    def scalar_multiplication(self, scalar):
        return PairVector(
            sparse_dict=self.sparse,
            c_sparse=self.c_sparse * scalar,
            factors=self.factors,
            constants=[scalar * c for c in self.constants],
            vertices=self.vertices
        )

    def normalize(self,to_length=1):
        return self*(to_length/self.length())

    # If vertices is not given, we assume len(self.factors)>0
    def longform(self, vertices=None):
        if vertices is None:
            vertices = self.vertices
        if vertices is None:
            raise ValueError('Either provide vertices or make sure self.vertices is set.')
        return [
            self[p]
            for p in it.combinations(vertices, 2)
        ]

    def __getitem__(self, key):
        i, j = PairVector._keytransform(key)
        factor_sum = sum([c * factor[i] * factor[j] for c, factor in zip(self.constants, self.factors)])
        sparse = self.c_sparse * self.sparse[i, j] if (i, j) in self.sparse else 0
        return sparse + factor_sum

    def __add__(self, other):
        sparse = {}
        c_sparse=1
        if len(self.sparse)>0 and len(other.sparse)==0:
            sparse = self.sparse
            c_sparse = self.c_sparse
        elif len(other.sparse)>0 and len(self.sparse)==0:
            sparse = other.sparse
            c_sparse = other.c_sparse
        elif len(other.sparse)>0 and len(self.sparse)>0:
            if isinstance(self.sparse,ClusteringDict) or isinstance(other.sparse,ClusteringDict):
                print('Warning: Adding a clustering vector with a vector that has a nonempty sparse dict is inefficient and may lead to numerical errors.')
            sparse = {
                p: self.c_sparse * v
                for p, v in self.sparse.items()
            }
            for pair, value in other.sparse.items():
                sparse[pair] = other.c_sparse * value + (sparse[pair] if pair in sparse else 0)
                if sparse[pair]==0:
                    del sparse[pair]
        # Combine all contant_dicts into one
        factors=[]
        constants=[]
        c_constantdict = 0
        for c,f in zip(self.constants+other.constants,self.factors+other.factors):
            if isinstance(f,ConstantDict):
                c_constantdict += c*f.constant**2
            else:
                factors.append(f)
                constants.append(c)
        if c_constantdict!=0:
            constants.append(c_constantdict)
            factors.append(ConstantDict(1))
        return PairVector(
            sparse_dict=sparse,
            c_sparse=c_sparse,
            factors=factors,
            constants=constants,
            vertices=other.vertices if self.vertices is None else self.vertices
        )

    def __sub__(self,other):
        if isinstance(other, PairVector):
            return self+(-other)
        else:
            raise TypeError('Cannot subtract object of type '+type(other).__name__+' from a PairVector')

    def __mul__(self, other):
        if isinstance(other, PairVector):
            return PairVector.innerproduct(self, other)
        elif isinstance(other, float) or isinstance(other, int):
            return self.scalar_multiplication(other)
        else:
            raise TypeError('Cannot multiply PairVector with object of type ' + type(other).__name__)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return self*(1/other)
        else:
            raise TypeError('Cannot divide PairVector by object of type ' + type(other).__name__)

    def __neg__(self):
        return self * -1

    '''
        Computes the inner product of v and w. It is assumed that v and w are vectors over the same vertices and
        that at least one of them has the vertices property set to something other than None.
    '''

    @staticmethod
    def innerproduct(v, w, vertices=None):
        if isinstance(w.sparse, ClusteringDict) and not isinstance(v.sparse, ClusteringDict):
            # If only one of v or w is a clustering, we will let v be the clustering
            return PairVector.innerproduct(w, v, vertices=vertices)
        result = 0
        # f_v*f_w
        if len(v.factors) > 0 and len(w.factors) > 0:
            if vertices is None:
                vertices = v.vertices if w.vertices is None else w.vertices
            if vertices is None:
                raise ValueError('Either v or w must have vertices specified.')
        for (c_v, f_v), (c_w, f_w) in it.product(zip(v.constants, v.factors), zip(w.constants, w.factors)):
            result += (c_v * c_w / 2) * (
                    sum([f_v[i] * f_w[i] for i in vertices]) ** 2
                    - sum([(f_v[i] * f_w[i]) ** 2 for i in vertices])
            )
        if isinstance(v.sparse, ClusteringDict) and isinstance(w.sparse, ClusteringDict):
            # s_v*s_w: constants times the number of common intra pairs
            result += v.c_sparse * w.c_sparse * (w.sparse.clustering * v.sparse.clustering).intra_pairs()
        elif isinstance(v.sparse, ClusteringDict):
            # s_v*s_w: constants times w.sparse pairs that are intra pairs
            result += v.c_sparse * w.c_sparse * sum([
                w.sparse[p]
                for p in w.sparse.keys()
                if p in v.sparse
            ])
        else:
            # s_v*s_w
            result += v.c_sparse * w.c_sparse * sum([
                v.sparse[p] * w.sparse[p]
                for p in set(v.sparse.keys()).intersection(w.sparse.keys())
            ])
        if isinstance(v.sparse, ClusteringDict):
            # s_v*f_w: sum of constants within clusters
            result += v.c_sparse * sum([
                sum([
                    (c / 2) * (sum([f[i] for i in cluster]) ** 2 - sum([f[i] ** 2 for i in cluster]))
                    for c, f in zip(w.constants, w.factors)
                ])
                for cluster in v.sparse.clustering.clusters.values()
            ])
        else:
            # s_v*f_w
            result += v.c_sparse * sum([
                s * sum([
                    c * f[i] * f[j]
                    for c, f in zip(w.constants, w.factors)
                ])
                for (i, j), s in v.sparse.items()
            ])
        if isinstance(w.sparse, ClusteringDict):
            # f_v*s_w: sum of constants within clusters
            result += w.c_sparse * sum([
                sum([
                    (c / 2) * (sum([f[i] for i in cluster]) ** 2 - sum([f[i] ** 2 for i in cluster]))
                    for c, f in zip(v.constants, v.factors)
                ])
                for cluster in w.sparse.clustering.clusters.values()
            ])
        else:
            # f_v*s_w
            result += w.c_sparse * sum([
                s * sum([
                    c * f[i] * f[j]
                    for c, f in zip(v.constants, v.factors)
                ])
                for (i, j), s in w.sparse.items()
            ])
        return result


class ClusteringDict(ImmutablePairDict):
    '''
        Returns 1 for intra-pairs and 0 otherwise. C should be a Clustering object
    '''
    def __init__(self, C):
        self.clustering = C

    def __contains__(self, key):
        return self.clustering[key[0]] == self.clustering[key[1]]

    def keys(self):
        return self.clustering.intra_pairs_iter()

    def values(self):
        return it.repeat(1, self.clustering.intra_pairs())

    def items(self):
        return zip(self.keys(), self.values())

    def __getitem__(self, key):
        return 1 if key in self else 0

    def __len__(self):
        return self.clustering.intra_pairs()


class ClusteringVector(PairVector):
    def __init__(self, C):
        self.clustering = C
        sparse = ClusteringDict(C)
        self.sparse = sparse
        self.c_sparse = 1
        self.vertices = list(C.keys())
        self.factors = []
        self.constants = []

    def sum(self,vertices=None):
        return self.c_sparse * self.clustering.intra_pairs()

    def length(self,vertices=None):
        return self.c_sparse * (self.clustering.intra_pairs() ** 0.5)


class ConstantDict(immutabledict):
    def __init__(self, constant=1):
        self.constant = constant

    def __getitem__(self, key):
        return self.constant

    def __contains__(self,key):
        return True


def constant_vector(constant,vertices=None):
    return PairVector(
        factors=[] if constant==0 else [ConstantDict(1)],
        constants=[] if constant==0 else [constant],
        vertices=vertices
    )


# Standard vectors
ones = constant_vector(1)
zeros = constant_vector(0)


'''
    Computes the heuristic query latitude for an input vector with correlation distance theta to the ground-truth
    clustering and a ground-truth latitude lambdaT.
'''
def heuristic_latitude(lambdaT, theta):
    return np.arccos(np.cos(lambdaT) * np.cos(theta) / (
            1 + np.sin(lambdaT) * np.sin(theta)
        ))


'''
    Applies the heuristic to the input vector x. If lambdaT and theta are set, it immediately passes them on to 
    heuristic_latitude. Otherwise, it calculates them from bT or T.
'''
def do_heuristic(x,bT=None,lambdaT=None,theta=None,T=None):
    if bT is None and T is not None:
        bT = clustering_binary(T)
    if bT is not None:
        lambdaT = bT.latitude()
        theta = x.meridian_angle(bT)
    lat = heuristic_latitude(lambdaT,theta)
    return x.latitude_on_meridian(lat)


def connectivity(G):
    return PairVector(
        sparse_dict={
            e: 1
            for e in G.edges
        },
        vertices=list(G.nodes)
    )


def degree_product(G, normalized=False):
    return PairVector(
        factors=[dict(G.degree)],
        vertices=list(G.nodes),
        constants=[
            1 if not normalized else 1 / (2 * len(G.edges))
        ]
    )


def wedges(G):
    from collections import defaultdict
    sparse = defaultdict(int)
    for i in G.nodes:
        for pair in it.combinations(G.neighbors(i), 2):
            sparse[min(pair), max(pair)] += 1
    return PairVector(
        sparse_dict=sparse,
        vertices=list(G.nodes)
    )


def jaccard(G):
    jac_sim = lambda A, B: len(A & B)/len(A | B)
    neighborhoods = {
        i: set(G[i]) | {i}
        for i in G.nodes
    }
    pairs = {
        (min(i, j), max(i, j))
        for i, j in G.edges
    }
    for v, neighborhood in neighborhoods.items():
        pairs |= {
            (min(i, j), max(i, j))
            for i, j in it.combinations(neighborhood - {v}, 2)
        }
    return PairVector(vertices=G.nodes, sparse_dict={
        (i, j): jac_sim(neighborhoods[i], neighborhoods[j])
        for i, j in pairs
    })


def query_connectivity(G,**kwargs):
    lat = np.pi/2
    eG = connectivity(G)
    return eG.latitude_on_meridian(lat)


def noise_vector(vertices=None,n=None,sampler=np.random.normal):
    """
        Computes a vector where each entry is drawn from sampler. Sampler needs to be a function that takes an argument 'size'.
    """
    if vertices is not None:
        n = len(list(vertices))
    else:
        vertices = list(range(n))
    return PairVector(
        vertices=vertices,
        sparse_dict=dict(zip(
            it.combinations(vertices,2),
            sampler(size=int(n*(n-1)/2))
        ))
    )


def query_PPM_MLE(G, mT=None, mix=None,T=None):
    mG = len(G.edges)
    N = len(G)*(len(G)-1)/2
    if T is not None:
        mT = T.intra_pairs()
        mix = len([1 for i,j in G.edges if T[i] != T[j]])/mG
    p_in = (1-mix)*mG/mT
    p_out = mix*mG/(N-mT)
    p = (p_in-p_out)/np.log(p_in/p_out)
    res = p / (mG/N)
    lat = np.arctan(((N-mG)/mG)**0.5 / (res-1))
    if lat<0:
        lat+=np.pi
    return connectivity(G).latitude_on_meridian(lat)


def query_CM(G,res=1):
    return 0.5 * ones + connectivity(G) - res * degree_product(G, normalized=True)


def query_ER(G,res=1):
    mG = len(G.edges)
    N = len(G) * (len(G) - 1) / 2
    return (0.5-res*mG/N) * ones + connectivity(G)


def query_CM_MLE(G,mT=None,mix=None,T=None):
    mG = len(G.edges)
    N = len(G) * (len(G) - 1) / 2
    if T is not None:
        mT = T.intra_pairs()
        mix = len([1 for i, j in G.edges if T[i] != T[j]]) / mG
    p_in = (1 - mix) * mG / mT
    p_out = mix * mG / (N - mT)
    p = (p_in - p_out) / np.log(p_in / p_out)
    res = p / (mG / N)
    return query_CM(G,res)


def query_wedges(G,**kwargs):
    lat = np.pi/2
    wG = wedges(G)
    return wG.latitude_on_meridian(lat)


def clustering_binary(C):
    return ClusteringVector(C)


def louvain_projection(target,lat=None,distance=euclidean_distance,silent=True,return_vec=False,C0=None):
    if lat is not None:
        target = target.latitude_on_meridian(lat)
    scorer=VectorScorer.FromPairVector(target,distance=distance,C0=C0)
    if not silent:
        print('Finding projection of query vec with latitude',target.latitude())
    opt=louv.Louvain(scorer)
    opt.optimize()
    candidate = opt.scorer.candidate()
    if return_vec:
        return clustering_binary(candidate)
    return candidate