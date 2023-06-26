from ..Clustering import Clustering
import networkx as nx
import numpy as np
import itertools as it
ABCD_PATH = 'ABCDGraphGenerator.jl/'


def LFR(n, exp=2.5, sizes_exp=4, mu=1 / 3, avg_deg=10, min_size=10,save=False,retries=10):
    import networkit as nk
    name = 'LFR_n{}_exp{}'.format(n, exp)
    lfr = nk.generators.LFRGenerator(n)
    lfr.generatePowerlawDegreeSequence(avg_deg, min(n, max(n ** (1 / (exp - 1)), 10 * avg_deg)), -exp)
    lfr.generatePowerlawCommunitySizeSequence(min_size, n / 10, -sizes_exp)
    lfr.setMu(mu)
    try:
        G = lfr.generate()
        T = Clustering(lfr.getPartition().getVector())
        if save:
            nk.writeGraph(G, name + '.edges', nk.Format.EdgeListTabZero)
            T.to_csv(name + '.csv')
        return G,T
    except:
        print('fail')
        if retries>0:
            return LFR(n, exp, sizes_exp, mu, avg_deg, min_size, save, retries-1)


class GraphGenerator:
    def __init__(self,display_parameters={}):
        self.display_parameters = display_parameters

    # Returns a tuple (G,T) where G is a networkx graph and T is a Clustering.
    def generate(self,seed=None):
        pass

    def __str__(self):
        pad = 1 + max([len(k) for k in self.display_parameters.keys()])
        return "\n\t".join([type(self).__name__+" with parameters"]+[
            "{}:{}{}".format(k,(pad-len(k))*" ",v)
            for k,v in self.display_parameters.items()
        ])
    


class ABCD_benchmark(GraphGenerator):
    default_params = {
        'n': 400,
        'xi': 0.25,
        'exp_sizes': 1.5,
        'min_size': 10,
        'max_size': None,
        'exp_deg': 2.5,
        'min_deg': 4,
        'max_deg': None,
        'destination_folder': None,
        'abcd_path': ABCD_PATH,
    }

    def __init__(self, **kwargs):
        self.parameters = ABCD_benchmark.default_params.copy()
        self.parameters.update(kwargs)
        n = self.parameters['n']
        if self.parameters['max_size'] is None:
            self.parameters['max_size'] = int(np.round(n**(1/(max(4/3,self.parameters['exp_sizes']-1)))))
        if self.parameters['max_deg'] is None:
            self.parameters['max_deg'] = int(np.round(n**(1/(self.parameters['exp_deg']-1))))
        if self.parameters['destination_folder'] is None:
            self.parameters['destination_folder'] = f'ABCD_n{n}/'
        super().__init__(self.parameters)


    def generate(self, seed, load=False):
        import os
        # Make sure the destination folder exists
        folder = self.parameters['destination_folder']
        if not os.path.isdir(folder):
            print('Creating directory',folder)
            os.mkdir(folder)
        if not load:
            ABCD_benchmark.generate_with_julia(**self.parameters,seed=seed)
        # Load the ground-truth clustering from comm[seed].dat
        T = Clustering(dict([
            map(int,l.strip().split('\t')) 
            for l in open(os.path.join(folder,f'comm{seed}.dat')).readlines()
        ]))
        # Load the graph from net[seed].dat
        G = nx.from_edgelist([
            tuple(map(int,l.strip().split('\t')))
            for l in open(os.path.join(folder,f'net{seed}.dat')).readlines()
        ])
        return G,T
        

    @staticmethod
    def generate_with_julia(n, xi, exp_sizes, min_size, max_size,
                  exp_deg, min_deg, max_deg,seed, abcd_path,destination_folder):
        import os
        name = str(seed)
        # Generate degrees
        os.system(
            'julia '+abcd_path+'utils/deg_sampler.jl '+destination_folder+'deg{}.dat '.format(name)
            +str(exp_deg)+' '
            +str(min_deg)+' '
            +str(max_deg)+' '
            +str(n)
            +' 1000 ' 
            + (str(seed) if seed is not None else ''))
        # Generate community sizes
        os.system(
            'julia '+abcd_path+'utils/com_sampler.jl '+destination_folder+'cs{}.dat '.format(name)
            +str(exp_sizes)+' '
            +str(min_size)+' '
            +str(max_size)+' '
            +str(n)
            +' 1000 '
            + (str(seed) if seed is not None else ''))
        # Generate graph
        os.system(
            'julia '+abcd_path+'utils/graph_sampler.jl {1}net{0}.dat {1}comm{0}.dat {1}deg{0}.dat {1}cs{0}.dat xi '.format(name,destination_folder)
            +str(xi)+' false false '+ (str(seed) if seed is not None else ''))


class SBM(GraphGenerator):
    def __init__(self,sizes,block_matrix=None,
                block_matrix_parameters={},display_parameters={},
                directed=False,sparse=True,selfloops=True):
        if block_matrix is None:
            block_matrix = self.block_matrix(sizes,**block_matrix_parameters)
        self.n = sum(sizes)
        self.k = len(sizes)
        display_parameters.update({
            "#Communities": self.k,
            "#Nodes": self.n
        })
        super().__init__(display_parameters)
        self.networkx_parameters = {
            "p": block_matrix,
            "sizes": sizes,
            "directed": directed,
            "sparse": sparse,
            "selfloops": selfloops
        }
        self.block_matrix_parameters=block_matrix_parameters

    def p_min(self):
        return min(min(c) for c in self.networkx_parameters['p'])

    def p_max(self):
        return max(max(c) for c in self.networkx_parameters['p'])

    def random_sizes(self,seed=None):
        raise NotImplementedError(type(self).__name__+" does not support randomized sizes")

    @staticmethod
    def make_block_matrix(m):
        maximum = m.max()
        if maximum > 1:
            print("Warning: HeterogeneousSizedPPM generated a block matrix with an entry {} > 1.".format(maximum))
            m = np.minimum(m,np.ones_like(m))
        return m

    @staticmethod
    def block_matrix(sizes,**block_matrix_parameters):
        raise NotImplementedError()

    def generate(self,seed=None,randomize_sizes=False):
        parameters = self.networkx_parameters
        if randomize_sizes:
            sizes = self.random_sizes(seed=seed)
            parameters = dict(
                self.networkx_parameters,
                sizes=sizes,
                p=self.block_matrix(sizes,**self.block_matrix_parameters)
            )
        sizes = parameters["sizes"]
        return (
            nx.stochastic_block_model(**parameters, seed=seed),
            Clustering.FromSizes(sizes)
        )

    def basic_parameters(self):
        N = self.n*(self.n-1)/2
        sizes = self.networkx_parameters['sizes']
        p = self.networkx_parameters['p']
        mT = sum([
            s*(s-1)/2 for s in sizes
        ])
        # The expected number of edges. Note that each combination i1!=i2 occurs twice.
        mG = sum([
            p[i1][i2]*s1*(s2 if i1!=i2 else s1-1)/2
            for i1,s1 in enumerate(sizes)
            for i2,s2 in enumerate(sizes)
        ])
        mG_intra = sum([
            p[i][i]*s*(s-1)/2
            for i,s in enumerate(sizes)
        ])
        mix = 1-mG_intra/mG
        return mG, mT, N, mix

    def intra_density(self):
        mG, mT, N, mix = self.basic_parameters()
        return (1-mix)*mG/mT

    def inter_density(self):
        mG, mT, N, mix = self.basic_parameters()
        return mix*mG/(N-mT)

class PPM(SBM):
    def __init__(self,p_in=0.5,p_out=0.02,s=10,k=10,n=None,mean_degree=None,mix=None,
                 directed=False,sparse=True,selfloops=False,exp_sizes=None,sizes=None):
        display_parameters = {}
        if not sizes is None:
            s = None
            k = len(sizes)
            n = sum(sizes)
            intra_pairs = sum(s*(s-1) for s in sizes)/2
        else:
            if not n is None:
                s = int(n/k)
            # When exp_sizes is given, we take power-law sizes. Otherwise, equally-sized
            sizes = [s]*k
            n = s*k
            intra_pairs = k*s*(s-1)/2
        self.exp_sizes = exp_sizes
        if not exp_sizes is None:
            from .powerlaws import powerlaw_fixed
            self.balanced=False
            display_parameters["Power-law community-size exponent"]=exp_sizes
            sizes = powerlaw_fixed(n,k,exp_sizes)
            intra_pairs = sum(s*(s-1) for s in sizes)/2
        else:
            self.balanced=True
        inter_pairs = n*(n-1)/2-intra_pairs
        # When mean_degree and mix are given, p_in,p_out are overruled
        if not (mean_degree is None or mix is None):
            p_in = (1-mix) * mean_degree * n / (2*intra_pairs)
            p_out = mix * mean_degree * n / (2*inter_pairs)
        else:
            out_neighbors = 2*p_out*inter_pairs/n
            mean_degree = 2*(intra_pairs * p_in + inter_pairs * p_out)/n
            mix = out_neighbors/mean_degree
        block_matrix_parameters = {
            "p_in": p_in,
            "p_out": p_out
        }
        display_parameters.update({
            "Internal density": p_in,
            "External density": p_out,
            "Community size": s,
            "Mean degree": mean_degree,
            "Mixing rate": mix
        })
        super().__init__(
            sizes=sizes,block_matrix_parameters=block_matrix_parameters,
            display_parameters=display_parameters,
            directed=directed,sparse=sparse,selfloops=selfloops
        )

    def expected_density(self):
        N = self.n*(self.n-1)/2
        intra = sum(s*(s-1) for s in self.networkx_parameters["sizes"])/2
        p_in,p_out = (self.block_matrix_parameters[ps] for ps in ["p_in","p_out"])
        return p_out * (1-intra/N) + p_in * intra/N

    def random_sizes(self,seed=None):
        if self.exp_sizes is None:
            return self.k*[int(self.n/self.k)]
        from .powerlaws import powerlaw_random
        return powerlaw_random(self.n,self.k,exp=self.exp_sizes,seed=seed)

    def order0_params(self):
        N = self.n * (self.n - 1) / 2
        mT = sum(s * (s - 1) for s in self.networkx_parameters["sizes"]) / 2
        p_in, p_out = (self.block_matrix_parameters[ps] for ps in ["p_in", "p_out"])
        mG = p_out * (N - mT) + p_in * mT
        mix = p_out * (N - mT) / mG
        return mG, mT, N, mix

    @staticmethod
    def block_matrix(sizes,p_in,p_out):
        return [
            [
                p_in if i==j else p_out
                for j in range(len(sizes))
            ] for i in range(len(sizes))
        ]

class HeterogeneousSizedPPM(SBM):
    def __init__(self, k=10, n=100, exp_sizes=2.5, mean_degree=10, mix=0.25,
                 directed=False, sparse=True, selfloops=False, sizes=None):
        from .powerlaws import powerlaw_fixed
        if sizes is None:
            sizes = powerlaw_fixed(n, k, exp_sizes)
        self.exp_sizes = exp_sizes
        super().__init__(
            sizes=sizes,
            block_matrix_parameters={
                "mean_degree": mean_degree,
                "mix": mix
            },
            display_parameters={
                "Mixing rate": mix,
                "Mean degree": mean_degree,
                "Power-law community-size exponent": exp_sizes,
            },
            directed=directed, sparse=sparse, selfloops=selfloops
        )

    def random_sizes(self,seed=None):
        from .powerlaws import powerlaw_random
        return powerlaw_random(self.n,self.k,exp=self.exp_sizes,seed=seed)

    def expected_density(self):
        return self.block_matrix_parameters["mean_degree"]/(self.n-1)

    @staticmethod
    def block_matrix(sizes,mean_degree,mix):
        out_neighbors = mean_degree * mix
        in_neighbors = mean_degree - out_neighbors
        n = sum(sizes)
        out_pairs = n*(n-1)/2 - sum([s*(s-1) for s in sizes])/2
        p_out = (n*out_neighbors/2) / out_pairs
        m = np.array([
            [
                in_neighbors / max(1,s-1) if i==j else p_out
                for j,s in enumerate(sizes)
            ]
            for i in range(len(sizes))
        ])
        return SBM.make_block_matrix(m)


def randomize_communities(self, rand=None):
    if self.fixed_sizes:
        return self.communities.random_same_sizes(rand=rand)
    else:
        from .powerlaws import powerlaw_random
        return Clustering.FromSizes(
            powerlaw_random(**self.community_parameters, rand=rand)
        ).random_same_sizes(rand=rand)


@staticmethod
def degrees_from_sizes(communities, exp_density, exp_degrees, mean_degree, silent=True):
    from community_detection_toolbox.random_graphs.powerlaws import powerlaw_fixed
    node2degree = {}
    for l, c in communities.clusters.items():
        s = len(c)
        # Inside each community, the degrees have a power-law distribution
        # with exponent exp_degrees and a mean proportional to s**exp_density
        node2degree.update(dict(zip(
            c,
            powerlaw_fixed(s ** (exp_density + 1), s, exp=exp_degrees, rounded=False)
        )))
    scaling = len(communities) * mean_degree / sum(node2degree.values())
    degrees = [
        scaling * node2degree[i]
        for i in communities.keys()
    ]
    if not silent:
        print("ILFR generated degrees in the range [{},{}]".format(min(degrees), max(degrees)))
    return degrees


def generate(self, randomize_communities=False, degrees_from_sizes=False, seed=None, silent=True):
    rand = np.random.RandomState(seed=seed)
    T = self.communities
    degrees = self.degrees
    if randomize_communities:
        T = self.randomize_communities(rand=rand)
    if degrees_from_sizes:
        degrees = self.degrees_from_sizes(T, **self.degrees_parameters)
    if degrees is None:
        from .powerlaws import powerlaw_random
        degrees = powerlaw_random(**self.random_degrees_parameters, rand=rand)
    community_degree = {
        label: sum([
            degrees[i]
            for i in c
        ])
        for label, c in T.clusters.items()
    }
    total_degree = sum(community_degree.values())
    edges = []
    excess = 0
    for (i, t_i), (j, t_j) in it.combinations(T.items(), 2):
        d_i, d_j = (degrees[i], degrees[j])
        p = self.mix * d_i * d_j / total_degree
        if t_i == t_j:
            p += (1 - self.mix) * d_i * d_j / community_degree[t_i]
        excess += max(0, p - 1)
        if rand.rand() <= p:
            edges.append((i, j))
    if not silent:
        print("ILFR was generated with excess expectation:", excess)
    # Make graph
    G = nx.Graph()
    G.add_nodes_from(T.keys())
    G.add_edges_from(edges)
    return (G, T)


class DegreeCorrectedSBM(GraphGenerator):
    def __init__(self,degree_seq,communities,block_matrix):
        self.default_degrees = degree_seq
        self.communities = communities
        n = len(degree_seq)
        k = len(communities.sizes())
        self.block_matrix = block_matrix
        self.community_parameters = {
            "total": n,
            "k": k
        }
        mean_degree = sum(degree_seq)/len(degree_seq)
        pass

    def generate(self,seed=None,randomize_sizes=False):
        pass


class IndependentLFR(DegreeCorrectedSBM):
    def __init__(self,degree_seq=None,communities=None,
                n=1000,k=100,exp_degrees=2.5,exp_sizes=2.5,mean_degree=7,mix=0.25,exp_density=0.5,balanced_sizes=False):
        if not degree_seq is None:
            n = len(degree_seq)
            mean_degree = sum(degree_seq)/n
            # We don't override exp_degrees so hopefully it is correctly provided
            self.degrees = degree_seq
        else:
            self.degrees = None
        if balanced_sizes and k>0 and n>0:
            communities = Clustering.FromSizes(Clustering.BalancedSizes(n, k))
        self.fixed_sizes = not communities is None
        if self.fixed_sizes:
            n = len(communities)
            self.communities = communities
            k = len(communities.sizes())
            # We don't override exp_sizes
        else:
            from .powerlaws import powerlaw_fixed
            self.communities = Clustering.FromSizes(powerlaw_fixed(n,k,exp=exp_sizes))
        self.community_parameters = {
            "total": n,
            "k": k,
            "exp": exp_sizes
        }
        self.degrees_parameters = {
            "mean_degree": mean_degree,
            "exp_density": exp_density,
            "exp_degrees": exp_degrees
        }
        self.random_degrees_parameters = {
            "total": n*mean_degree,
            "k": n,
            "exp": exp_degrees
        }
        self.mix = mix
        self.n = n
        self.k = k
        self.mean_degree = mean_degree

    def randomize_communities(self,rand=None):
        if self.fixed_sizes:
            return self.communities.random_same_sizes(rand=rand)
        else:
            from .powerlaws import powerlaw_random
            return Clustering.FromSizes(
                powerlaw_random(**self.community_parameters,rand=rand)
            ).random_same_sizes(rand=rand)

    @staticmethod
    def degrees_from_sizes(communities,exp_density,exp_degrees,mean_degree,silent=False):
        from community_detection_toolbox.random_graphs.powerlaws import powerlaw_fixed
        node2degree = {}
        for l,c in communities.clusters.items():
            s = len(c)
            # Inside each community, the degrees have a power-law distribution
            # with exponent exp_degrees and a mean proportional to s**exp_density
            node2degree.update(dict(zip(
                c,
                powerlaw_fixed(s**(exp_density+1), s, exp=exp_degrees, rounded=False)
            )))
        scaling = len(communities) * mean_degree / sum(node2degree.values())
        degrees = [
            scaling * node2degree[i]
            for i in communities.keys()
        ]
        if not silent:
            print("ILFR generated degrees in the range [{},{}]".format(min(degrees),max(degrees)))
        return degrees

    def generate(self,randomize_communities=False,degrees_from_sizes=False,seed=None,silent=True):
        rand = np.random.RandomState(seed=seed)
        T = self.communities
        degrees = self.degrees
        if randomize_communities:
            T = self.randomize_communities(rand=rand)
        if degrees_from_sizes:
            degrees= self.degrees_from_sizes(T,**self.degrees_parameters)
        if degrees is None:
            from .powerlaws import powerlaw_random
            degrees = powerlaw_random(**self.random_degrees_parameters,rand=rand)
        community_degree = {
            label: sum([
                degrees[i]
                for i in c
            ])
            for label,c in T.clusters.items()
        }
        total_degree = sum(community_degree.values())

        edges = []
        excess = 0
        for (i,t_i),(j,t_j) in it.combinations(T.items(),2):
            d_i,d_j = (degrees[i],degrees[j])
            p = self.mix*d_i*d_j/total_degree
            if t_i==t_j:
                p += (1-self.mix) * d_i*d_j / community_degree[t_i]
            excess += max(0,p-1)
            if rand.rand() <= p:
                edges.append((i,j))
        if not silent:
            print("ILFR was generated with excess expectation:",excess)
        # Make graph
        G = nx.Graph()
        G.add_nodes_from(T.keys())
        G.add_edges_from(edges)
        return G, T
