from ..Clustering import Clustering
import networkx as nx
import numpy as np
import itertools as it



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
    def __init__(self,k=10,n=100,exp_sizes=2.5,mean_degree=10,mix=0.25,
                directed=False,sparse=True,selfloops=False):
        from .powerlaws import powerlaw_fixed
        sizes = powerlaw_fixed(n,k,exp_sizes)
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
            directed=directed,sparse=sparse,selfloops=selfloops
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