import numpy as np
from community_detection_toolbox.Clustering import Clustering
rand = np.random

def powerlaw_mle(observations):
    minimum = min(observations)
    data = np.array(observations) / minimum
    return 1+1/np.log(data).mean()

def powerlaw_continuous(n_samples,exponent=2):
    return np.exp(
        rand.exponential(
            1/(exponent-1),
            n_samples
        )
    )
def powerlaw_discrete(n_samples,minimum=1, exponent=2):
    return [
        int(minimum*c)
        for c in powerlaw_continuous(n_samples,exponent)
    ]

def powerlaw_vertex_fractions(n_clusters, exponent=3):
    exp = (exponent-2)/(exponent-1)
    # Generate cluster sizes for all but the largest cluster
    fractions = np.array([
        (i+1)**exp - i**exp
        for i in range(1,n_clusters)
    ]) / n_clusters**exp
    return np.insert(fractions,0,1-fractions.sum())

# We require exponent>2
def powerlaw_fixed_alt(n_vertices, n_clusters, exponent=3):
    exp = (exponent-2)/(exponent-1)
    # Generate cluster sizes for all but the largest cluster
    cluster_fractions = np.array([
        (i+1)**exp - i**exp
        for i in range(1,n_clusters+1)
    ]) / n_clusters**exp
    return round_sample(n_vertices*cluster_fractions)

def round_sample(sample):
    total = round(sum(sample))
    rounded = [
        int(s)
        for s in sample
    ]
    dif = sample-rounded
    nchanges = int(total-sum(rounded))
    changes = sorted(range(len(dif)), key=lambda k: -dif[k])[:nchanges]
    return [
        r+1 if i in changes else r
        for i,r in enumerate(rounded)
    ]

def powerlaw_fixed(total,k,exp=3,rounded=True):
    x=np.array(range(1,k+1))/k
    pls=x**(-1/(exp-1))
    unrounded = total*pls/sum(pls)
    if rounded:
        return round_sample(unrounded)
    else:
        return unrounded

def powerlaw_random(total,k,exp=3,seed=None,rand=None,rounded=True):
    if rand is None:
        rand = np.random.RandomState(seed=seed)
    x = rand.rand(k)
    pls=x**(-1/(exp-1))
    unrounded = total*pls/sum(pls)
    if rounded:
        return round_sample(unrounded)
    else:
        return unrounded

def generate_frequencies_from_cdf(cdf,n_samples = 1):
    rands = rand.rand(n_samples)
    rands.sort()
    frequencies = [0]*len(cdf)
    i_sample = 0
    for i,p in enumerate(cdf):
        while rands[i_sample] < p:
            frequencies[i] = frequencies[i]+1
            i_sample = i_sample+1
    return frequencies

def powerlaw_fixed_but_random(n_vertices, n_clusters, exponent=3):
    pmf = powerlaw_vertex_fractions(n_clusters, exponent)
    return rand.multinomial(n_vertices,pmf)
