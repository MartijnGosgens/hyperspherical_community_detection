from ..random_graphs.generators import LFR
from ..Clustering import Clustering
from ..algorithms import pair_vector as pv
import itertools as it

ns = [1000,2000,5000,10000,20000,50000,100000]
exps = [2.9,4]

'''
    The networkit LFR generator seems to crash when calling it multiple times at once (e.g. in loops).
    If the function below does not work, then it may help to call LFR(n,exp=exp,save=True) manually.
'''
def generate_all_graphs():
    for n,exp in it.product(ns,exps):
        LFR(n,exp=exp,save=True)

def load_graph(n,exp):
    import networkx as nx
    name = 'LFR_n{}_exp{}'.format(n,exp)
    G = nx.read_edgelist(name+'.edges',nodetype=int)
    T = Clustering.FromCSV(name+'.csv')
    return G,T

def perform_experiment(load=True):
    import json
    from time import time
    import networkit as nk
    import numpy as np
    results = {}
    if load:
        with open('lfr_results.json') as f:
            results = json.load(f)
    for n, exp in it.product(ns, exps):
        name = 'LFR_n{}_exp{}'.format(n, exp)
        if name in results and 'Wedges latitude' in results[name]:
            print('skipping', name)
            continue
        G, T = load_graph(n, exp)
        bT = pv.clustering_binary(T)
        t_before = time()
        q = pv.query_CM(G)
        C = pv.louvain_projection(q)
        bC = pv.clustering_binary(C)
        results[name] = {
            'Modularity time': time() - t_before,
            'Modularity score': bC.meridian_angle(bT),
            'GT latitude': bT.latitude(),
            'Modularity latitude': bC.latitude()
        }
        t_before = time()
        q = pv.wedges(G).latitude_on_meridian(np.pi / 2)
        C = pv.louvain_projection(q)
        bC = pv.clustering_binary(C)
        results_wedges = {
            'Wedges time': time() - t_before,
            'Wedges score': bC.meridian_angle(bT),
            'Wedges latitude': bC.latitude()
        }
        results[name].update(results_wedges)
        with open('lfr_results.json', 'w') as fp:
            json.dump(results, fp)
