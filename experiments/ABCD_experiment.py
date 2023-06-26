import os
import numpy as np
from time import time
from collections import defaultdict
from statistics import median
from ..random_graphs.generators import ABCD_benchmark
from ..algorithms import pair_vector as pv


ns = [20000,50000]#1000,10000,100000]
vec2mapping = {
    #r'$q^{(MLE)}(G)$': lambda G, T: pv.query_PPM_MLE(G, T=T),
    r'$q^{(MLE)}_{DC}(G)$': lambda G, T: pv.query_CM_MLE(G, T=T),
    #r'$q^*(e(G))$': lambda G, T: pv.do_heuristic(pv.connectivity(G), T=T),
    r'$q^*(q^{(MLE)}_{DC}(G))$': lambda G,T: pv.do_heuristic(pv.query_CM_MLE(G, T=T), T=T),
    r'$q^*(w(G))$': lambda G, T: pv.do_heuristic(pv.wedges(G), T=T),
    r'$q^*(j(G))$': lambda G, T: pv.do_heuristic(pv.jaccard(G), T=T),
}
n2vec2time = defaultdict(dict)
n2vec2result = defaultdict(dict)
n2vec2lat_err = defaultdict(dict)
repeats = 1


for n in ns:
    generator = ABCD_benchmark(n=n)
    generated = False
    for vec,mapping in vec2mapping.items():
        # Skip wedges and jaccard for large graphs
        if n>20000 and (('w(G)' in vec) or ('j(G)' in vec)):
            n2vec2time[n][vec] = -1
            n2vec2result[n][vec] = -1
            n2vec2lat_err[n][vec] = -1
            print('skipping',vec)
            continue
        times = []
        results = []
        lat_err = []
        for seed in range(repeats):
            G,T = generator.generate(seed=seed,load=generated)
            start = time()
            print('starting',vec,seed)
            bC = pv.louvain_projection(mapping(G,T),return_vec=True)
            times.append(time()-start)
            bT = pv.clustering_binary(T)
            results.append(bC.meridian_angle(bT))
            lat_err.append(bC.latitude()/bT.latitude()-1)
        generated=True
        n2vec2time[n][vec] = median(times)
        n2vec2result[n][vec] = median(results)
        n2vec2lat_err[n][vec] = median(lat_err)
        print(f'n = {n} {vec} achieved dCC={median(results)} in {median(times)}s and latitude error {median(lat_err)}')
    print(n,'\t'.join(['${:.4f}$'.format(v) for v in n2vec2result[n].values()]))
    print(n,'\t'.join(['${:.1f}$'.format(v) for v in n2vec2time[n].values()]))

        
import pandas as pd
for d in (n2vec2time,n2vec2result,n2vec2lat_err):
    print(pd.DataFrame.from_dict(d, orient='index').to_latex(float_format="%.1f", escape=False))


