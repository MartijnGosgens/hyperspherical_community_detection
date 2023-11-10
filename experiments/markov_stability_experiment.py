import pandas as pd
from os import path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from time import time
import pandas as pd
from ..random_graphs.generators import ABCD_benchmark,PPM,HeterogeneousSizedPPM,IndependentLFR
from ..algorithms import pair_vector as pv
repeats = 50
n=1000
output_dir = path.join(path.dirname(__file__), 'output')
m_names = ['PPM','HPPM','DCPPM','ABCD']

def perform_experiment():
    k=50
    mean_degree=8 
    mix=0.25
    ABCD_seeds = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99]
    models = {
        'PPM': PPM(n=n, k=k, mean_degree=mean_degree, mix=mix),
        'HPPM': HeterogeneousSizedPPM(n=n, k=k, mean_degree=mean_degree, mix=mix),
        'DCPPM': IndependentLFR(n=n, k=k, mean_degree=mean_degree, mix=mix, balanced_sizes=True),
        'ABCD': ABCD_benchmark(n=n,min_deg=int(mean_degree/2),xi=mix,max_deg=100,max_size=300)
    }
    columns = ['model','seed','$\\ell(b(T))$','query mapping','corrected','timestep',
            '$\\ell(q)$','$\\ell(b(C))$','$d_a(q,b(T))$','$d_{CC}(q,b(T))$','$d_a(q,b(C))$','$d_{CC}(q,b(C))$', '$d_{CC}(b(C),b(T))$' ,'time']
    data = []
    for m_name,gen in models.items():
        for seed in range(repeats):
            kwargs = {'load': True} if m_name=='ABCD' else {}
            G,T=gen.generate(seed=seed if m_name!='ABCD' else ABCD_seeds[seed],**kwargs)
            bT = pv.clustering_binary(T)
            graph_columns = [m_name,seed,bT.latitude()]
            for timestep in range(1,6):
                q=pv.discrete_markov_stability(G,timestep=timestep)
                q_h=pv.do_heuristic(pv.discrete_markov_stability(G,timestep=timestep),T=T)
                q_desc = [f'$q^{{(MS)}}_{{t={timestep}}}$',False]
                q_desc_h = [f'$q^*(q^{{(MS)}}_{{t={timestep}}})$',True]
                for q,desc in [(q,q_desc),(q_h,q_desc_h)]:
                    start = time()
                    bC = pv.louvain_projection(q,return_vec=True)
                    comptime = time()-start
                    data.append(graph_columns+desc+[
                        timestep,
                        q.latitude(),
                        bC.latitude(),
                        q.angular_distance(bT),
                        q.meridian_angle(bT),
                        q.angular_distance(bC),
                        q.meridian_angle(bC),
                        bT.meridian_angle(bC),
                        comptime
                    ])
        df = pd.DataFrame(data,columns=columns)
        df.to_csv(path.join(output_dir,f'markov_stability_benchmark_n{n}_repeats{repeats}_{m_name}.csv'))

def generate_boxplots():
    for m_name in ['PPM','HPPM','DCPPM','ABCD']:
        df = pd.read_csv(path.join(output_dir,f'markov_stability_benchmark_n{n}_repeats{repeats}_{m_name}.csv'),index_col='Unnamed: 0')
        df['Relative latitude error'] = df['$\\ell(b(C))$']/df['$\\ell(b(T))$']-1
        ax=df.boxplot(column='Relative latitude error',by=['timestep','corrected'],figsize=(10,3))
        ax.set_xticklabels([
            f'$q^*(q^{{(MS)}}_{{t={timestep}}})$' if corrected else f'$q^{{(MS)}}_{{t={timestep}}}$'
            for timestep in range(1,6)
            for corrected in [False,True]
        ])
        ax.get_figure().suptitle(None)
        ax.set_title('Relative granularity error')
        ax.set_xlabel('Query mapping')
        ax.set_ylabel('$\ell(b(C))/\ell(b(T))-1$')
        plt.savefig(f'markov_stability_benchmark_n{n}_repeats{repeats}_lat_boxplots_{m_name}.jpg',bbox_inches='tight')


        df['$\\rho(C,T)$'] = np.cos(df['$d_{CC}(b(C),b(T))$'])
        ax=df.boxplot(column='$\\rho(C,T)$',by=['timestep','corrected'],figsize=(10,3))
        ax.set_xticklabels([
            f'$q^*(q^{{(MS)}}_{{t={timestep}}})$' if corrected else f'$q^{{(MS)}}_{{t={timestep}}}$'
            for timestep in range(1,6)
            for corrected in [False,True]
        ])
        ax.get_figure().suptitle(None)
        ax.set_title('Similarity between detected and planted clustering')
        ax.set_xlabel('Query mapping')
        ax.set_ylabel('$\\rho(C,T)$')
        plt.savefig(f'markov_stability_benchmark_n{n}_repeats{repeats}_CC_boxplots_{m_name}.jpg',bbox_inches='tight')

def generate_figures():
    generate_boxplots()