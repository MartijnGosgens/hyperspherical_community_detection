import numpy as np
from time import time
import numpy as np
import pandas as pd
import itertools as it
from ..algorithms import pair_vector as pv
from ..random_graphs.generators import ABCD_benchmark
import matplotlib.pyplot as plt
from os import path
output_dir = path.join(path.dirname(__file__),'output')

n=1000
k=50
ntrain=15
mean_degree=8 
mix=0.25
m_name = 'ABCD'
repeats = 50
gen = ABCD_benchmark(n=n,min_deg=int(mean_degree/2),xi=mix,max_deg=100,max_size=300)
# With the above parameters, the ABCD generator does not always succeed in generating a graph. Below are the seeds for which it did succeed.
ABCD_seeds = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99]

columns = ['model','seed','$\\ell(b(T))$',
           'c_jac','c_deg',
           '$\\ell(q)$','$\\ell(b(C))$','$d_a(q,b(T))$','$d_{CC}(q,b(T))$','$d_a(q,b(C))$','$d_{CC}(q,b(C))$', '$d_{CC}(b(C),b(T))$' ,'time']
c_adjs=1
c_jacs = np.linspace(0,1,11)
c_degs = np.linspace(-6,0,13)
coefss = list(it.product(c_jacs,c_degs))

def train():
    data = []
    for seed in range(repeats,repeats+ntrain):
        kwargs = {'load': True} if m_name=='ABCD' else {}
        G,T=gen.generate(seed=seed if m_name!='ABCD' else ABCD_seeds[seed],**kwargs)
        bT = pv.clustering_binary(T)
        graph_columns = [m_name,seed,bT.latitude()]

        for c_jac,c_deg in coefss:
            q=pv.do_heuristic(
                pv.connectivity(G)
                +(c_jac*pv.jaccard(G) if c_jac!=0 else pv.zeros)
                +(c_deg*pv.degree_product(G,normalized=True) if c_deg!=0 else pv.zeros),
                T=T)
            start = time()
            bC = pv.louvain_projection(q,return_vec=True)
            comptime = time()-start
            data.append(graph_columns+[c_jac,c_deg]+[
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
    df.to_csv(path.join(output_dir,f'gridsearch_n{n}_ntrain{ntrain}.csv'))


def plot_heatmap():
    df=pd.read_csv(path.join(output_dir,f'gridsearch_n{n}_ntrain{ntrain}.csv'))
    c_jacs = df['c_jac'].unique()
    degs_mesh, jacs_mesh = np.meshgrid(c_degs,c_jacs)
    df['$\\rho(C,T)$'] = np.cos(df['$d_{CC}(b(C),b(T))$'])
    grouped = df.groupby(by=['c_jac','c_deg'])['$\\rho(C,T)$'].median()
    grouped_avg = df.groupby(by=['c_jac','c_deg'])['$\\rho(C,T)$'].mean()
    medians = [
        [
            grouped[c_jac,c_deg]
            for c_deg in c_degs
        ]
        for c_jac in c_jacs
    ]
    fig, ax = plt.subplots(1, 1)
    c = ax.pcolormesh(jacs_mesh, degs_mesh, medians, cmap='RdYlGn', vmin=0.5, vmax=1,shading='auto')
    plt.colorbar(c,ax=ax,label='Median similarity $\\rho(C,T)$')
    ax.set_xlabel(r'Jaccard coefficient $c_j$')
    ax.set_ylabel(r'Degree-product coefficient $c_d$')
    maxcor = grouped.max()
    bests = [
        (c_j,c_d) for (c_j,c_d),v in grouped.items() if v==maxcor
    ]
    best=max(bests,key=lambda v: grouped_avg[v])

    ax.scatter([best[0]],[best[1]],marker='^',c='white')
    plt.savefig(f'training_performance_n{n}_ntrain{ntrain}_{m_name}.jpg',bbox_inches='tight')
    return best

def validate(c_jac,c_deg):
    data = []
    for seed in range(50):
        kwargs = {'load': True} if m_name=='ABCD' else {}
        G,T=gen.generate(seed=seed if m_name!='ABCD' else ABCD_seeds[seed],**kwargs)
        bT = pv.clustering_binary(T)
        
        q=pv.do_heuristic(
            pv.connectivity(G)
            +(c_jac*pv.jaccard(G) if c_jac!=0 else pv.zeros)
            +(c_deg*pv.degree_product(G,normalized=True) if c_deg!=0 else pv.zeros),
            T=T
        )
        start = time()
        bC = pv.louvain_projection(q,return_vec=True)
        comptime = time()-start
        data.append([
            m_name,
            seed,
            bT.latitude(),
            q.latitude(),
            bC.latitude(),
            q.angular_distance(bT),
            q.meridian_angle(bT),
            q.angular_distance(bC),
            q.meridian_angle(bC),
            bT.meridian_angle(bC),
            comptime
        ])
        print(dict(zip(columns,data[-1])))

    df = pd.DataFrame(data,columns=columns)
    df.to_csv(path.join(output_dir,f'gridsearch_validation_n{n}_{m_name}.csv'))
    print('Median performance',np.cos(df['$d_{CC}(b(C),b(T))$']).median())

def generate_figures():
    plot_heatmap()