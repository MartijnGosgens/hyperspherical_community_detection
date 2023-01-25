import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..algorithms import pair_vector as pv
from ..Clustering import Clustering


name2latex = {
    'cordists': r'$d_{CC}(q^*,b(C))\ /\ d_{CC}(q^*,b(T))$',
    'angdists': r'$d_a(q^*,b(C))\ /\ d_a(q^*,b(T))$',
    'candilats': r'$\ell(b(C))\ /\ \ell(b(T))$'
}
names = name2latex.keys()



def perform_experiment(n=100, k=10, repeats=50):
    T = Clustering.BalancedClustering(n, k)
    bT = pv.clustering_binary(T)
    stds = np.linspace(0.5, 2, 7)
    std2qs = {
        std: [
            pv.do_heuristic(bT + (std / 2) * pv.noise_vector(vertices=T.keys()), bT=bT)
            for _ in range(repeats)
        ]
        for std in stds
    }
    print('Generated noise vectors. Will generate candidates now.')
    std2candidates = {
        std: [
            pv.clustering_binary(pv.louvain_projection(q))
            for q in qs
        ]
        for std, qs in std2qs.items()
    }
    print('Generated candidates')

    std2dCC = {
        std: np.array(list(map(bT.meridian_angle, qs))).mean()
        for std, qs in std2qs.items()
    }

    name2scorer = {
        'cordists': lambda q, bC: bC.meridian_angle(q)/bT.meridian_angle(q),
        'angdists': lambda q, bC: bC.angular_distance(q)/bT.angular_distance(q),
        'candilats': lambda q, bC: bC.latitude()/bT.latitude(),
    }

    format_label = lambda std, dCC: r'$\sigma=' + '{:.2f}$\n'.format(std) + r'$d_{CC}=' + '{:.2f}$'.format(dCC)
    name2df = {
        name: pd.DataFrame.from_dict({
            format_label(std,dCC): [
                scorer(q, bC)
                for q, bC in zip(std2qs[std], std2candidates[std])
            ]
            for std, dCC in std2dCC.items()
        }, orient='columns')
        for name, scorer in name2scorer.items()
    }
    return name2df


def plot_measure(name, df):
    plt.figure(figsize=(14, 3))
    ax = df.boxplot(column=list(df.columns))
    ax.set_title(name2latex[name])
    print('Plot',name)
    plt.savefig('noisy_vector_' + name + '.pdf', bbox_inches='tight')


def plot_experiment(n=100, k=10, repeats=50):
    name2df = perform_experiment(n=n, k=k, repeats=repeats)
    for name, df in name2df.items():
        plot_measure(name, df)


def generate_figures():
    plot_experiment()
