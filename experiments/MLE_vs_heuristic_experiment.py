from statistics import median
import matplotlib.pyplot as plt
import pandas as pd
from ..algorithms import pair_vector as pv
from ..random_graphs.generators import PPM


name2mapping = {
    r'$q^*(e(G))$': lambda G, T: pv.do_heuristic(pv.connectivity(G), T=T),
    r'$q^{(MLE)}(G)$': pv.query_PPM_MLE,
    r'$q^{(Mod)}(G)$': lambda G, T: pv.query_ER(G)
}


def perform_experiment(n=400, k=20, mean_degree=8, repeats=50, mixs=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65]):
    vec2mix2lats = {
        vec_name: {
            mix: []
            for mix in mixs
        }
        for vec_name in name2mapping.keys()
    }
    vec2mix2scores = {
        vec_name: {
            mix: []
            for mix in mixs
        }
        for vec_name in name2mapping.keys()
    }
    for mix in mixs:
        model = PPM(n=n, k=k, mean_degree=mean_degree, mix=mix)
        sample = [model.generate(seed=seed) for seed in range(repeats)]
        bTs = [
            pv.clustering_binary(T)
            for (_, T) in sample
        ]
        vec2qs = {
            vec_name: [
                mapping(G, T=T)
                for (G, T) in sample
            ]
            for vec_name, mapping in name2mapping.items()
        }
        vec2bCs = {
            vec_name: [
                pv.clustering_binary(pv.louvain_projection(q))
                for q in qs
            ]
            for vec_name, qs in vec2qs.items()
        }
        for vec_name, bCs in vec2bCs.items():
            vec2mix2lats[vec_name][mix] = [bC.latitude() / bT.latitude() for bC, bT in zip(bCs, bTs)]
            vec2mix2scores[vec_name][mix] = [bT.meridian_angle(bC) for bC, bT in zip(bCs, bTs)]

    lats_dict = {
        vec_name: {
            mix: median(lats)
            for mix, lats in mix2lats.items()
        }
        for vec_name, mix2lats in vec2mix2lats.items()
    }
    scores_dict = {
        vec_name: {
            mix: median(scores)
            for mix, scores in mix2scores.items()
        }
        for vec_name, mix2scores in vec2mix2scores.items()
    }
    return lats_dict, scores_dict


def plot_experiment(n=400, k=20, mean_degree=8, repeats=50, mixs=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65]):
    lats_dict, scores_dict = perform_experiment(n=n, k=k, mean_degree=mean_degree, repeats=repeats, mixs=mixs)
    fig, ax = plt.subplots(figsize=(6, 3.75))
    df_lats = pd.DataFrame.from_dict(lats_dict, orient='columns')
    df_lats.plot(ax=ax)
    ax.set_xlabel('Fraction inter-community edges')
    ax.set_ylabel(r'Median $\ell(b(C))\ /\ \ell(b(T))$')
    ax.legend()
    plt.savefig('MLE_heuristic_comparison_lats.pdf', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(6, 3.75))
    df_scores = pd.DataFrame.from_dict(scores_dict, orient='columns')
    df_scores.plot(ax=ax)
    ax.set_xlabel('Fraction inter-community edges')
    ax.set_ylabel(r'Median $d_{CC}(b(C),b(T))$')
    ax.legend()
    plt.savefig('MLE_heuristic_comparison_scores.pdf', bbox_inches='tight')


def generate_figures():
    plot_experiment()
