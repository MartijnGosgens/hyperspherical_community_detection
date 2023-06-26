from statistics import median
import pandas as pd
from ..algorithms import pair_vector as pv
from ..random_graphs.generators import PPM, HeterogeneousSizedPPM, ABCD_benchmark, IndependentLFR


vec2mapping = {
    r'$q^{(MLE)}(G)$': lambda G, T: pv.query_PPM_MLE(G, T=T),
    r'$q^{(MLE)}_{DC}(G)$': lambda G, T: pv.query_CM_MLE(G, T=T),
    r'$q^*(e(G))$': lambda G, T: pv.do_heuristic(pv.connectivity(G), T=T),
    r'$q^*(q^{(MLE)}_{DC}(G))$': lambda G,T: pv.do_heuristic(pv.query_CM_MLE(G, T=T), T=T),
    r'$q^*(w(G))$': lambda G, T: pv.do_heuristic(pv.wedges(G), T=T),
    r'$q^*(j(G))$': lambda G, T: pv.do_heuristic(pv.jaccard(G), T=T),
}


def perform_experiment(n=400, k=20, mean_degree=8, mix=0.25, repeats=50):
    models = {
        'PPM': PPM(n=n, k=k, mean_degree=mean_degree, mix=mix),
        'HPPM': HeterogeneousSizedPPM(n=n, k=k, mean_degree=mean_degree, mix=mix),
        'DCPPM': IndependentLFR(n=n, k=k, mean_degree=mean_degree, mix=mix, balanced_sizes=True),
        'ABCD': ABCD_benchmark(n=n,min_deg=int(mean_degree/2),xi=mix,max_deg=100,max_size=300)
    }
    model2sample = {
        m_name: [model.generate(seed=seed) for seed in range(repeats)]
        for m_name, model in models.items()
    }

    model2vec2sample = {
        m_name: {
            v_name: [
                pv.clustering_binary(T).meridian_angle(pv.clustering_binary(pv.louvain_projection(mapping(G, T))))
                for (G, T) in sample
            ]
            for v_name, mapping in vec2mapping.items()
        }
        for m_name, sample in model2sample.items()
    }

    return {
        m_name: {
            v_name: median(scores)
            for v_name,scores in vec2scores.items()
        }
        for m_name,vec2scores in model2vec2sample.items()
    }


def generate_table(print_result=True, n=1000, k=50, mean_degree=8, mix=0.25, repeats=50, save_json=True):
    model2vec2median = perform_experiment(n=n, k=k, mean_degree=mean_degree, mix=mix, repeats=repeats)
    if save_json:
        import json
        filename = 'heuristic_comparison_n{}_k{}_deg{}_mix{:.2f}_repeats{}.json'.format(n, k, mean_degree, mix, repeats)
        with open(filename, 'w') as fp:
            json.dump(model2vec2median, fp)
    df = pd.DataFrame.from_dict(model2vec2median, orient='index')
    result = df.to_latex(float_format="%.4f", escape=False)
    if print_result:
        print(result)
    return result


def generate_figures():
    generate_table()