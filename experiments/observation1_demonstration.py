import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ..algorithms import pair_vector as pv
from ..random_graphs.generators import PPM, HeterogeneousSizedPPM, IndependentLFR

default_lats = np.linspace(0, np.pi / 2, 12)[1:]


def short2mathjax(desc, name='x'):
    if desc == 'l(b(T))':
        return r'$\ell(b(T))$'
    if desc == 'l(b(C))':
        return r'$\ell(\mathcal{L}(\mathcal{P}_\lambda(' + name + r')))$'
    if desc == 'd_a(b(T),b(C))':
        return r'$d_a(b(T),\mathcal{L}(\mathcal{P}_\lambda(' + name + r')))$'
    if desc == 'd_a(q,b(T))':
        return r'$d_a(\mathcal{P}_\lambda(' + name + r'),b(T))$'
    if desc == 'd_a(q,b(C))':
        return r'$d_a(\mathcal{P}_\lambda(' + name + r'),\mathcal{L}(\mathcal{P}_\lambda(' + name + r')))$'
    if desc == 'd_CC(b(T),b(C))':
        return r'$d_{CC}(b(T),\mathcal{L}(\mathcal{P}_\lambda(' + name + r')))$'
    if desc == 'd_CC(x,b(T))':
        return r'$d_{CC}(' + name + r',b(T))$'
    if desc == 'd_CC(x,b(C))':
        return r'$d_{CC}(' + name + r',\mathcal{L}(\mathcal{P}_\lambda(' + name + r')))$'
    if desc == 'lambda(x)':
        return r'$\lambda^*(' + name + ')$'
    if desc == 'l(b(C)) theo':
        return r'$\hat{\lambda}_C$'


def vary_lats(name2vec, bT, lats=default_lats):
    y_labels = {
        short2mathjax('d_CC(x,b(T))', name): vec.meridian_angle(bT)
        for name, vec in name2vec.items()
    }
    y_labels[short2mathjax('l(b(T))')] = bT.latitude()
    target_lats = [pv.heuristic_latitude(bT.latitude(), x.meridian_angle(bT)) for x in name2vec.values()]
    x_labels = {
        short2mathjax('lambda(x)', name): t_lat
        for name, t_lat in zip(name2vec.keys(), target_lats)
    }
    lats_sorted = np.concatenate([lats, target_lats])
    lats_sorted.sort()
    lat2name2query = {
        lat: {
            name: vec.latitude_on_meridian(lat)
            for name, vec in name2vec.items()
        }
        for lat in lats_sorted
    }
    lat2name2candidate = {
        lat: {
            name: pv.clustering_binary(pv.louvain_projection(q))
            for name, q in lat2name2query[lat].items()
        }
        for lat in lats_sorted
    }
    df = pd.DataFrame(index=lats_sorted)
    df.index.name = r'$\lambda$'
    for name in name2vec.keys():
        df[short2mathjax('l(b(C))', name)] = [
            name2candidate[name].latitude()
            for name2candidate in lat2name2candidate.values()
        ]
        df[short2mathjax('d_a(b(T),b(C))', name)] = [
            name2candidate[name].angular_distance(bT)
            for name2candidate in lat2name2candidate.values()
        ]
        df[short2mathjax('d_CC(b(T),b(C))', name)] = [
            name2candidate[name].meridian_angle(bT)
            for name2candidate in lat2name2candidate.values()
        ]
        df[short2mathjax('d_a(q,b(T))', name)] = [
            name2query[name].angular_distance(bT)
            for name2query in lat2name2query.values()
        ]
        df[short2mathjax('d_a(q,b(C))', name)] = [
            name2query[name].angular_distance(name2candidate[name])
            for name2query, name2candidate in zip(lat2name2query.values(), lat2name2candidate.values())
        ]
        df[short2mathjax('d_CC(x,b(C))', name)] = [
            name2query[name].meridian_angle(name2candidate[name])
            for name2query, name2candidate in zip(lat2name2query.values(), lat2name2candidate.values())
        ]
        theta = y_labels[short2mathjax('d_CC(x,b(T))', name)]
        df[short2mathjax('l(b(C)) theo', name)] = [
            np.arccos(np.cos(theta) * np.cos(lat) / (1 - np.sin(theta) * np.sin(lat)))
            for (lat, name2query), name2candidate in zip(lat2name2query.items(), lat2name2candidate.values())
        ]
    return df, x_labels, y_labels


def plot_performance(vecname, df, x_labels, y_labels):
    cols = [short2mathjax(desc, vecname) for desc in ['d_CC(b(T),b(C))']]
    rename_cols = [short2mathjax(desc) for desc in ['d_CC(b(T),b(C))']]
    lambdaX = short2mathjax('lambda(x)', vecname)
    lambdaX_rename = short2mathjax('lambda(x)')
    x_annotations = {
        lambdaX_rename: x_labels[lambdaX]
    }
    df = df[cols].rename(dict(zip(cols, rename_cols), axis='columns'))
    ax = plot_lats(df, x_labels=x_annotations)
    ax.set_title('$x={}$'.format(vecname))


def plot_candilats(vecname, df, x_labels, y_labels):
    cols = [short2mathjax(desc, vecname) for desc in ['l(b(C))']]
    rename_cols = [short2mathjax(desc) for desc in ['l(b(C))']]
    lambdaX = short2mathjax('lambda(x)', vecname)
    lambdaX_rename = short2mathjax('lambda(x)')
    x_annotations = {
        lambdaX_rename: x_labels[lambdaX]
    }
    lbT = short2mathjax('l(b(T))', vecname)
    y_annotations = {
        lbT: y_labels[lbT]
    }
    markers = ([x_labels[lambdaX]], [y_labels[lbT]])
    df = df[cols].rename(dict(zip(cols, rename_cols)), axis='columns')
    ax = plot_lats(df, x_labels=x_annotations, y_labels=y_annotations, markers=markers)
    ax.set_title('$x={}$'.format(vecname))


def plot_validation(vecname, df, x_labels, y_labels):
    cols = [short2mathjax(desc, vecname) for desc in ['d_a(q,b(C))', 'd_CC(x,b(C))']]
    rename_cols = [short2mathjax(desc) for desc in ['d_a(q,b(C))', 'd_CC(x,b(C))']]
    lambdaX = short2mathjax('lambda(x)', vecname)
    lambdaX_rename = short2mathjax('lambda(x)')
    x_annotations = {
        lambdaX_rename: x_labels[lambdaX]
    }
    dCC = short2mathjax('d_CC(x,b(T))', vecname)
    dCC_rename = short2mathjax('d_CC(x,b(T))')
    y_annotations = {
        dCC_rename: y_labels[dCC]
    }
    markers = ([x_labels[lambdaX]], [y_labels[dCC]])
    df = df[cols].rename(dict(zip(cols, rename_cols)), axis='columns')
    ax = plot_lats(df, x_labels=x_annotations, y_labels=y_annotations, markers=markers)
    ax.set_title('$x={}$'.format(vecname))


def plot_lats(df, x_labels={}, y_labels={}, markers=(), ax=None, **kwargs):
    # print(kwargs)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    df.plot(ax=ax, **kwargs)
    plt.yticks(
        [np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi] + list(y_labels.values()),
        [r'$\frac{1}{4}\pi$', r'$\frac{1}{2}\pi$', r'$\frac{3}{4}\pi$', r'$\pi$'] + list(y_labels.keys()),
        rotation=45
    )
    plt.xticks(
        [0, np.pi / 4, np.pi / 2] + list(x_labels.values()),
        [r'$0$', r'$\frac{1}{4}\pi$', r'$\frac{1}{2}\pi$'] + list(x_labels.keys())
    )
    ax.set_xlim(0, max(df.index))
    ax.set_ylim(0, np.pi)
    if len(markers) > 0:
        ax.plot(*markers, marker='x', color='red', markersize=12)
    return ax


def plot_all(n=100,
             mean_degree=8,
             exp_sizes=2.5,
             mixs=[0.5],
             ks=[10],
             vec_names=['e(G)','r(T)'],
             model_names=['balanced_PPM']):
    dfs = []
    models = {}
    if 'balanced_PPM' in model_names:
        models['balanced_PPM'] = {
            (k, mix): PPM(n=n, k=k, mean_degree=mean_degree, mix=mix)
            for k in ks
            for mix in mixs
        }
    if 'DCPPM' in model_names:
        models['DCPPM'] = {
            (k,mix): IndependentLFR(n=n,k=k,mean_degree=mean_degree,mix=mix,balanced_sizes=True)
            for k in ks
            for mix in mixs
         }
    if 'HPPM' in model_names:
        models['HPPM'] = {
            (k,mix): HeterogeneousSizedPPM(n=n,k=k,mean_degree=mean_degree,mix=mix,exp_sizes=exp_sizes)
            for k in ks
            for mix in mixs
        }
    vec_format = lambda s: s.replace('(', '').replace(')', '')
    for model_name, params2model in models.items():
        for (k, mix), model in params2model.items():
            G, T = model.generate(seed=0)
            name2vec = {}
            if 'e(G)' in vec_names:
                name2vec['e(G)'] = pv.connectivity(G)
            if 'w(G)' in vec_names:
                name2vec['e(G)'] = pv.wedges(G)
            if 'r(T)' in vec_names:
                bT = pv.clustering_binary(T)
                name2vec['r(T)'] = bT + 0.75 * pv.noise_vector(vertices=G.nodes)

            for vec_name, vec in name2vec.items():
                print(vec_name, 'has dCC', vec.meridian_angle(bT))

            df, x_labels, y_labels = vary_lats(name2vec, bT)
            dfs.append(df)
            for x_name in vec_names:
                file_prefix = '{}_n{}_k{}_degree{}_mix{:.2f}_{}_'.format(
                    model_name,
                    len(G),
                    k,
                    mean_degree,
                    mix,
                    vec_format(x_name)
                )
                for plotter_name, plotter in zip(
                        ['candilats', 'performance', 'validation'],
                        [plot_candilats, plot_performance, plot_validation]
                ):
                    plotter(x_name, df, x_labels, y_labels)
                    plt.savefig(file_prefix + 'lats_' + plotter_name + '.pdf', bbox_inches='tight')


def generate_figures():
    plot_all()
