from ..algorithms import pair_vector as pv
from ..random_graphs.generators import PPM
import matplotlib.pyplot as plt
import numpy as np

'''
    Plots Figure 5 of the paper. nlats is the number of latitudes between pi/8 and 5*pi/8.
'''
def plot_wedges_comparison(nlats=41,save=True):
    ppm=PPM(n=1000,k=10,mix=0.5,mean_degree=10)
    G,T = ppm.generate(seed=0)
    eG=pv.connectivity(G)
    wG=pv.wedges(G)
    bT=pv.clustering_binary(T)
    lats = np.linspace(np.pi/8,5*np.pi/8,nlats)
    q_e,q_w=([x.latitude_on_meridian(l) for l in lats] for x in (eG,wG))
    bC_e,bC_w = ([pv.clustering_binary(pv.louvain_projection(q)) for q in qs] for qs in (q_e,q_w))
    dCCs_e,dCCs_w = ([bC.meridian_angle(bT) for bC in bCs] for bCs in [bC_e,bC_w])

    fig, ax = plt.subplots(1, 1)
    ax.plot(lats,dCCs_e,label=r'$\mathcal{P}_\lambda(e(G))$')
    ax.plot(lats,dCCs_w,label=r'$\mathcal{P}_\lambda(w(G))$')
    plt.setp(ax,
             xticks=[np.pi/8, np.pi/4, 3*np.pi/8, np.pi / 2, 5 * np.pi / 8],
             xticklabels=[r'$\frac{1}{8}\pi$',r'$\frac{1}{4}\pi$',r'$\frac{3}{8}\pi$',r'$\frac{1}{2}\pi$', r'$\frac{5}{8}\pi$'],
             yticks=[np.pi/2, min(dCCs_w), min(dCCs_e)],
             yticklabels=[r'$\frac{1}{2}\pi$','${}\\pi$'.format(str(round(min(dCCs_w)/np.pi,2))),'${}\\pi$'.format(str(round(min(dCCs_e)/np.pi,2)))]
             )
    ax.set_xlabel(r'Query latitude $\lambda$')
    ax.set_ylabel(r'$d_{CC}(b(T),\mathcal{L}(q))$')
    ax.legend()
    if save:
        ax.get_figure().savefig('wedges_experiment.svg',bbox_inches='tight')
    return ax

def generate_figures():
    plot_wedges_comparison()