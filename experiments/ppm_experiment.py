import numpy as np
from ..algorithms import pair_vector as pv
from ..random_graphs.generators import PPM
import pandas as pd
import matplotlib.pyplot as plt

'''
    Generates Figure 6 of the paper. To generate all four subfigures, simply run generate_all().
    perform_ppm_experiment performs the experiment and stores the data in a dataframe. The four subfigures are generated
    by
    6(a): ppm_experiment_plot_cluster_lats
    6(b): ppm_experiment_plot_angular_dists
    6(c): ppm_experiment_plot_correlation_dists
    6(d): ppm_experiment_plot_validation
'''


def format_angle(angle,digits=3):
    return '${angle:.{digits}f}\\pi$'.format(angle=angle/np.pi,digits=digits)

'''
    Performs the experiment for a Planted Partition Model (PPM) with n vertices, partitioned into k equally-sized
    ground-truth communities, with mean degree 10 and mixing rate 0.4. nlats denotes the number of query latitudes for
    which we compute the performance of both edge- and wedge-based vectors.
'''
def perform_ppm_experiment(n=400,k=20,mean_degree=10,mix=0.4,seed=0,nlats=33):
    ppm = PPM(n=n,k=k,mean_degree=mean_degree,mix=mix)
    G,T = ppm.generate(seed=seed)
    eG = pv.connectivity(G)
    bT = pv.clustering_binary(T)
    global bT_lat,dCC_eG_bT
    bT_lat = bT.latitude()
    dCC_eG_bT = eG.meridian_angle(bT)
    lats = np.linspace(0,np.pi,nlats)
    qs = [
        eG.latitude_on_meridian(lat)
        for lat in lats
    ]
    bCs = [
        pv.clustering_binary(pv.louvain_projection(q))
        for q in qs
    ]
    df=pd.DataFrame.from_dict({
        lat: {
            r'$d_a(q,b(C))$': bC.angular_distance(q),
            r'$d_a(q,b(T))$': bT.angular_distance(q),
            r'$d_a(b(T),b(C))$': bC.angular_distance(bT),
            r'$d_{CC}(q,b(C))$': bC.meridian_angle(q),
            r'$d_{CC}(e(G),b(T))$': bT.meridian_angle(eG),
            r'$d_{CC}(b(T),b(C))$': bC.meridian_angle(bT),
            r'$\ell(b(C))$': bC.latitude(),
            r'$\ell(b(T))$': bT.latitude(),
            r'$\ell(q)$': lat
        }
        for lat,bC,q in zip(lats,bCs,qs)
    },orient='index')
    return df


def ppm_experiment_plot_cluster_lats(df):
    ax=df.plot(y=[r'$\ell(b(C))$',r'$\ell(b(T))$'],style=['-','--'])
    intersect=df.index[(df[r'$\ell(b(C))$'] - bT_lat).abs().argmin()]
    ax.set_xlabel(r'Query latitude $\ell(q)$')
    ax.set_ylabel(r'Candidate latitude')
    plt.setp(ax,
             xticks=[0,np.pi / 4, intersect, 3*np.pi / 4,np.pi],
             xticklabels=[r'$0$',r'$\frac{1}{4}\pi$', format_angle(intersect, digits=2), r'$\frac{3}{4}\pi$', r'$\pi$'],
             yticks=[bT_lat, np.pi / 2,np.pi],
             yticklabels=[format_angle(bT_lat,2), r'$\frac{1}{2}\pi$',r'$\pi$'])
    ax.set_ylim(0,np.pi)
    ax.set_xlim(0,np.pi)
    plt.savefig('ppm_experiment_cluster_lats.svg', bbox_inches='tight')


def ppm_experiment_plot_angular_dists(df):
    ax=df.plot(y=[r'$d_a(q,b(C))$',r'$d_a(q,b(T))$',r'$\ell(q)$'],style=['-','-','--'])
    ax.set_xlabel(r'Query latitude $\ell(q)$')
    ax.set_ylabel(r'Angular distance')
    plt.setp(ax,
             xticks=[0,np.pi / 4, np.pi/2, 3*np.pi / 4,np.pi],
             xticklabels=[r'$0$',r'$\frac{1}{4}\pi$', r'$\frac{1}{2}\pi$', r'$\frac{3}{4}\pi$', r'$\pi$'],
             yticks=[bT_lat, np.pi/4, np.pi / 2,3*np.pi/4,np.pi],
             yticklabels=[r'$\ell(b(T))$', r'$\frac{1}{4}\pi$', r'$\frac{1}{2}\pi$', r'$\frac{3}{4}\pi$',r'$\pi$'])
    ax.set_ylim(0,np.pi)
    ax.set_xlim(0,np.pi)
    plt.savefig('ppm_experiment_angular_dists.svg', bbox_inches='tight')


def ppm_experiment_plot_correlation_dists(df):
    ax=df.plot(y=[r'$d_{CC}(q,b(C))$',r'$d_{CC}(e(G),b(T))$'],style=['-','--'])
    ax.set_xlabel(r'Query latitude $\ell(q)$')
    ax.set_ylabel(r'Correlation distance')
    plt.setp(ax,
             xticks=[0,np.pi / 4, np.pi/2, 3*np.pi / 4,np.pi],
             xticklabels=[r'$0$',r'$\frac{1}{4}\pi$', r'$\frac{1}{2}\pi$', r'$\frac{3}{4}\pi$', r'$\pi$'],
             yticks=[dCC_eG_bT, np.pi/4, np.pi / 2,3*np.pi/4,np.pi],
             yticklabels=[format_angle(dCC_eG_bT,digits=2), r'$\frac{1}{4}\pi$', r'$\frac{1}{2}\pi$', r'$\frac{3}{4}\pi$',r'$\pi$'])
    ax.set_ylim(np.pi/4,3*np.pi/4)
    ax.set_xlim(0,np.pi)
    plt.savefig('ppm_experiment_correlation_dists.svg', bbox_inches='tight')

def ppm_experiment_plot_validation(df,triangle_bounds=False):
    ax=df.plot(y=r'$d_{CC}(b(T),b(C))$',color=u'#ff7f0e')
    if triangle_bounds:
        ax.fill_between(
            lats,
            (df[r'$d_{CC}(q,b(C))$']-df[r'$d_{CC}(e(G),b(T))$']).abs(),
            df[r'$d_{CC}(q,b(C))$']+df[r'$d_{CC}(e(G),b(T))$'],
            label='Triangle inequality bounds',
            alpha=0.5)
    ax.set_xlabel(r'Query latitude $\ell(q)$')
    ax.set_ylabel(r'Correlation distance')
    dCC_min = df.index[df[r'$d_{CC}(b(T),b(C))$'].argmin()]
    da_min = df.index[df[r'$d_a(q,b(T))$'].argmin()]
    plt.setp(ax,
             xticks=[0,da_min,bT_lat, dCC_min, 3*np.pi / 4,np.pi],
             xticklabels=[r'$0$',r"$\lambda'$",r'$\ell(b(T))$', format_angle(dCC_min,digits=2), r'$\frac{3}{4}\pi$', r'$\pi$'],
             yticks=[np.pi/4, np.pi / 2,3*np.pi/4,np.pi],
             yticklabels=[r'$\frac{1}{4}\pi$', r'$\frac{1}{2}\pi$', r'$\frac{3}{4}\pi$',r'$\pi$'])
    ax.set_ylim(0,np.pi)
    ax.set_xlim(0,np.pi)
    ax.legend()
    plt.savefig('ppm_experiment_validation.svg', bbox_inches='tight')

def generate_figures():
    df = perform_ppm_experiment()
    ppm_experiment_plot_validation(df)
    ppm_experiment_plot_angular_dists(df)
    ppm_experiment_plot_correlation_dists(df)
    ppm_experiment_plot_cluster_lats(df)
    return df