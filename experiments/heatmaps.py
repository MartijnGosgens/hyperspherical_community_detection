from ..algorithms import pair_vector as pv
import numpy as np
import matplotlib.pyplot as plt
from .benchmarknetworks import load_dataset

'''
    This file contains the implementations of the heatmap figures from Figures 3, 4 and 8.
    Call generate_figures() to generate all three figures.
'''

# Resolution of the jpg figures. The figures are too large when saving in the svg format.
dpi=400

'''
    Computes the heatmap values for the graph G with ground truth T. The grid will have dimensions nx*ny, with bounds 
    defined by resmin, resmax (minimum/maximum values of the resolution parameter) and latmin and latmax 
    (minimal/maximal latitudes).
    The output is a tuple containing all the parameters that are needed for the function plot_heatmap below.
    That is, to generate the heatmap directly, simply run plot_heatmap(*compute_heatmap(...)).
'''
def compute_heatmap(G,T,resmin=-1.5,resmax=3,latmin=np.pi/3,latmax=2*np.pi/3,nx=46,ny=46):
    bT = pv.clustering_binary(T)
    eG = pv.connectivity(G)

    ress = list(np.linspace(resmax,resmin,nx))
    q_CMs = [
        pv.query_CM(G,res=res)
        for res in ress
    ]
    lats = list(np.linspace(min(latmin,q_CMs[0].latitude()),max(latmax,q_CMs[-1].latitude()),ny))
    dCCs = {
        res: q.meridian_angle(bT)
        for res,q in zip(ress,q_CMs)
    }
    dCC2s = {
        res: np.sign(res)*q.meridian_angle(eG)
        for res,q in zip(ress,q_CMs)
    }
    best_res = min(dCCs, key=dCCs.get)
    q_lats = [q.latitude() for q in q_CMs]

    lats_mesh, cd_mesh = np.meshgrid(lats,list(dCC2s.values()))
    cors = [
        [
            np.cos(pv.clustering_binary(
                pv.louvain_projection(q.latitude_on_meridian(lat),silent=True)).meridian_angle(bT))
            for lat in lats
        ]
        for q in q_CMs
    ]
    return (
        cd_mesh,
        lats_mesh,
        cors,
        [cds[0] for cds in cd_mesh],
        dCC2s[best_res],
        q_lats
    )

'''
    Plots the heatmap and returns the axes. The required input parameters are the outputs of compute_heatmap.
    When plot_CM_meridian==True, the function will additionally plot the meridian that minimizes the correlation
    distance to b(T).
'''
def plot_heatmap(cd_mesh, lats_mesh, cors, dCC2s, dCC2_best, q_lats,plot_CM_meridian=False):
    fig, ax = plt.subplots(1, 1)
    c = ax.pcolormesh(cd_mesh, lats_mesh, cors, cmap='RdYlGn', vmin=0, vmax=1,shading='auto')
    plt.colorbar(c,ax=ax)
    ax.plot(dCC2s, q_lats, linestyle='dotted')
    ax.axvline(x=0, linestyle='dotted')
    if plot_CM_meridian:
        ax.axvline(x=dCC2_best, linestyle='dotted')
    ax.set_xlabel(r'(Signed) correlation distance to ER meridian')
    ax.set_ylabel(r'Query latitude $\ell(q)$')
    return ax


def plot_karate_heatmap(gridsize=91,save=True,plot_CM_meridian=False):
    G, T = load_dataset('karate')
    cd_mesh, lats_mesh, cors, dCC2s, dCC2_best, q_lats = compute_heatmap(G, T, nx=gridsize, ny=gridsize)
    dCC2s = [cds[0] for cds in cd_mesh]

    ax=plot_heatmap(cd_mesh, lats_mesh, cors, dCC2s, dCC2_best, q_lats, plot_CM_meridian=plot_CM_meridian)
    plt.setp(ax,
             xticks=[-np.pi / 8, 0, np.pi / 8, np.pi / 4, 3 * np.pi / 8],
             xticklabels=[r'$-\frac{1}{8}\pi$', r'$0$', r'$\frac{1}{8}\pi$', r'$\frac{1}{4}\pi$', r'$\frac{3}{8}\pi$'],
             yticks=[np.pi / 3, np.pi / 2, 2 * np.pi / 3],
             yticklabels=[r'$\frac{1}{3}\pi$', r'$\frac{1}{2}\pi$', r'$\frac{2}{3}\pi$'])
    ax.text(0, np.pi / 2, 'ER-modularity', {'ha': 'right', 'va': 'top'}, rotation=90)
    if plot_CM_meridian:
        ax.text(dCC2_best, np.pi / 2, "'Nearest' meridian", {'ha': 'right', 'va': 'top'}, rotation=90)
    ax.text(np.pi / 4, np.pi / 2, 'CM-modularity', {'ha': 'center', 'va': 'top'}, rotation=-42)
    if save:
        ax.get_figure().savefig('karate_heatmap.jpg', bbox_inches='tight',dpi=dpi)

def plot_dolphins_heatmap(gridsize=91,save=True,plot_CM_meridian=False):
    G, T = load_dataset('dolphins')
    cd_mesh, lats_mesh, cors, dCC2s, dCC2_best, q_lats = compute_heatmap(G, T, nx=gridsize, ny=gridsize)

    ax=plot_heatmap(cd_mesh, lats_mesh, cors, dCC2s, dCC2_best, q_lats, plot_CM_meridian=plot_CM_meridian)
    plt.setp(ax,
             xticks=[-np.pi / 16, 0, np.pi / 16, np.pi / 8, 3 * np.pi / 16],
             xticklabels=[r'$-\frac{1}{16}\pi$', r'$0$', r'$\frac{1}{16}\pi$', r'$\frac{1}{8}\pi$', r'$\frac{3}{16}\pi$'],
             yticks=[np.pi / 3, np.pi / 2, 2 * np.pi / 3],
             yticklabels=[r'$\frac{1}{3}\pi$', r'$\frac{1}{2}\pi$', r'$\frac{2}{3}\pi$'])
    ax.text(0, np.pi / 2, 'ER-modularity', {'ha': 'right', 'va': 'top'}, rotation=90)
    if plot_CM_meridian:
        ax.text(dCC2_best, np.pi / 2, "'Nearest' meridian", {'ha': 'left', 'va': 'top'}, rotation=-90)
    ax.text(np.pi / 7.5, np.pi / 2, 'CM-modularity', {'ha': 'center', 'va': 'top'}, rotation=-42)
    if save:
        ax.get_figure().savefig('dolphins_heatmap.jpg', bbox_inches='tight',dpi=dpi)

def plot_football_heatmap(gridsize=81,save=True,plot_CM_meridian=False):
    G, T = load_dataset('football')
    cd_mesh, lats_mesh, cors, dCC2s, dCC2_best, q_lats = compute_heatmap(G, T, nx=gridsize, ny=gridsize,resmin=-1,resmax=14,latmin=np.pi,latmax=0)

    ax=plot_heatmap(cd_mesh, lats_mesh, cors, dCC2s, dCC2_best, q_lats, plot_CM_meridian=plot_CM_meridian)
    plt.setp(ax,
             xticks=[0,np.pi/16,np.pi / 8],
             xticklabels=[r'$0$', r'$\frac{1}{16}\pi$', r'$\frac{1}{8}\pi$'],
             yticks=[np.pi/6,np.pi / 3, np.pi / 2, 2 * np.pi / 3],
             yticklabels=[r'$\frac{1}{6}\pi$',r'$\frac{1}{3}\pi$', r'$\frac{1}{2}\pi$', r'$\frac{2}{3}\pi$'])
    ax.text(0, np.pi / 2, 'ER-modularity', {'ha': 'right', 'va': 'top'}, rotation=90)
    if plot_CM_meridian:
        ax.text(dCC2_best, np.pi / 2, "'Nearest' meridian", {'ha': 'left', 'va': 'top'}, rotation=-90)
    ax.text(np.pi / 34, np.pi / 2, 'CM-modularity', {'ha': 'center', 'va': 'top'}, rotation=-60)
    if save:
        ax.get_figure().savefig('football_heatmap.jpg', bbox_inches='tight',dpi=dpi)

def generate_figures():
    plot_karate_heatmap()
    plot_football_heatmap()
    plot_dolphins_heatmap()