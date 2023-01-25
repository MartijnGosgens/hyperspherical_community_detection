import numpy as np
import matplotlib.pyplot as plt
from ..algorithms import pair_vector as pv
from ..algorithms.lat2sizes import lat2sizes
from ..Clustering import Clustering


def perform_experiment(theta_target=np.pi/3, n=100, res=30):
    bTlats = np.linspace(0, np.pi, res)[1:-1]
    bTs = [pv.clustering_binary(Clustering.FromSizes(lat2sizes(n=n, lat=lat))) for lat in bTlats]
    candilats = []
    lats_plus = []
    lats_min = []
    for bT in bTs:
        c = 1
        r = pv.noise_vector(vertices=range(n))
        theta = bT.meridian_angle(bT + c * r)
        step = 0.5
        theta_prev = 0
        while abs(theta - theta_target) > 0.01:
            c += step if theta < theta_target else -step
            theta_prev = theta
            theta = bT.meridian_angle(bT + c * r)
            if theta_prev < theta_target < theta or theta_prev > theta_target > theta:
                step /= 2
        print(c, theta)
        x = bT + c * r
        q = pv.do_heuristic(x, bT=bT)
        qlat = q.latitude()
        lats_plus.append(np.arccos(np.cos(theta) * np.cos(qlat) / (1 + np.sin(theta) * np.sin(qlat))))
        lats_min.append(np.arccos(np.cos(theta) * np.cos(qlat) / (1 - np.sin(theta) * np.sin(qlat))))
        candilats.append(pv.clustering_binary(pv.louvain_projection(q)).latitude())
    return bTlats, lats_plus, lats_min, candilats


def plot_experiment(theta_target=np.pi/3, n=100):
    bTlats, lats_plus, lats_min, candilats = perform_experiment(theta_target=theta_target, n=n)
    plt.figure(figsize=(7, 4))
    plt.plot(bTlats, np.array(lats_plus) / np.array(candilats), label=r'$\lambda_+\ /\ \ell(\mathcal{L}(q^*))$')
    plt.plot(bTlats, np.array(lats_min) / np.array(candilats), label=r'$\lambda_-\ /\ \ell(\mathcal{L}(q^*))$')

    plt.xticks(
        [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
        [r'$0$', r'$\frac{1}{4}\pi$', r'$\frac{1}{2}\pi$', r'$\frac{3}{4}\pi$', r'$\pi$']
    )
    plt.legend()
    plt.xlabel(r'$\ell(b(T))$')
    plt.ylabel('Relative error')
    plt.savefig('plus_v_min_experiment_n{}_theta{:.2f}.pdf'.format(n, theta_target), bbox_inches='tight')

def generate_figures():
    plot_experiment()