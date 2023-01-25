import numpy as np
"""
    A tool for generating clusterings of specified latitude (or sum of squared sizes). The clusterings are generated
    such that their sizes are (in a way) maximally balanced. For example, if the effective number of clusters (k_eff) is
    an integer, the resulting clustering will consist of equally-sized clusters.
"""


def k_eff(lat, n):
    """
        Effective number of clusters of a clustering with latitude lat
    """
    return n/(1+(n-1)*np.sin(lat/2)**2)


def clustering_lat(C):
    """
        The latitude of a clustering
    """
    m = C.intra_pairs()
    N = len(C)*(len(C)-1)
    return np.arccos(1-2*m/N)


def sizes_lat(sizes):
    """
        The latitude of a clustering with given sizes
    """
    n = sum(sizes)
    ss = (np.array(sizes)**2).sum()
    return np.arccos((n**2+n-2*ss)/(n**2-n))


def max_ss(n, s_min, k):
    """
        The maximal sum of squares of a clustering consisting of k clusters, each of size at least smin
    """
    if k * s_min > n:
        return -float('inf')
    return (k-1)*(s_min**2)+(n-(k-1)*s_min)**2


def lat2ss(lat, n):
    """
        The sum of squares corresponding to a latitude for n vertices.
    """
    return n+n*(n-1)*np.sin(lat/2)**2


# The minimal cluster size for which the squared sum of sizes of a clustering of k clusters can exceed ss
def ss2s_min(ss, n, k):
    if ss <= n**2/k:
        return int(n/k)
    s = (n/k)*(1-((k*(ss/n**2)-1)/(k-1))**0.5)
    if s > 1:
        return int(s)
    return 1


def ss_k2sizes(ss, k, n):
    """
        Returns k cluster sizes summing to n, such that the sum of squares is approximately equal to ss.
    """
    if k == 1:
        return [n]
    s_min = ss2s_min(ss, n, k)
    return ss_k2sizes(ss-s_min**2, k-1, n-s_min)+[s_min]


def lat2sizes(lat,n):
    """
        Returns k cluster sizes summing to n, such that the latitude is approximately equal to lat.
    """
    ss = lat2ss(lat, n)
    k = int(np.ceil(k_eff(lat, n)))
    return ss_k2sizes(ss, k, n)
