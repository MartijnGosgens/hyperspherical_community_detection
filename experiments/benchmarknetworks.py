import networkx as nx
import os
import urllib.request as req
from ..Clustering import Clustering

'''
    This file retrieves the datasets from https://github.com/altsoph/community_loglike 
    To retrieve the tuple (graph,groundtruth) of a dataset, simply use load_dataset(name).
    The graph is returned as a networkx graph, while the ground truth is returned as a Clustering.
    To retrieve a dictionary indexed by the network names, simply use load_datasets().
'''

network_names = [
    'karate',
    'dolphins',
    'polbooks',
    'football',
    'eu-core',
    'polblogs',
    'cora',
    'as'
]

def load_network(url):
    file = req.urlopen(url)
    return  nx.read_edgelist(file)


def load_clustering(url):
    file = req.urlopen(url)
    return Clustering(dict([
        l.strip().decode("utf-8").split('\t')
        for l in file.readlines()
    ]))


def load_dataset(name):
    url_start = "https://raw.githubusercontent.com/altsoph/community_loglike/master/datasets/{}/{}".format(
        name if name!="cora" else "cora_full", name
    )
    return load_network(url_start + ".edges"), load_clustering(url_start + ".clusters")


def load_datasets():
    networks = {
        name: load_dataset(name)
        for name in network_names
    }
    # Sort ascendingly in network size
    networks_sorted = sorted(network_names, key=lambda name: len(networks[name][0]))
    return {
        name: networks[name]
        for name in networks_sorted
    }