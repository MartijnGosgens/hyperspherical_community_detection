

def generate_figures_hyperspherical_geometry():
    """
        Running this function generates all figures of the paper
        "The Hyperspherical Geometry of Community Detection: Modularity as a Distance".
        Each of the imported modules contains a function generate_figures() that generates the corresponding figures.
    """
    from .experiments import heatmaps as hm
    from .experiments import ppm_experiment as ppme
    from .experiments import realworldexperiments as rwe
    from .experiments import wedgescomparison as wc

    for f in [ppme, hm, wc, rwe]:
        f.generate_figures()


def generate_figures_granularity_bias():
    """
        Running this function generates all figures of the paper
        "Correcting for Granularity Bias in Modularity-Based Community Detection Methods".
        Each of the imported modules contains a function generate_figures() that generates the corresponding figures.
    """
    from .experiments import heuristic_comparison as hc
    from .experiments import MLE_vs_heuristic_experiment as mvh
    from .experiments import observation1_validation as o1v
    from .experiments import observation1_demonstration as o1d
    from .experiments import observation2_demonstration as o2d

    for f in [hc, mvh, o1v, o1d, o2d]:
        f.generate_figures()


def generate_all_figures():
    """
        Running this function generates all figures of both papers.
    """
    generate_figures_hyperspherical_geometry()
    generate_figures_granularity_bias()
