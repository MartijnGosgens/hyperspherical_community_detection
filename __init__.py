"""
    Running the function below generates all figures. Each of the imported modules contains a function generate_all()
    that generates the corresponding figures.
"""


def generate_all_figures():
    from .experiments import heatmaps as hm
    from .experiments import ppm_experiment as ppme
    from .experiments import realworldexperiments as rwe
    from .experiments import wedgescomparison as wc

    for f in [ppme, hm, wc, rwe]:
        f.generate_figures()
