# README
This repository contains the code to run the experiments of the papers "*The Hyperspherical Geometry of Community Detection: Modularity as a Distance*" and "*Correcting for Granularity Bias in Modularity-Based Community Detection Methods*".

To run these experiments, you will need to following python packages:
* `numpy`
* `matplotlib`
* `networkx`
* `json`
* `pandas`

For the ABCD benchmark graphs, we have saved the graphs that were used in the "*The Projection Method: a Unified Formalism for Community Detection*" in `random_graphs/ABCD_n1000`. Which are automatically loaded when using `random_graphs.generators.ABCD_benchmark.generate` with parameter `load=True`.
To generate new graphs from the ABCD graph generator, one needs to clone [this repository](https://github.com/bkamins/ABCDGraphGenerator.jl) and change the `ABCD_PATH` variable in `random_graphs.generators` to the directory where you cloned this repository.

## Reproducing the experiments
The folder `experiments` contains the implementations of the experiments. To reproduce all the figures, simply run the following code from a directory outside this module (but from where this module can be found).
```python
from hyperspherical_community_detection import generate_all_figures
generate_all_figures()
```
Performing all the experiments and generating the figures may take a few hours. The implementations of these experiments and figures are found in the folder `experiments`. To only generate the figures of the paper "*The Hyperspherical Geometry of Community Detection: Modularity as a Distance*", run

```python
from hyperspherical_community_detection import generate_figures_hyperspherical_geometry
generate_figures_hyperspherical_geometry()
```

Similarly, to generate all figures and the table of "*Correcting for Granularity Bias in Modularity-Based Community Detection Methods*", run

```python
from hyperspherical_community_detection import generate_figures_granularity_bias
generate_figures_granularity_bias()
```
And for the figures of "*The Projection Method: a Unified Formalism for Community Detection*", run
```python
from hyperspherical_community_detection import generate_figures_projection_method
generate_figures_projection_method()
```

## Implementation of PairVectors and Louvain projection
The implementation of our modification of Louvain can be found in the folder `algorithms`. The hyperspherical computations are also implemented there. The most important implementations are found in `algorithms.pair_vector`, where we implement clustering vectors and query mappings. We implement these in a way that avoids explicitly storing each entry of the pair-vector. For example, for a clustering vector, we store the partition and implement all relevant operations in ways that avoid iterating over all vertex-pairs.

### Demonstration
To demonstrate the methods implemented in this package, we compute the wedge vector for the Karate network, project it to the equator, compute the corresponding Louvain projection, and compare the obtained candidate clustering to the ground truth using the Correlation Distance.

```python
from hyperspherical_community_detection.experiments.benchmarknetworks import load_dataset
from hyperspherical_community_detection.algorithms import pair_vector as pv 
from numpy import pi, cos

# Obtain karate network and ground truth clustering
G,T = load_dataset('karate')
# Compute wedge vector
w = pv.wedges(G)
# Project to equator
q = w.latitude_on_meridian(pi/2)
# Apply the Louvain projection to obtain a candidate clustering
C = pv.louvain_projection(q)
# Convert clusterings to vectors
bC = pv.clustering_binary(C)
bT = pv.clustering_binary(T)
# print the meridian angle (equal to the correlation distance)
cd = bC.meridian_angle(bT)
CC = cos(cd)
print("The correlation distance equals {:.2f}, which corresponds to a Pearson correlation of {:.2f}".format(cd,CC))
```
