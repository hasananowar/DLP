## DLP
This repository provides the source code for the implementation of the methods described in: **Hydrogen Bond Prediction With Spatio-Temporal Evolution Awareness** (NOTE: Submitted as an application paper at ACM SIGSPATIAL 2026).

In this work, we propose a novel link prediction framework (DLP) that predicts the coexistence of *dual links* from dynamic molecular graphs.
![DLP Framework](./figs/DLP_model_2.png)


### Dependencies

The following libraries and frameworks are required to run the code. Make sure to install these dependencies using `pip` or `conda`.

- **torch**: PyTorch, a deep learning framework.
- **torch-geometric**: A library for graph neural networks.
- **torch_scatter**: Operations for sparse data.
- **torch_sparse**: Sparse matrix operations.
- **torch_cluster**: Clustering algorithms for graphs.
- **torch_spline_conv**: Spline-based convolution for graphs.
- **pybind11**: A library for creating Python bindings for C++ code.
- **torchmetrics**: Metrics for evaluating PyTorch models.
- **pandas**: Data manipulation and analysis library.
- **numpy**: Library for numerical computations.
- **scipy**: Library for scientific computing.


### How to use

#### Install libraries

```shell
pip install -r requirements.txt
```

#### Generate edge from time-stamped 3D coordinates of atoms

```shell
python gen_data.py --raw_data HB1000frames.csv
```

#### Compile C++ sampler

```shell
python setup.py build_ext --inplace
```
#### Graph generation

```shell
python gen_graph.py --data FILT_HB/edges1.csv
python gen_graph.py --data FILT_HB/edges2.csv
```
#### Dual Link prediction

```shell
python train.py --data FILT_HB
```

#### Baselines

To run the adapted baselines, go to the corresponding folder and run the following:

```shell
bash run.sh
```


### Datasets

Presently, we only provide a small sample of the dataset used: [Download Sample Dataset](https://drive.google.com/file/d/1cYsPavp3G7H16gU5ysX_ldmjD8uA7by-/view?usp=sharing). Since the data was obtained from our collaborators working on Molecular Dynamics Simulation at other institutions, please send us a formal request via email and we will promptly respond and provide it.



#### Data Preprocessing
This section describes the preprocessing pipeline that converts raw molecular dynamics trajectories into temporal graphs. The procedure computes geometric descriptors (distances and angles), derives time-stamped molecular interaction edges, and produces node and edge attributes for downstream temporal modeling.

We begin with molecular dynamics simulation (MDS) trajectories that record the 3D positions and metadata of atoms at each timestamp. Each record provides the spatial coordinates $(x, y, z)$ at time $t$, together with *Atom Name*, *Atom Type*, *Molecule Name*, and *Molecule ID*. All datasets contain two types of molecules:

1. a polymer substrate, Amylose Tris (3,5-dimethylphenyl carbamate), commonly referred to as ADMPC, and
2. a drug molecule — Flavanone for the *Flavanone255k* and *Flavanone80k* datasets, and Benzoin for the *Benzoin* dataset.

<!-- <p align="center">
  <img src="./figs/ADMPC.png" width="30%" alt="ADMPC polymer repeating unit"/>
</p>
<p align="center"><em>A repeating unit of the polymer (ADMPC).</em></p> -->

For HB analysis, we focus exclusively on donors, covalently bonded H atoms, and acceptors. Timestamps are normalized by shifting all frames to a common temporal origin. For each time $t$, we enumerate all donor–hydrogen–acceptor triplets and compute the geometric criteria associated with HB formation.

1. **Distance edges.** For an acceptor $u$ and a donor $v$ at time $t$, we compute the Euclidean distance $d_{u,v}(t)$ and create a distance-type edge $e^{\text{dist}}_{u,v}(t)$ if $d_{u,v}(t) \leq 3.5\ \text{Å}$.
2. **Angle edges.** Given an HB-capable triplet $(u, v, w)$ at time $t$, we compute the angle $\theta_{u,w,v}(t)$ and create an angle-type edge $e^{\text{angle}}_{u,w}(t)$ if $135^\circ \leq \theta_{u,w,v}(t) < 180^\circ$.

This step yields two temporal edge streams (distance-type and angle-type), each with attributes: *(source, destination, timestamp)*. We merge the distance-type and angle-type edges on common identifiers. We also assign all node identifiers to a *global* node index, ensuring that distance and angle edges share a common node space. Since a dual link corresponds to the simultaneous satisfaction of both geometric criteria at a given timestamp, we label entries appearing in both streams as positive dual interactions ($y=1$). Entries present in only one stream are labeled as negatives ($y=0$). Finally, we export the processed data as two CSV files, `edges1.csv` (distance-type) and `edges2.csv` (angle-type), along with a `node_features.pt` file that encodes molecule-level node attributes.

### Contact

If you have any questions, feel free to contact us.
Emails: `mhanowar@iastate.edu` or `gocet25@iastate.edu`

### License

This project is released under the MIT License. See `LICENSE` and `THIRD_PARTY_LICENSES.md` for details.
