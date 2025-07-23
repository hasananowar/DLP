## Dual Link Prediction

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

#### Generate edge from 3D coordinates of atoms

```shell
python gen_data.ipynb
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



### Datasets

Presently, we only provide a small sample of the dataset used. Since the data was obtained from our collaborators working on Molecular Dynamics Simulation from other Universities - please send us a formal request via email and we will promptly respond and provide it.

## Contact

If you have any questions, feel free to contact us.
Emails: `mhanowar@iastate.edu` or `gocet25@iastate.edu`



