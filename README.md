## Dual Link Prediction


### How to run the code

#### Generate edge from 3D coordinates of atoms

```shell
python gen_edge.py
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
#### 3. Dual Link prediction

```shell
python train.py --data FILT_HB --epochs 50
```



### Dependencies

torch
torch-geometric 
torch_scatter 
torch_sparse 
torch_cluster 
torch_spline_conv
pybind11 torchmetrics
pandas
numpy
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



