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

### Adapted Baselines

```shell
python extra_preprocess.py DG_data/FILT_HB_DIST/edges1.csv DG_data/FILT_HB_DIST/FILT_HB_DIST.csv
python extra_preprocess.py DG_data/FILT_HB_ANGLE/edges2.csv DG_data/FILT_HB_ANGLE/FILT_HB_ANGLE.csv
cd preprocess_data/
python preprocess_data.py --dataset_name FILT_HB_DIST
python preprocess_data.py --dataset_name FILT_HB_ANGLE
cd ../
python train_link_prediction.py --dataset_name FILT_HB_DIST --model_name TGAT --optimizer RMSprop --num_layers 1 --batch_size 100 --num_neighbors 50 --num_runs 1 --dropout 0.1 --weight_decay 0.0001 --test_interval_epochs 1 --num_epochs 300
python train_link_prediction.py --dataset_name FILT_HB_ANGLE --model_name TGAT --optimizer RMSprop --num_layers 1 --batch_size 100 --num_neighbors 50 --num_runs 1 --dropout 0.1 --weight_decay 0.0001 --test_interval_epochs 1 --num_epochs 300
python postprocess.py --file1_pattern temp_result/outputs_FILT_HB_DIST_TGAT_seed{}.pt --file2_pattern temp_result/outputs_FILT_HB_ANGLE_TGAT_seed{}.pt --summary1_pattern temp_result/summary_FILT_HB_DIST_TGAT_seed{}.json --summary2_pattern temp_result/summary_FILT_HB_ANGLE_TGAT_seed{}.json
```


### Datasets

Presently, we only provide a small sample of the dataset used [Download Sample Dataset](https://drive.google.com/file/d/1cYsPavp3G7H16gU5ysX_ldmjD8uA7by-/view?usp=sharing). Since the data was obtained from our collaborators working on Molecular Dynamics Simulation at other institutions - please send us a formal request via email and we will promptly respond and provide it.

## Contact

If you have any questions, feel free to contact us.
Emails: `mhanowar@iastate.edu` or `gocet25@iastate.edu`



