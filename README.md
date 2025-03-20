# Dual Link Prediction



### Data Preparation

Prepare data for DLP by formating the data.

## Generate graph structure from 3D coordinates



### How to run the code

## 1. Compile C++ sampler 

```shell
python setup.py build_ext --inplace
```
## 2. Data pre-process

```shell
python gen_graph.py --data FILT_HB/edges1.csv
python gen_graph.py --data FILT_HB/edges2.csv
```
### 3. Dual Link prediction

```shell
python train.py --data FILT_HB --epochs 10
```