python setup.py build_ext --inplace
python gen_graph.py --data FILT_HB/edges1.csv
python gen_graph.py --data FILT_HB/edges2.csv
python train.py --data FILT_HB --epochs 1