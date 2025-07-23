python setup.py build_ext --inplace
python gen_graph.py --data benzoin/edges1.csv
python gen_graph.py --data benzoin/edges2.csv
python train.py --data benzoin --epochs 300
python train.py --data benzoin --epochs 300 --use_pair_index
