python setup.py build_ext --inplace
python gen_graph.py --data flavanone/edges1.csv
python gen_graph.py --data flavanone/edges2.csv
python train.py --data flavanone --epochs 300