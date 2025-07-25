import argparse
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm

import os

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='FILT_HB/edges1.csv', type=str, help='dataset name')
parser.add_argument('--add_reverse', default=True, action='store_true')

args = parser.parse_args()
print(args)


# df = pd.read_csv('DATA/{}/edges.csv'.format(args.data))

# Extract the base name of the input CSV file (e.g., edges1 from HB/edges1.csv)
npz_filename = os.path.splitext(os.path.basename(args.data))[0] + '.npz'
npz_output_path = os.path.join('DATA', os.path.dirname(args.data), npz_filename)

# Load CSV Data
csv_path = os.path.join('DATA', args.data)
df = pd.read_csv(csv_path)

# num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1 

# src_num_nodes = int(df['src'].max()) - int(df['src'].min()) + 1
# dst_num_nodes = int(df['dst'].max()) - int(df['dst'].min()) + 1
# num_nodes = src_num_nodes + dst_num_nodes
num_nodes = int(max(df['src'].max(), df['dst'].max())) + 1
print('num_nodes: ', num_nodes)

ext_full_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
ext_full_indices = [[] for _ in range(num_nodes)]
ext_full_ts = [[] for _ in range(num_nodes)]
ext_full_eid = [[] for _ in range(num_nodes)]

for idx, row in tqdm(df.iterrows(), total=len(df)):
    src = int(row['src'])
    dst = int(row['dst'])
    
    ext_full_indices[src].append(dst)
    ext_full_ts[src].append(row['time'])
    ext_full_eid[src].append(idx)
    
    if args.add_reverse:
        ext_full_indices[dst].append(src)
        ext_full_ts[dst].append(row['time'])
        ext_full_eid[dst].append(idx)

for i in tqdm(range(num_nodes)):
    ext_full_indptr[i + 1] = ext_full_indptr[i] + len(ext_full_indices[i])

ext_full_indices = np.array(list(itertools.chain(*ext_full_indices)))
ext_full_ts = np.array(list(itertools.chain(*ext_full_ts)))
ext_full_eid = np.array(list(itertools.chain(*ext_full_eid)))

print('Sorting...')
# sort each neighbor list by timestamp
def tsort(i, indptr, indices, t, eid):
    beg = indptr[i]
    end = indptr[i + 1]
    sidx = np.argsort(t[beg:end])
    indices[beg:end] = indices[beg:end][sidx]
    t[beg:end] = t[beg:end][sidx]
    eid[beg:end] = eid[beg:end][sidx] 

for i in tqdm(range(ext_full_indptr.shape[0] - 1)):
    tsort(i, ext_full_indptr, ext_full_indices, ext_full_ts, ext_full_eid)

print('saving...')

# np.savez('DATA/{}/ext_full.npz'.format(args.data), indptr=ext_full_indptr,
#         indices=ext_full_indices, ts=ext_full_ts, eid=ext_full_eid)

np.savez(npz_output_path, indptr=ext_full_indptr,
         indices=ext_full_indices, ts=ext_full_ts, eid=ext_full_eid)


