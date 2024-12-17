import os
import pickle
from tqdm import tqdm
import numpy as np
from construct_subgraph import get_parallel_sampler, get_mini_batch

######################################################################################################
######################################################################################################
######################################################################################################
# From Graphmixer, for large dataset, we need to sample on the fly

class SubgraphSampler:
    def __init__(self, all_root_nodes, all_ts, sampler, args):
        self.all_root_nodes = all_root_nodes
        self.all_ts = all_ts
        self.sampler = sampler
        self.sampled_num_hops = args.sampled_num_hops

    def mini_batch(self, ind, mini_batch_inds):
        root_nodes = self.all_root_nodes[ind][mini_batch_inds]
        ts = self.all_ts[ind][mini_batch_inds]
        return get_mini_batch(self.sampler, root_nodes, ts, self.sampled_num_hops)

def get_subgraph_sampler(args, g, df, mode):
    ###################################################
    # get cached file_name
    if mode == 'train':
        extra_neg_samples = args.extra_neg_samples
    else:
        extra_neg_samples = 1

    ###################################################
    # for each node, sample its neighbors with the most recent neighbors (sorted) 
    print('Sample subgraphs ... for %s mode'%mode)
    sampler, neg_link_sampler = get_parallel_sampler(g, args.num_neighbors)

    ###################################################
    # setup modes
    if mode == 'train':
        cur_df = df[:args.train_edge_end]

    elif mode == 'valid':
        cur_df = df[args.train_edge_end:args.val_edge_end]

    elif mode == 'test':
        cur_df = df[args.val_edge_end:]

    loader = cur_df.groupby(cur_df.index // args.batch_size)
    pbar = tqdm(total=len(loader))
    pbar.set_description('Pre-sampling: %s mode with negative sampleds %s ...'%(mode, extra_neg_samples))

    all_root_nodes = []
    all_ts = []
    for _, rows in loader:

        root_nodes = np.concatenate(
            [rows.src.values, 
            rows.dst.values, 
            neg_link_sampler.sample(len(rows) * extra_neg_samples)]
        ).astype(np.int32)
        all_root_nodes.append(root_nodes)

        # time-stamp for node = edge time-stamp
        ts = np.tile(rows.time.values, extra_neg_samples + 2).astype(np.float32)
        all_ts.append(ts)

        pbar.update(1)
    pbar.close()
    return SubgraphSampler(all_root_nodes, all_ts, sampler, args)

######################################################################################################
######################################################################################################
######################################################################################################
# for small dataset, we can cache each graph
def pre_compute_subgraphs(args, g, df, mode):
    ###################################################
    # get cached file_name
    if mode == 'train':
        extra_neg_samples = args.extra_neg_samples
    else:
        extra_neg_samples = 1
        
    fn = os.path.join(os.getcwd(), 'DATA', args.data, 
                        '%s_neg_sample_neg%d_bs%d_hops%d_neighbors%d.pickle'%(mode, 
                                                                            extra_neg_samples, 
                                                                            args.batch_size, 
                                                                            args.sampled_num_hops, 
                                                                          args.num_neighbors))
    ###################################################

    # try:
    if os.path.exists(fn):
        subgraph_elabel = pickle.load(open(fn, 'rb'))
        # print('load ', fn)

    else:
        ###################################################
        # for each node, sample its neighbors with the most recent neighbors (sorted) 
        print('Sample subgraphs ... for %s mode'%mode)
        sampler, neg_link_sampler = get_parallel_sampler(g, args.num_neighbors)
    
        ###################################################
        # setup modes
        if mode == 'train':
            cur_df = df[:args.train_edge_end]

        elif mode == 'valid':
            cur_df = df[args.train_edge_end:args.val_edge_end]

        elif mode == 'test':
            cur_df = df[args.val_edge_end:]

        loader = cur_df.groupby(cur_df.index // args.batch_size)
        pbar = tqdm(total=len(loader))
        pbar.set_description('Pre-sampling: %s mode with negative sampleds %s ...'%(mode, extra_neg_samples))

        ###################################################
        all_subgraphs = []
        all_elabel = []
        sampler.reset()
        for _, rows in loader:

            root_nodes = np.concatenate(
                [rows.src.values, 
                 rows.dst.values, 
                 neg_link_sampler.sample(len(rows) * extra_neg_samples)]
            ).astype(np.int32)

            # time-stamp for node = edge time-stamp
            ts = np.tile(rows.time.values, extra_neg_samples + 2).astype(np.float32)
            all_elabel.append(rows.label.values)
            all_subgraphs.append(get_mini_batch(sampler, root_nodes, ts, args.sampled_num_hops))
            
            pbar.update(1)
        pbar.close()
        subgraph_elabel = (all_subgraphs, all_elabel)
        try:
            pickle.dump(subgraph_elabel, open(fn, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        except:
            print('For some shit reason pickle cannot save ... but anyway ...')
        
        ###################################################
        
    return subgraph_elabel


def get_random_inds(num_subgraph, cached_neg_samples, neg_samples):
    ###################################################
    batch_size = num_subgraph // (2+cached_neg_samples)
    pos_src_inds = np.arange(batch_size)
    pos_dst_inds = np.arange(batch_size) + batch_size
    neg_dst_inds = np.random.randint(low=2, high=2+cached_neg_samples, size=batch_size*neg_samples)
    neg_dst_inds = batch_size * neg_dst_inds + np.arange(batch_size)
    mini_batch_inds = np.concatenate([pos_src_inds, pos_dst_inds, neg_dst_inds]).astype(np.int32)
    ###################################################

    return mini_batch_inds



def check_data_leakage(args, g1, df1, g2, df2):
    """
    Double-check to ensure that sampled subgraphs do not leak information from future edges.
    
    Args:
        args: Configuration arguments.
        g1: Graph for edges1.csv.
        df1: DataFrame for edges1.csv.
        g2: Graph for edges2.csv.
        df2: DataFrame for edges2.csv.
    """
    for mode in ['train', 'valid', 'test']:
        if mode == 'train':
            cur_df1 = df1[:args.train_edge_end]
            cur_df2 = df2[:args.train_edge_end]
        elif mode == 'valid':
            cur_df1 = df1[args.train_edge_end:args.val_edge_end]
            cur_df2 = df2[args.train_edge_end:args.val_edge_end]
        elif mode == 'test':
            cur_df1 = df1[args.val_edge_end:]
            cur_df2 = df2[args.val_edge_end:]
    
        loader1 = cur_df1.groupby(cur_df1.index // args.batch_size)
        subgraphs1 = pre_compute_subgraphs(args, g1, df1, mode)
        
        loader2 = cur_df2.groupby(cur_df2.index // args.batch_size)
        subgraphs2 = pre_compute_subgraphs(args, g2, df2, mode)
    
        for i, (_, rows) in enumerate(loader1):
            root_nodes = np.concatenate([rows.src.values, rows.dst1.values]).astype(np.int32)
            eids = np.tile(rows.index.values, 2)
            cur_subgraphs = subgraphs1[i][:args.batch_size*2]
    
            for eid, cur_subgraph in zip(eids, cur_subgraphs):
                all_eids_in_subgraph = cur_subgraph['eid']
                if len(all_eids_in_subgraph) == 0:
                    continue
                # Ensure sampled edges have eid < current eid to prevent leakage
                assert all(all_eids_in_subgraph < eid), "Data leakage detected in edges1."
    
        for i, (_, rows) in enumerate(loader2):
            root_nodes = np.concatenate([rows.src.values, rows.dst2.values]).astype(np.int32)
            eids = np.tile(rows.index.values, 2)
            cur_subgraphs = subgraphs2[i][:args.batch_size*2]
    
            for eid, cur_subgraph in zip(eids, cur_subgraphs):
                all_eids_in_subgraph = cur_subgraph['eid']
                if len(all_eids_in_subgraph) == 0:
                    continue
                # Ensure sampled edges have eid < current eid to prevent leakage
                assert all(all_eids_in_subgraph < eid), "Data leakage detected in edges2."
    
    print('Does not detect information leakage ...')
