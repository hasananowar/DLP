import torch
import numpy as np
import argparse
# from utils import set_seed, load_feat, load_graph
from data_process_utils import check_data_leakage

import os
import pandas as pd
import random

####################################################################
####################################################################
####################################################################


def print_model_info(model):
    # print(model)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters])
    print('Trainable Parameters: %d' % parameters)

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='FILT_HB', help="Base directory for the data files")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--max_edges', type=int, default=50)
    parser.add_argument('--num_edgeType', type=int, default=0, help='num of edgeType')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--predict_class', action='store_true')
    
    # model
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--model', type=str, default='DLP') 
    parser.add_argument('--neg_samples', type=int, default=1)
    parser.add_argument('--extra_neg_samples', type=int, default=1)
    parser.add_argument('--num_neighbors', type=int, default=50)
    parser.add_argument('--channel_expansion_factor', type=int, default=1)
    parser.add_argument('--sampled_num_hops', type=int, default=1)
    parser.add_argument('--pair_dims', type=int, default=100)
    parser.add_argument('--time_dims', type=int, default=100)
    parser.add_argument('--hidden_dims', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--check_data_leakage', action='store_true')
    
    parser.add_argument('--ignore_node_feats', action='store_true')
    parser.add_argument('--node_feats_as_edge_feats', action='store_true')
    parser.add_argument('--ignore_edge_feats', action='store_true')
    parser.add_argument('--use_onehot_node_feats', action='store_true')
    parser.add_argument('--use_type_feats', action='store_true')
    parser.add_argument('--use_pair_index', action='store_true')

    parser.add_argument('--use_graph_structure', action='store_true')
    parser.add_argument('--structure_time_gap', type=int, default=2000)
    parser.add_argument('--structure_hops', type=int, default=1) 

    parser.add_argument('--use_node_cls', action='store_true')
    parser.add_argument('--use_cached_subgraph', action='store_true')
    parser.add_argument('--early_stop_patience', type=int, default=30)
    return parser.parse_args()

# utility function
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_feat(d):
    node_feats = None
    if os.path.exists('DATA/{}/node_features.pt'.format(d)):
        node_feats = torch.load('DATA/{}/node_features.pt'.format(d))
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.float32)


    edge_feats1 = None
    edge_feats2 = None
    if os.path.exists('DATA/{}/edge_features.pt'.format(d)):
        edge_feats = torch.load('DATA/{}/edge_features.pt'.format(d))
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)
    return node_feats, edge_feats1, edge_feats2   

def load_graph(d):
    """
    Load graph and edge list data from the specified directory.

    Parameters:
    - d (str): Base directory name containing graph and edge files.

    Returns:
    - g1, df1: First graph and its edge list data.
    - g2, df2: Second graph and its edge list data.
    """
    df1 = pd.read_csv(f'DATA/{d}/edges1.csv')
    g1 = np.load(f'DATA/{d}/edges1.npz')
    df2 = pd.read_csv(f'DATA/{d}/edges2.csv')
    g2 = np.load(f'DATA/{d}/edges2.npz')
    return g1, g2, df1, df2

def load_all_data(args):

    # load graph
    g1, g2, df1, df2 = load_graph(args.data)
    print(f"Loaded graphs and data from DATA/{args.data}")

    # Determine split indices based on one of the datasets
    args.train_edge_end = df1[df1['ext_roll'].gt(0)].index[0]
    args.val_edge_end   = df1[df1['ext_roll'].gt(1)].index[0]

    args.num_nodes1 = max(int(df1['src'].max()), int(df1['dst'].max())) + 1
    args.num_nodes2 = max(int(df2['src'].max()), int(df2['dst'].max())) + 1
    args.num_edges = len(df1) # the number of edges are same for both datasets

    print('Train %d, Valid %d, Test %d'%(args.train_edge_end, 
                                         args.val_edge_end-args.train_edge_end,
                                         len(df1)-args.val_edge_end))
    print('Num nodes for data1 %d, Num nodes for data2 %d, num edges %d' % (args.num_nodes1, args.num_nodes2, args.num_edges))
    
    # Load features 
    node_feats, edge_feats1, edge_feats2  = load_feat(args.data)  
    
    # Feature pre-processing
    node_feat_dims = 0 if node_feats is None else node_feats.shape[1]
    edge_feat1_dims = 0 if edge_feats1 is None else edge_feats1.shape[1]
    edge_feat2_dims = 0 if edge_feats2 is None else edge_feats2.shape[1]
    
    if args.use_onehot_node_feats:
        print('>>> Use one-hot node features')

        num_classes = int(node_feats.max().item())+1  
        node_feats = torch.nn.functional.one_hot(node_feats.to(torch.int64).squeeze(), num_classes=num_classes)
        node_feats = node_feats.to(torch.float32)
        node_feat_dims = node_feats.size(1)

        print('One-hot node feature dim =', (node_feats.shape))
    
    if args.ignore_node_feats:
        print('>>> Ignore node features')
        node_feats = None
        node_feat_dims = 0


    if args.use_pair_index:
        num_pair_index = max(df1['dst_idx'].max(), df2['dst_idx'].max()) + 1 

        pair_index1 = torch.tensor(df1.dst_idx.values, dtype=torch.long)
        pair_index2 = torch.tensor(df2.dst_idx.values, dtype=torch.long)

        # One-hot encode the indices 
        pair_feats1 = torch.nn.functional.one_hot(pair_index1, num_classes=num_pair_index).float()
        pair_feats2 = torch.nn.functional.one_hot(pair_index2, num_classes=num_pair_index).float()

        #################### 
        pair_diff1 = torch.abs(pair_feats1 - pair_feats2)   
        pair_mul1 = pair_feats1 * pair_feats2              

        pair_diff2 = torch.abs(pair_feats2 - pair_feats1)
        pair_mul2 = pair_feats2 * pair_feats1

        pair_feats_combined1 = torch.cat([pair_diff1, pair_mul1], dim=1)
        pair_feats_combined2 = torch.cat([pair_diff2, pair_mul2], dim=1)

        ####################

        print('Pair embedding feature dim 1: %d, feature dim 2: %d' % (pair_feats_combined1.size(1), pair_feats_combined2.size(1)))

    if args.use_type_feats:
        edge_type1 = df1.label.values
        args.num_edgeType1 = len(set(edge_type1.tolist()))
        type_feats1 = torch.nn.functional.one_hot(torch.as_tensor(edge_type1, dtype=torch.long),num_classes=args.num_edgeType1)
        type_feats1_dims = type_feats1.size(1)

        edge_type2 = df2.label.values
        args.num_edgeType2 = len(set(edge_type2.tolist()))
        type_feats2 = torch.nn.functional.one_hot(torch.as_tensor(edge_type2, dtype=torch.long),num_classes=args.num_edgeType2)
        type_feats2_dims = type_feats2.size(1)

        print('Type feature dim 1: %d, type feature dim 2: %d' % (type_feats1_dims, type_feats2_dims))

    
    if args.use_type_feats and args.use_pair_index:
        # Concatenate type features and pair features along dimension 1.
        edge_feats1 = torch.cat([type_feats1, pair_feats1], dim=1)
        edge_feats2 = torch.cat([type_feats2, pair_feats2], dim=1)

        edge_feat1_dims = edge_feats1.size(1)
        edge_feat2_dims = edge_feats2.size(1)
        
        print('Final Edge 1 feat dim: %d, Final Edge 2 feat dim: %d' % (edge_feat1_dims, edge_feat2_dims))
        
    elif args.use_type_feats and not args.use_pair_index:
        # Use only type features.
        edge_feats1 = type_feats1
        edge_feats2 = type_feats2

        edge_feat1_dims = edge_feats1.size(1)
        edge_feat2_dims = edge_feats2.size(1)
        
        print('Final Edge 1 feat dim: %d, Final Edge 2 feat dim: %d' % (edge_feat1_dims, edge_feat2_dims))
        
    elif args.use_pair_index and not args.use_type_feats:
        # Use only pair index features.
        edge_feats1 = pair_feats_combined1
        edge_feats2 = pair_feats_combined2

        edge_feat1_dims = edge_feats1.size(1)
        edge_feat2_dims = edge_feats2.size(1)
        
        print('Final Edge 1 feat dim: %d, Final Edge 2 feat dim: %d' % (edge_feat1_dims, edge_feat2_dims))

    # Data leakage check 
    if args.check_data_leakage:
        check_data_leakage(args, g1, df1)
        check_data_leakage(args, g2, df2)
    
    args.node_feat_dims = node_feat_dims
    args.edge_feat_dims = max(edge_feat1_dims, edge_feat2_dims)
    
    # Move features to device
    if node_feats is not None:
        node_feats = node_feats.to(args.device)
    if edge_feats1 is not None:
        edge_feats1 = edge_feats1.to(args.device)
    if edge_feats2 is not None:
        edge_feats2 = edge_feats2.to(args.device)
    
    return node_feats, edge_feats1, edge_feats2, g1, g2, df1, df2, args


def load_model_dual(args):
    # Define edge predictor configurations
    edge_predictor_configs = {
        'dim_in_time': args.time_dims,
        'dim_in_node': args.node_feat_dims,
        'predict_class': 1 if not args.predict_class else args.num_edgeType + 1,
    }

    if args.model == 'DLP':

        from model import Dual_Interface as DLP_Interface
        from dual_link_pred_train_utils import link_pred_train_dual

        mixer_configs = {
            'per_graph_size'  : args.max_edges,
            'time_channels'   : args.time_dims, 
            'input_channels'  : args.edge_feat_dims, 
            'hidden_channels' : args.hidden_dims, 
            'out_channels'    : args.hidden_dims,
            'num_layers'      : args.num_layers,
            'dropout'         : args.dropout,
            'channel_expansion_factor': args.channel_expansion_factor,
            'window_size'     : args.window_size,
            'use_single_layer' : False
        }  
    
    else:
        raise NotImplementedError("Model {} is not implemented.".format(args.model))
    
    model = DLP_Interface(mixer_configs, edge_predictor_configs)
    for k, v in model.named_parameters():
        print(k, v.requires_grad)

    # print_model_info(model)

    return model, args, link_pred_train_dual
        
####################################################################
####################################################################

if __name__ == "__main__":
    args = get_args()

    # Set specific arguments related to graph structure and feature usage
    args.use_graph_structure = True
    args.ignore_node_feats = True  # We only use graph structure
    # args.use_onehot_node_feats = True # Use node features
    # args.use_type_feats = True     # Type encoding
    # args.use_pair_index = True     # Pair encoding
    args.use_cached_subgraph = True

    print(args)

    # Determine device
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.device = torch.device(args.device)
    
    # Set random seed for reproducibility
    set_seed(0)

    ###################################################
    # Load features and graphs
    node_feats, edge_feats1, edge_feats2, g1, g2, df1, df2, args = load_all_data(args)
        
    ###################################################
    # Load model
    model, args, link_pred_train_dual = load_model_dual(args)

    ###################################################
    # Link prediction training
    print('Train dual link prediction task from scratch ...')
    model = link_pred_train_dual(model.to(args.device), args, g1, g2, df1, df2, node_feats, edge_feats1, edge_feats2)