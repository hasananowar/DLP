import torch
import numpy as np
import argparse
# from utils import set_seed, load_feat, load_graph
from data_process_utils import check_data_leakage

import os
import pandas as pd
import random

import json
import optuna
import gc
torch.cuda.empty_cache()
gc.collect()
# Add PYTORCH_CUDA_ALLOC_CONF to environment
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


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
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--predict_class', action='store_true')
    
    # model
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--model', type=str, default='sthn') 
    parser.add_argument('--neg_samples', type=int, default=1)
    parser.add_argument('--extra_neg_samples', type=int, default=5)
    parser.add_argument('--num_neighbors', type=int, default=50)
    parser.add_argument('--channel_expansion_factor', type=int, default=2)
    parser.add_argument('--sampled_num_hops', type=int, default=1)
    parser.add_argument('--time_dims', type=int, default=100)
    parser.add_argument('--hidden_dims', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--check_data_leakage', action='store_true')
    
    parser.add_argument('--ignore_node_feats', action='store_true')
    parser.add_argument('--node_feats_as_edge_feats', action='store_true')
    parser.add_argument('--ignore_edge_feats', action='store_true')
    parser.add_argument('--use_onehot_node_feats', action='store_true')
    parser.add_argument('--use_type_feats', action='store_true')

    parser.add_argument('--use_graph_structure', action='store_true')
    parser.add_argument('--structure_time_gap', type=int, default=2000)
    parser.add_argument('--structure_hops', type=int, default=1) 

    parser.add_argument('--use_node_cls', action='store_true')
    parser.add_argument('--use_cached_subgraph', action='store_true')
    parser.add_argument('--early_stop_patience',type=int, default=30)
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
    # g, df = load_graph(args.data)
    g1, g2, df1, df2 = load_graph(args.data)
    print(f"Loaded graphs and data from DATA/{args.data}")

    # Determine split indices based on one of the datasets (assuming they are aligned)
    args.train_edge_end = df1[df1['ext_roll'].gt(0)].index[0]
    args.val_edge_end   = df1[df1['ext_roll'].gt(1)].index[0]
    args.num_nodes1 = max(int(df1['src'].max()), int(df1['dst'].max())) + 1
    args.num_nodes2 = max(int(df2['src'].max()), int(df2['dst'].max())) + 1
    args.num_edges = len(df1) # the number of edges are same for both datasets

    print('Train %d, Valid %d, Test %d'%(args.train_edge_end, 
                                         args.val_edge_end-args.train_edge_end,
                                         len(df1)-args.val_edge_end))
    print('Num nodes for data1 %d, Num nodes for data2 %d, num edges %d' % (args.num_nodes1, args.num_nodes2, args.num_edges))
    
    # Load features (assuming node features are same for both datasets)
    node_feats, edge_feats1, edge_feats2  = load_feat(args.data)  # Modify as needed
    
    # Feature pre-processing
    node_feat_dims = 0 if node_feats is None else node_feats.shape[1]
    edge_feat1_dims = 0 if edge_feats1 is None else edge_feats1.shape[1]
    edge_feat2_dims = 0 if edge_feats2 is None else edge_feats2.shape[1]
    
    if args.use_onehot_node_feats:
        print('>>> Use one-hot node features')
        node_feats1 = torch.eye(args.num_nodes1)
        node_feat1_dims = node_feats1.size(1)
        node_feats2 = torch.eye(args.num_nodes2)
        node_feat2_dims = node_feats2.size(1)
    
    if args.ignore_node_feats:
        print('>>> Ignore node features')
        node_feats = None
        node_feat_dims = 0
    
    if args.use_type_feats:


        edge_type1 = df1.label.values
        args.num_edgeType1 = len(set(edge_type1.tolist()))
        edge_feats1 = torch.nn.functional.one_hot(torch.from_numpy(edge_type1-1), 
                                                 num_classes=args.num_edgeType1)
        edge_feat1_dims = edge_feats1.size(1)

        edge_type2 = df2.label.values
        args.num_edgeType2 = len(set(edge_type2.tolist()))
        edge_feats2 = torch.nn.functional.one_hot(torch.from_numpy(edge_type2-1), 
                                                 num_classes=args.num_edgeType2)
        edge_feat2_dims = edge_feats2.size(1)
    
    print('Node feature dim %d, Edges1 feature dim %d, Edges2 feature dim %d' % (node_feat_dims, edge_feat1_dims, edge_feat2_dims))
    
    # Data leakage check (assuming it should be applied to both datasets)
    if args.check_data_leakage:
        check_data_leakage(args, g1, df1)
    
    args.node_feat_dims = node_feat_dims
    args.edge_feat_dims = edge_feat1_dims
    
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

    if args.model == 'sthn':
        if args.predict_class:
            from model import Multiclass_Dual_Interface as STHN_Interface
        else:
            from model import Dual_Interface as STHN_Interface
        from link_pred_train_utils import link_pred_train_dual

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
        NotImplementedError
    
    model = STHN_Interface(mixer_configs, edge_predictor_configs)
    # for k, v in model.named_parameters():
    #     print(k, v.requires_grad)

    # print_model_info(model)

    return model, args, link_pred_train_dual
        
def objective(trial, args):
    """Define the hyperparameter search space and train the model with sampled parameters."""
    # Sample hyperparameters
    args.lr = trial.suggest_loguniform("lr", 1e-4, 1e-3)
    args.weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-4)
    args.dropout = trial.suggest_float("dropout", 0.0, 0.1)
    args.hidden_dims = trial.suggest_int("hidden_dims", 100, 300, step=50)
    args.time_dims = args.hidden_dims
    args.batch_size = trial.suggest_int("batch_size", 100, 300, step=50)
    # args.max_edges = trial.suggest_int("max_edges", 50, 500, step=50)
    args.channel_expansion_factor = trial.suggest_int("channel_expansion_factor", 1, 3)
    args.num_neighbors = trial.suggest_int("num_neighbors", 10, 100, step=10)
    args.optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])


    args.time_dims = args.hidden_dims

    # Load features and graphs
    try:
        node_feats, edge_feats1, edge_feats2, g1, g2, df1, df2, args = load_all_data(args)
    except Exception as e:
        print(f"Trial {trial.number} failed during data loading: {e}")
        return float("inf")

    # Load model
    model, args, link_pred_train_dual = load_model_dual(args)

    # Train model with error handling
    try:
        best_model, best_valid_loss = link_pred_train_dual(
            model.to(args.device), args, g1, g2, df1, df2, node_feats, edge_feats1, edge_feats2
        )
    except torch.cuda.OutOfMemoryError:
        print(f"Trial {trial.number} failed due to CUDA memory error. Retrying with smaller batch size.")
        args.batch_size = max(100, args.batch_size // 2)
        torch.cuda.empty_cache()
        gc.collect()
        return objective(trial, args)
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float("inf")
    finally:
        torch.cuda.empty_cache()
        gc.collect()

    # Save best model dynamically
    os.makedirs("models", exist_ok=True)
    if best_valid_loss < objective.best_loss:
        torch.save(best_model.state_dict(), f"params/best_model_trial_{trial.number}.pt")
        objective.best_loss = best_valid_loss

    # Log all trials
    with open("params/all_trials.json", "a") as f:
        json.dump({
            "trial": trial.number,
            "params": trial.params,
            "valid_loss": best_valid_loss
        }, f)
        f.write("\n")

    print(f"Trial {trial.number}: Loss {best_valid_loss:.4f}, Hyperparams: {trial.params}")
    return best_valid_loss

# Initialize best loss
objective.best_loss = float("inf")

####################################################################
# Main Execution
####################################################################

if __name__ == "__main__":
    args = get_args()

    # Set specific arguments related to graph structure and feature usage
    args.use_graph_structure = True
    args.ignore_node_feats = True
    args.use_type_feats = True
    args.use_cached_subgraph = True

    print(args)

    # Determine device
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.device = torch.device(args.device)

    # Set Optuna study
    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///models/optuna.db",
        study_name="link_pred_tuning",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(lambda trial: objective(trial, args), n_trials=100, n_jobs=1, show_progress_bar=True)

    # Save final best parameters and trial metadata
    with open("params/final_best_params.json", "w") as f:
        json.dump({
            "best_trial": study.best_trial.number,
            "best_params": study.best_params,
            "best_validation_loss": study.best_value
        }, f)

    print("Best hyperparameters:", study.best_params)
    print("Best validation loss:", study.best_value)