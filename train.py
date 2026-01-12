import argparse
from utils import set_seed, load_feat, load_graph
from data_process_utils import check_data_leakage
import pandas as pd
import random
import numpy as np
from pathlib import Path
import json
import torch
torch.use_deterministic_algorithms(True, warn_only=True)
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"   



####################################################################
####################################################################

def print_model_info(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters])
    print('Trainable Parameters: %d' % parameters)

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='FILT_HB', help="data directory")
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
    parser.add_argument('--use_node_feats', action='store_true')
    parser.add_argument('--enable_preference', dest='enable_preference', action='store_true',
                    help='Enable preference memory (default: True)')
    parser.add_argument('--disable_preference', dest='enable_preference', action='store_false',
                        help='Disable preference memory')
    parser.set_defaults(enable_preference=True)
    parser.add_argument('--use_atomic_group', action='store_true')
    parser.add_argument('--use_embedding', action='store_true',
                    help='Use nn.Embedding instead of one-hot')
    parser.add_argument('--emb_dim', type=int, default=100)

    parser.add_argument('--use_graph_structure', action='store_true')
    parser.add_argument('--structure_time_gap', type=int, default=2000)
    parser.add_argument('--structure_hops', type=int, default=1) 
    parser.add_argument('--use_cached_subgraph', action='store_true')
    parser.add_argument('--early_stop_patience', type=int, default=30)
    return parser.parse_args()


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
    print('Num nodes for data1 %d, Num nodes for data2 %d, num edges %d' % (args.num_nodes1, 
                                                                            args.num_nodes2, args.num_edges))
    
    # Load features 
    node_feats, edge_feats1, edge_feats2  = load_feat(args.data)  
    
    # Feature pre-processing
    node_feat_dims = 0 if node_feats is None else node_feats.shape[1]
    edge_feat1_dims = 0 if edge_feats1 is None else edge_feats1.shape[1]
    edge_feat2_dims = 0 if edge_feats2 is None else edge_feats2.shape[1]
    
    if args.use_node_feats:
        print('>>> Use node features')
        num_classes = int(node_feats.max().item())+1  
        node_feats = torch.nn.functional.one_hot(node_feats.to(torch.int64).squeeze(), num_classes=num_classes)
        node_feats = node_feats.to(torch.float32)
        node_feat_dims = node_feats.size(1)
        print('node feature dim =', (node_feats.shape))
    
    if args.ignore_node_feats:
        print('>>> Ignore node features')
        node_feats = None
        node_feat_dims = 0

    # Atomic Group Encoding
    pair_feats_combined1 = pair_feats_combined2 = None
    if args.use_atomic_group:
        num_pair_index = int(max(df1['dst_idx'].max(), df2['dst_idx'].max()) + 1)

        if getattr(args, 'use_embedding', False):
            # Save info for embedding path (indices only; actual embeddings used later in run_dual)
            args.num_pair_index = num_pair_index
            args.pair_index1 = torch.tensor(df1.dst_idx.values, dtype=torch.long)
            args.pair_index2 = torch.tensor(df2.dst_idx.values, dtype=torch.long)

            edge_feat1_dims = edge_feat2_dims = 2 * args.emb_dim
            edge_feats1 = None
            edge_feats2 = None
            print(f'Embedding path: num_pair_index={num_pair_index}, edge_feat_dims={edge_feat1_dims}')
        else:
            # One-hot path
            pair_index1 = torch.tensor(df1.dst_idx.values, dtype=torch.long)
            pair_index2 = torch.tensor(df2.dst_idx.values, dtype=torch.long)
            pair_feats1 = torch.nn.functional.one_hot(pair_index1, num_classes=num_pair_index).float()
            pair_feats2 = torch.nn.functional.one_hot(pair_index2, num_classes=num_pair_index).float()
            pair_diff1 = torch.abs(pair_feats1 - pair_feats2)
            pair_mul1  = pair_feats1 * pair_feats2
            pair_diff2 = torch.abs(pair_feats2 - pair_feats1)
            pair_mul2  = pair_feats2 * pair_feats1
            pair_feats_combined1 = torch.cat([pair_diff1, pair_mul1], dim=1)
            pair_feats_combined2 = torch.cat([pair_diff2, pair_mul2], dim=1)
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
    if args.use_atomic_group and getattr(args, 'use_embedding', False):
        args.edge_feat_dims = 2 * args.emb_dim
    elif edge_feats1 is not None and edge_feats2 is not None:
        args.edge_feat_dims = max(edge_feat1_dims, edge_feat2_dims)
    else:
        args.edge_feat_dims = 0
    
    # Move features to device
    if node_feats is not None:
        node_feats = node_feats.to(args.device)
    if edge_feats1 is not None:
        edge_feats1 = edge_feats1.to(args.device)
    if edge_feats2 is not None:
        edge_feats2 = edge_feats2.to(args.device)
    
    return node_feats, edge_feats1, edge_feats2, g1, g2, df1, df2, args


def load_model(args):
    # Define edge predictor configurations
    edge_predictor_configs = {
        'dim_in_time': args.time_dims,
        'dim_in_node': args.node_feat_dims,
        'predict_class': 1 if not args.predict_class else args.num_edgeType + 1
    }

    if args.model == 'DLP':
        from model import Dual_Interface
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
        raise NotImplementedError(f"Model {args.model} is not implemented.")


    model = Dual_Interface(
    mixer_configs,
    edge_predictor_configs,
    num_nodes=args.num_nodes1,
    enable_preference=args.enable_preference
)
    
    # Attach trainable embedding to the model 
    if getattr(args, 'use_atomic_group', False) and getattr(args, 'use_embedding', False):
        model.atomic_group_embedding = torch.nn.Embedding(args.num_pair_index, args.emb_dim).to(args.device)
        # Cache full pair indices on the model 
        model.register_buffer('pair_index1_full', args.pair_index1.to(args.device))
        model.register_buffer('pair_index2_full', args.pair_index2.to(args.device))
    else:
        model.atomic_group_embedding = None

    for k, v in model.named_parameters():
        print(k, v.requires_grad)

    print_model_info(model)

    return model, args, link_pred_train_dual
        
####################################################################
####################################################################

if __name__ == "__main__":
    args = get_args()

    args.use_graph_structure = True
    args.use_node_feats = True # Use node features
    args.use_atomic_group = True # Atomic group encoding
    args.use_cached_subgraph = True

    print(args)

    # device
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.device = torch.device(args.device)


    NUM_RUNS = 5
    BASE_SEED = 0 

    set_seed(BASE_SEED)
    # ###################################################
    # # Load features and graphs
    node_feats, edge_feats1, edge_feats2, g1, g2, df1, df2, args = load_all_data(args)

    run_summaries = []
    test_auroc, test_auprc, test_f1 = [], [], []
    train_time, test_time = [], []

    for run in range(NUM_RUNS):
        seed = BASE_SEED + run
        print(f"\n RUN {run+1}/{NUM_RUNS} (seed={seed})")
       
        set_seed(seed)

        # Load model

        model, args, link_pred_train_dual = load_model(args)

        results = link_pred_train_dual(
            model.to(args.device), args, g1, g2, df1, df2, node_feats, edge_feats1, edge_feats2
        )

        run_summaries.append({
            "run": run + 1,
            "seed": seed,
            # metrics
            "best_test_auc": results["best_test_auc"],
            "best_test_ap": results["best_test_ap"],
            "best_test_f1": results["best_test_f1"],
            "best_epoch": results["best_epoch"],
            "lowest_loss": results["lowest loss"],
            # time
            "Total train time": results["Total train time"],
            "Test time": results["Test time"],
            # memory / model size
            "no_params": results["no_params"],
            "no_buffers": results["no_buffers"],
            "param_size": results["param_size"],
            "buffer_size": results["buffer_size"],
            "total_memory": results["total_memory"],
        })

        test_auroc.append(results["best_test_auc"])
        test_auprc.append(results["best_test_ap"])
        test_f1.append(results["best_test_f1"])
        train_time.append(results["Total train time"])
        test_time.append(results["Test time"])

    # ---- summary stats (use sample std, ddof=1) ----
    def mean_std(x):
        x = np.asarray(x, dtype=np.float64)
        mu = float(np.mean(x))
        sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        return mu, sd

    mu_auc, sd_auc = mean_std(test_auroc)
    mu_ap,  sd_ap  = mean_std(test_auprc)
    mu_f1,  sd_f1  = mean_std(test_f1)

    def pm(mu, sd):
        return f"{mu:.4f} ± {sd:.4f}"

    print(f"AUROC: {pm(mu_auc, sd_auc)}")
    print(f"AUPRC: {pm(mu_ap,  sd_ap)}")
    print(f"F1:    {pm(mu_f1,  sd_f1)}")

    # keep memory
    mem0 = {k: run_summaries[0][k] for k in ["no_params","no_buffers","param_size","buffer_size","total_memory"]}

    summary = {
        "num_runs": NUM_RUNS,
        "metrics_test": {
            "AUROC": pm(mu_auc, sd_auc),
            "AUPRC": pm(mu_ap,  sd_ap),
            "F1":    pm(mu_f1,  sd_f1),
            "AUROC_mean": mu_auc, "AUROC_std": sd_auc,
            "AUPRC_mean": mu_ap,  "AUPRC_std": sd_ap,
            "F1_mean":    mu_f1,  "F1_std":    sd_f1,
        },
        "time": {
            "Total train time (mean)": float(np.mean(train_time)),
            "Total train time (std)":  float(np.std(train_time, ddof=1)) if NUM_RUNS > 1 else 0.0,
            "Test time (mean)":        float(np.mean(test_time)),
            "Test time (std)":         float(np.std(test_time, ddof=1)) if NUM_RUNS > 1 else 0.0,
        },
        "memory": mem0,
        "per_run": run_summaries,
    }

    # save JSON
    out_dir = Path("results") / str(args.data)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "runs_5.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")
    