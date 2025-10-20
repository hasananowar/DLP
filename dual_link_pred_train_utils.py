from tqdm import tqdm
import torch
import time
import copy
import json
import numpy as np
from torch_sparse import SparseTensor
from data_process_utils import pre_compute_subgraphs, get_random_inds, get_subgraph_sampler
from construct_subgraph import construct_mini_batch_giant_graph, print_subgraph_data
from utils import row_norm
from torchmetrics.classification import MulticlassAUROC, MulticlassAveragePrecision, MulticlassF1Score
from sklearn.metrics import f1_score
import logging
import math
from pathlib import Path
# Set up logging with proper format
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Example logging usage
logging.info("This is an info message")
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryF1Score
from sklearn.preprocessing import MinMaxScaler



rng1_train = np.random.default_rng(0)  # edges1 (train)
rng2_train = np.random.default_rng(1)  # edges2 (train)

# F1: need thresholded predictions
def prepare_preds_for_f1(preds: torch.Tensor, threshold: float):
    return (torch.sigmoid(preds) > threshold).long()

def run_dual(model, optimizer, args,
             subgraphs1, subgraphs2, df1, df2,
             node_feats, edge_feats1, edge_feats2,
             MLAUROC, MLAUPRC, MLF1, mode):
    """
    Executes a training or evaluation epoch for dual link prediction tasks.
    - TRAIN metrics are assumed to live on GPU (args.device)
    - EVAL (valid/test) metrics are assumed to live on CPU
    """

    # For validation only: keep a list of all preds+labels
    all_val_preds = []
    all_val_labels = []
    best_threshold = getattr(args, 'best_threshold', 0.5)
    time_epoch = 0.0

    # -----------------------------
    # Mode setup
    # -----------------------------
    if mode == 'train':
        model.train()
        cur_df1 = df1[:args.train_edge_end]
        cur_df2 = df2[:args.train_edge_end]
        neg_samples1 = args.neg_samples
        neg_samples2 = args.neg_samples
        cached_neg_samples1 = args.extra_neg_samples
        cached_neg_samples2 = args.extra_neg_samples
        cur_inds1 = 0
        cur_inds2 = 0
        # training uses the advancing RNG streams
        rng1_local = rng1_train
        rng2_local = rng2_train

    elif mode == 'valid':
        model.eval()
        cur_df1 = df1[args.train_edge_end:args.val_edge_end]
        cur_df2 = df2[args.train_edge_end:args.val_edge_end]
        neg_samples1 = 1
        neg_samples2 = 1
        cached_neg_samples1 = 1
        cached_neg_samples2 = 1
        cur_inds1 = args.train_edge_end
        cur_inds2 = args.train_edge_end
        # eval uses fixed RNGs (independent, reproducible every run)
        rng1_local = np.random.default_rng(4242)
        rng2_local = np.random.default_rng(4243)

    elif mode == 'test':
        model.eval()
        cur_df1 = df1[args.val_edge_end:]
        cur_df2 = df2[args.val_edge_end:]
        neg_samples1 = 1
        neg_samples2 = 1
        cached_neg_samples1 = 1
        cached_neg_samples2 = 1
        cur_inds1 = args.val_edge_end
        cur_inds2 = args.val_edge_end
        # test uses fixed RNGs (independent, reproducible every run)
        rng1_local = np.random.default_rng(4342)
        rng2_local = np.random.default_rng(4343)

    else:
        raise ValueError("Mode should be 'train', 'valid', or 'test'")

    # Create loaders and sanity check
    train_loader1 = cur_df1.groupby(cur_df1.index // args.batch_size)
    train_loader2 = cur_df2.groupby(cur_df2.index // args.batch_size)
    assert len(train_loader1) == len(train_loader2), "Mismatch in number of batches between subgraphs1 and subgraphs2"

    pbar = tqdm(total=len(train_loader1))
    pbar.set_description(f'{mode} mode with negative samples1 {neg_samples1} ...')

    # -----------------------------
    # Metrics/reset
    # -----------------------------
    subgraphs1, elabel1 = subgraphs1
    subgraphs2, elabel2 = subgraphs2

    loss_lst = []
    MLAUROC.reset()
    MLAUPRC.reset()
    MLF1.reset()

    # -----------------------------
    # Loop
    # -----------------------------
    for ind in range(len(train_loader1)):
        # -------- choose mini-batch items deterministically per mode --------
        if args.use_cached_subgraph is False and mode == 'train':
            # on-the-fly sampling path
            subgraph_data_list1 = subgraphs1.all_root_nodes[ind]
            subgraph_data_list2 = subgraphs2.all_root_nodes[ind]

            mini_batch_inds1 = get_random_inds(len(subgraph_data_list1),
                                               cached_neg_samples1, neg_samples1,
                                               rng=rng1_local)
            mini_batch_inds2 = get_random_inds(len(subgraph_data_list2),
                                               cached_neg_samples2, neg_samples2,
                                               rng=rng2_local)

            subgraph_data1 = subgraphs1.mini_batch(ind, mini_batch_inds1)
            subgraph_data2 = subgraphs2.mini_batch(ind, mini_batch_inds2)

        else:
            # cached path (valid/test, or train if you cached)
            subgraph_data_list1 = subgraphs1[ind]
            subgraph_data_list2 = subgraphs2[ind]

            mini_batch_inds1 = get_random_inds(len(subgraph_data_list1),
                                               cached_neg_samples1, neg_samples1,
                                               rng=rng1_local)
            mini_batch_inds2 = get_random_inds(len(subgraph_data_list2),
                                               cached_neg_samples2, neg_samples2,
                                               rng=rng2_local)

            subgraph_data1 = [subgraph_data_list1[i] for i in mini_batch_inds1]
            subgraph_data2 = [subgraph_data_list2[i] for i in mini_batch_inds2]

        # -------- build giant graphs --------
        subgraph_data1 = construct_mini_batch_giant_graph(subgraph_data1, args.max_edges)
        subgraph_data2 = construct_mini_batch_giant_graph(subgraph_data2, args.max_edges)

        # Edge timestamps
        subgraph_edts1 = torch.as_tensor(subgraph_data1['edts'], dtype=torch.float32, device=args.device)
        subgraph_edts2 = torch.as_tensor(subgraph_data2['edts'], dtype=torch.float32, device=args.device)
        merged_edge_ts = torch.cat([subgraph_edts1, subgraph_edts2], dim=0)

        # Raw edge feats
        def get_subgraph_edge_feats(edge_feats, subgraph_data, subgraph_edts, edge_feat_dims, device):
            if edge_feats is not None:
                eid_tensor = torch.as_tensor(subgraph_data['eid'], dtype=torch.long, device=edge_feats.device)
                return edge_feats[eid_tensor]
            else:
                return torch.zeros((subgraph_edts.shape[0], edge_feat_dims), device=device)

        subgraph_edge_feats1 = get_subgraph_edge_feats(edge_feats1, subgraph_data1, subgraph_edts1, args.edge_feat_dims, args.device)
        subgraph_edge_feats2 = get_subgraph_edge_feats(edge_feats2, subgraph_data2, subgraph_edts2, args.edge_feat_dims, args.device)
        merged_edge_feats = torch.cat([subgraph_edge_feats1, subgraph_edge_feats2], dim=0)

        # Node features (optional graph structure path)
        if args.use_graph_structure and node_feats is not None:
            num_of_df_links1 = len(subgraph_data_list1) // (cached_neg_samples1 + 2)
            subgraph_node_feats1 = compute_sign_feats(node_feats[:args.num_nodes1], df1, cur_inds1, num_of_df_links1,
                                                      subgraph_data1['root_nodes'], args, num_nodes=args.num_nodes1)
            cur_inds1 += num_of_df_links1

            num_of_df_links2 = len(subgraph_data_list2) // (cached_neg_samples2 + 2)
            subgraph_node_feats2 = compute_sign_feats(node_feats, df2, cur_inds2, num_of_df_links2,
                                                      subgraph_data2['root_nodes'], args, num_nodes=args.num_nodes2)
            cur_inds2 += num_of_df_links2
        else:
            subgraph_node_feats1 = None
            subgraph_node_feats2 = None

        # Scale edge timestamps to [0, 1000]
        min1, max1 = subgraph_edts1.min(), subgraph_edts1.max()
        span1 = (max1 - min1).clamp(min=1e-6)
        subgraph_edts1 = ((subgraph_edts1 - min1) / span1) * 1000

        min2, max2 = subgraph_edts2.min(), subgraph_edts2.max()
        span2 = (max2 - min2).clamp(min=1e-6)
        subgraph_edts2 = ((subgraph_edts2 - min2) / span2) * 1000

        # Compute inds1/inds2 based on all_edge_indptr
        all_inds1, has_temporal_neighbors1 = [], []
        aei1 = subgraph_data1['all_edge_indptr']
        for i in range(len(aei1) - 1):
            num_edges1 = aei1[i+1] - aei1[i]
            all_inds1.extend([args.max_edges * i + j for j in range(num_edges1)])
            has_temporal_neighbors1.append(num_edges1 > 0)

        all_inds2, has_temporal_neighbors2 = [], []
        aei2 = subgraph_data2['all_edge_indptr']
        for i in range(len(aei2) - 1):
            num_edges2 = aei2[i+1] - aei2[i]
            all_inds2.extend([args.max_edges * i + j for j in range(num_edges2)])
            has_temporal_neighbors2.append(num_edges2 > 0)

        merged_batch_size = len(has_temporal_neighbors1) + len(has_temporal_neighbors2)

        # Merge node feats if present
        if subgraph_node_feats1 is None and subgraph_node_feats2 is None:
            merged_node_feats = None
        elif subgraph_node_feats1 is None:
            merged_node_feats = subgraph_node_feats2
        elif subgraph_node_feats2 is None:
            merged_node_feats = subgraph_node_feats1
        else:
            merged_node_feats = torch.cat([subgraph_node_feats1, subgraph_node_feats2], dim=0)

        # Edge labels for dataset2 (mask)
        mask = (elabel2[ind] == 1)
        subgraph_edge_type = torch.from_numpy(mask.astype(np.int64)).to(args.device)

        # Pack model inputs
        inputs = [
            merged_edge_feats.to(args.device),
            merged_edge_ts.to(args.device),
            merged_batch_size,
            torch.tensor(all_inds1 + all_inds2, dtype=torch.long, device=args.device),
            torch.tensor(list(subgraph_edge_type), dtype=torch.long, device=args.device),
        ]

        # Source node ID tensors
        if 'root_nodes' not in subgraph_data1 or 'root_nodes' not in subgraph_data2:
            raise KeyError("Both subgraph_data1 and subgraph_data2 must include 'root_nodes'")


        src_ids1 = torch.as_tensor(subgraph_data1['root_nodes'], dtype=torch.long, device=args.device)
        src_ids2 = torch.as_tensor(subgraph_data2['root_nodes'], dtype=torch.long, device=args.device)
        src_ts1  = torch.as_tensor(subgraph_data1['root_times'], dtype=torch.float32, device=args.device)
        src_ts2  = torch.as_tensor(subgraph_data2['root_times'], dtype=torch.float32, device=args.device)
        inputs.append(src_ids1)
        inputs.append(src_ids2)
        inputs.append(src_ts1)
        inputs.append(src_ts2)

        # Forward + optimize
        start_time = time.time()
        loss, pred, edge_label = model(
            inputs,
            neg_samples=max(neg_samples1, neg_samples2),
            node_feats=merged_node_feats
        )

        if mode == 'train' and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()  
            optimizer.step()
        time_epoch += (time.time() - start_time)

       
        if mode == 'train':
            # GPU metrics
            _pred  = pred.detach().cpu()
            _label = edge_label.detach().cpu()
            MLAUROC.update(_pred, _label)
            MLAUPRC.update(_pred, _label)
            bin_preds = prepare_preds_for_f1(_pred, best_threshold).squeeze()
            MLF1.update(bin_preds, _label.view(-1).long())
        else:
            # CPU metrics (avoid non-deterministic CUDA cumsum)
            _pred  = pred.detach().cpu()
            _label = edge_label.detach().cpu()
            MLAUROC.update(_pred, _label)
            MLAUPRC.update(_pred, _label)

            if mode == 'valid':
                all_val_preds.append(_pred)
                all_val_labels.append(_label)
            else:
                # test F1 at best known threshold
                bin_preds = prepare_preds_for_f1(_pred, best_threshold).squeeze()
                MLF1.update(bin_preds, _label.view(-1).long())

        # Accumulate loss
        loss_lst.append(float(loss))
        pbar.update(1)

    pbar.close()

    total_auroc = MLAUROC.compute()
    total_auprc = MLAUPRC.compute()

    if mode == 'valid':
        val_preds  = torch.cat(all_val_preds).view(-1)           # CPU
        val_labels = torch.cat(all_val_labels).view(-1)          # CPU
        val_probs  = torch.sigmoid(val_preds).tolist()

        best_tau, best_f1 = 0.5, -1.0
        for τ in np.linspace(0.0, 1.0, 101):
            pred_bin = [1 if prob >= τ else 0 for prob in val_probs]
            f1       = f1_score(val_labels, pred_bin)
            if f1 > best_f1:
                best_f1, best_tau = f1, τ

        args.best_threshold = best_tau
        # F1 at best_tau for reporting
        val_bin  = [1 if prob >= best_tau else 0 for prob in val_probs]
        total_f1 = f1_score(val_labels, val_bin)
    else:
        total_f1 = MLF1.compute().item()

    # Scalars
    if torch.is_tensor(total_auroc): total_auroc = total_auroc.item()
    if torch.is_tensor(total_auprc): total_auprc = total_auprc.item()
    if torch.is_tensor(total_f1):    total_f1    = total_f1.item()

    print('%s mode with time %.4f, AUROC %.4f, AUPRC %.4f, F1 %.4f, loss %.4f'
          % (mode, time_epoch, total_auroc, total_auprc, total_f1, float(np.mean(loss_lst))))
    return total_auroc, total_auprc, total_f1, float(np.mean(loss_lst)), time_epoch


# def link_pred_train_dual(model, args, g1, g2, df1, df2, node_feats, edge_feats1, edge_feats2):
#     """
#     Train the model for dual link prediction tasks.

#     Returns:
#         best_auc_model: The best-performing model based on validation loss.
#     """
#     import copy, json
#     from pathlib import Path
#     import torch

#     # -----------------------------
#     # Optimizer + step counter wrap
#     # -----------------------------
#     base_opt = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

#     class _OptWrapper:
#         """Proxy that increments model.global_step every time step() is called."""
#         def __init__(self, opt, model_ref):
#             self._opt = opt
#             self._model = model_ref
#         def __getattr__(self, name):
#             return getattr(self._opt, name)
#         def step(self, *a, **kw):
#             out = self._opt.step(*a, **kw)
#             self._model.global_step = getattr(self._model, "global_step", 0) + 1
#             return out
#         def zero_grad(self, *a, **kw):
#             return self._opt.zero_grad(*a, **kw)

#     optimizer = _OptWrapper(base_opt, model)

#     # -----------------------------------
#     # Memory read warm-up / read-dropout
#     # -----------------------------------
#     # Set sensible defaults if args don't have these fields
#     model.mem_read_warmup = getattr(args, "mem_read_warmup", 1000)  # steps to keep reads OFF
#     model.p_read_drop     = getattr(args, "p_read_drop", 0.10)      # prob to skip reads (train only)
#     model.global_step     = 0                                       # start counter at 0

#     # -----------------
#     # Subgraph samplers
#     # -----------------
#     if args.use_cached_subgraph:
#         train_subgraphs1 = pre_compute_subgraphs(args, g1, df1, mode='train', input_data='edges1')
#         train_subgraphs2 = pre_compute_subgraphs(args, g2, df2, mode='train', input_data="edges2")
#     else:
#         train_subgraphs1 = get_subgraph_sampler(args, g1, df1, mode='train', input_data='edges1')
#         train_subgraphs2 = get_subgraph_sampler(args, g2, df2, mode='train', input_data='edges2')

#     valid_subgraphs1 = pre_compute_subgraphs(args, g1, df1, mode='valid', input_data='edges1')
#     valid_subgraphs2 = pre_compute_subgraphs(args, g2, df2, mode='valid', input_data="edges2")
#     test_subgraphs1  = pre_compute_subgraphs(args, g1, df1, mode='test',  input_data='edges1')
#     test_subgraphs2  = pre_compute_subgraphs(args, g2, df2, mode='test',  input_data='edges2')

#     # ------------
#     # Metric setup
#     # ------------
#     all_results = {
#         'train_ap': [], 'valid_ap': [], 'test_ap': [],
#         'train_auc': [], 'valid_auc': [], 'test_auc': [],
#         'train_f1': [], 'valid_f1': [], 'test_f1': [],
#         'train_loss': [], 'valid_loss': [], 'test_loss': [],
#     }

#     low_loss = float('inf')
#     user_train_total_time = 0.0
#     user_epoch_num = 0
#     best_epoch = -1
#     best_test_auc = best_test_ap = best_test_f1 = 0.0
#     best_auc_model = copy.deepcopy(model).cpu()  # safe fallback in case val never improves

#     if args.predict_class:
#         num_classes = args.num_edgeType + 1
#         train_AUROC = MulticlassAUROC(num_classes, average="macro", thresholds=None)
#         valid_AUROC = MulticlassAUROC(num_classes, average="macro", thresholds=None)
#         test_AUROC  = MulticlassAUROC(num_classes, average="macro", thresholds=None)
#         train_AUPRC = MulticlassAveragePrecision(num_classes, average="macro", thresholds=None)
#         valid_AUPRC = MulticlassAveragePrecision(num_classes, average="macro", thresholds=None)
#         test_AUPRC  = MulticlassAveragePrecision(num_classes, average="macro", thresholds=None)
#         train_F1    = MulticlassF1Score(num_classes, average="macro")
#         valid_F1    = MulticlassF1Score(num_classes, average="macro")
#         test_F1     = MulticlassF1Score(num_classes, average="macro")
#     else:
#         train_AUROC = BinaryAUROC(thresholds=None)
#         train_AUPRC = BinaryAveragePrecision(thresholds=None)
#         train_F1    = BinaryF1Score()

#         valid_AUROC = BinaryAUROC(thresholds=None)
#         valid_AUPRC = BinaryAveragePrecision(thresholds=None)
#         valid_F1    = BinaryF1Score()

#         test_AUROC  = BinaryAUROC(thresholds=None)
#         test_AUPRC  = BinaryAveragePrecision(thresholds=None)
#         test_F1     = BinaryF1Score()

#     # -------------
#     # Training loop
#     # -------------
#     for epoch in range(args.epochs):
#         print(f'>>> Epoch {epoch + 1}')

#         # -------- Train --------
#         model.train()  # enables read-dropout; predict() checks self.training
#         train_auc, train_ap, train_f1, train_loss, time_train = run_dual(
#             model, optimizer, args, train_subgraphs1, train_subgraphs2,
#             df1, df2, node_feats, edge_feats1, edge_feats2,
#             train_AUROC, train_AUPRC, train_F1, mode='train'
#         )

#         # -------- Validate --------
#         model.eval()   # disable read-dropout and any BatchNorm/Dropout
#         with torch.no_grad():
#             valid_auc, valid_ap, valid_f1, valid_loss, time_valid = run_dual(
#                 model, None, args, valid_subgraphs1, valid_subgraphs2,
#                 df1, df2, node_feats, edge_feats1, edge_feats2,
#                 valid_AUROC, valid_AUPRC, valid_F1, mode='valid'
#             )

#             test_auc, test_ap, test_f1, test_loss, time_test = run_dual(
#                 model, None, args, test_subgraphs1, test_subgraphs2,
#                 df1, df2, node_feats, edge_feats1, edge_feats2,
#                 test_AUROC, test_AUPRC, test_F1, mode='test'
#             )

#         # -------- Track best by valid loss --------
#         if valid_loss < low_loss:
#             low_loss = float(valid_loss)
#             best_epoch = epoch
#             best_test_auc, best_test_ap, best_test_f1 = float(test_auc), float(test_ap), float(test_f1)
#             best_auc_model = copy.deepcopy(model).cpu()

#         user_train_total_time += (time_train + time_valid)
#         user_epoch_num += 1

#         # Early stop if no improvement for 30 epochs
#         if epoch > best_epoch + 30:
#             break

#         # Logs
#         all_results['train_ap'].append(float(train_ap))
#         all_results['valid_ap'].append(float(valid_ap))
#         all_results['test_ap'].append(float(test_ap))

#         all_results['train_auc'].append(float(train_auc))
#         all_results['valid_auc'].append(float(valid_auc))
#         all_results['test_auc'].append(float(test_auc))

#         all_results['train_f1'].append(float(train_f1))
#         all_results['valid_f1'].append(float(valid_f1))
#         all_results['test_f1'].append(float(test_f1))

#         all_results['train_loss'].append(float(train_loss))
#         all_results['valid_loss'].append(float(valid_loss))
#         all_results['test_loss'].append(float(test_loss))

#         print(f"Train: AUROC {train_auc:.4f}, AUPRC {train_ap:.4f}, F1 {train_f1:.4f}, Loss {train_loss:.4f}")
#         print(f"Valid: AUROC {valid_auc:.4f}, AUPRC {valid_ap:.4f}, F1 {valid_f1:.4f}, Loss {valid_loss:.4f}")
#         print(f"Test : AUROC {test_auc:.4f}, AUPRC {test_ap:.4f}, F1 {test_f1:.4f}, Loss {test_loss:.4f}")

#     print(f'Best Epoch: {best_epoch}, '
#           f'Best Test AUROC: {best_test_auc:.4f}, '
#           f'Best Test AUPRC: {best_test_ap:.4f}, '
#           f'Best Test F1: {best_test_f1:.4f}, '
#           f'Best Valid Loss: {low_loss:.4f}')

#     # Save the best metrics and epoch info
#     results = {
#         'Total epochs': user_epoch_num,
#         'best_epoch': best_epoch,
#         'best_test_auc': best_test_auc,
#         'best_test_ap': best_test_ap,
#         'best_test_f1': best_test_f1,
#         'lowest loss': low_loss,
#         'Total train time': user_train_total_time,
#         'mem_read_warmup': model.mem_read_warmup,
#         'p_read_drop': model.p_read_drop,
#     }
#     save_result_folder = Path("results") / f"no_memory_{str(args.no_memory).lower()}" / str(args.data)
#     save_result_folder.mkdir(parents=True, exist_ok=True)
#     save_result_path = save_result_folder / f"node{args.use_node_feats}_pair{args.use_pair_index}.json"
#     with open(save_result_path, "w") as f:
#         json.dump(results, f, indent=2)
#     print(f"Results written to {save_result_path}")

#     return best_auc_model




def link_pred_train_dual(model, args, g1, g2, df1, df2, node_feats, edge_feats1, edge_feats2):
    """
    Train the model for dual link prediction tasks.
    
    Returns:
        best_auc_model: The best-performing model based on validation loss.
    """
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_auc_model = None


    ###################################################
    # Get cached subgraphs
    if args.use_cached_subgraph:
        train_subgraphs1 = pre_compute_subgraphs(args, g1, df1, mode='train', input_data='edges1')
        train_subgraphs2 = pre_compute_subgraphs(args, g2, df2, mode='train', input_data="edges2")
    else:
        train_subgraphs1 = get_subgraph_sampler(args, g1, df1, mode='train', input_data='edges1')
        train_subgraphs2 = get_subgraph_sampler(args, g2, df2, mode='train', input_data='edges2')

    valid_subgraphs1 = pre_compute_subgraphs(args, g1, df1, mode='valid', input_data='edges1')
    valid_subgraphs2 = pre_compute_subgraphs(args, g2, df2, mode='valid', input_data="edges2")
    test_subgraphs1  = pre_compute_subgraphs(args, g1, df1, mode='test', input_data='edges1')
    test_subgraphs2  = pre_compute_subgraphs(args, g2, df2, mode='test', input_data="edges2")
    
    ###################################################
    # Initialize metrics
    all_results = {
        'train_ap': [],
        'valid_ap': [],
        'test_ap' : [],
        'train_auc': [],
        'valid_auc': [],
        'test_auc' : [],
        'train_f1': [],
        'valid_f1': [],
        'test_f1' : [],
        'train_loss': [],
        'valid_loss': [],
        'test_loss': [],
    }
    
    low_loss = 100000
    user_train_total_time = 0
    user_epoch_num = 0
    best_epoch = -1
    best_test_auc, best_test_ap = 0, 0
    
    if args.predict_class:
        num_classes = args.num_edgeType + 1
        train_AUROC = MulticlassAUROC(num_classes, average="macro", thresholds=None)
        valid_AUROC = MulticlassAUROC(num_classes, average="macro", thresholds=None)
        test_AUROC = MulticlassAUROC(num_classes, average="macro", thresholds=None)
        train_AUPRC = MulticlassAveragePrecision(num_classes, average="macro", thresholds=None)
        valid_AUPRC = MulticlassAveragePrecision(num_classes, average="macro", thresholds=None)
        test_AUPRC = MulticlassAveragePrecision(num_classes, average="macro", thresholds=None)
        train_F1 = MulticlassF1Score(num_classes, average="macro")
        valid_F1 = MulticlassF1Score(num_classes, average="macro")
        test_F1 = MulticlassF1Score(num_classes, average="macro")
    else:
        train_AUROC = BinaryAUROC(thresholds=None)            # CPU
        train_AUPRC = BinaryAveragePrecision(thresholds=None) # CPU
        train_F1    = BinaryF1Score()                         # CPU

        valid_AUROC = BinaryAUROC(thresholds=None)            # CPU
        valid_AUPRC = BinaryAveragePrecision(thresholds=None) # CPU
        valid_F1    = BinaryF1Score()                         # CPU

        test_AUROC  = BinaryAUROC(thresholds=None)            # CPU
        test_AUPRC  = BinaryAveragePrecision(thresholds=None) # CPU
        test_F1     = BinaryF1Score()                         # CPU
        
    
    ###################################################
    # Training loop
    for epoch in range(args.epochs):
        print(f'>>> Epoch {epoch + 1}')

        
        # Train
        train_auc, train_ap, train_f1, train_loss, time_train = run_dual(
            model, optimizer, args, train_subgraphs1, train_subgraphs2,
            df1,df2, node_feats, edge_feats1, edge_feats2,
            train_AUROC, train_AUPRC, train_F1, mode='train')
        
        # Validate

        with torch.no_grad():
            valid_auc, valid_ap, valid_f1, valid_loss, time_valid = run_dual(
                copy.deepcopy(model), None, args, valid_subgraphs1, valid_subgraphs2,
                df1, df2, node_feats, edge_feats1, edge_feats2,
                valid_AUROC, valid_AUPRC, valid_F1, mode='valid')
            
            test_auc, test_ap, test_f1, test_loss, time_test = run_dual(
                copy.deepcopy(model), None, args, test_subgraphs1, test_subgraphs2,
                df1, df2, node_feats, edge_feats1, edge_feats2,
                test_AUROC, test_AUPRC, test_F1, mode='test')
        
        # Check for improvement
        if valid_loss < low_loss:
            best_auc_model = copy.deepcopy(model).cpu()
            low_loss = valid_loss
            best_epoch = epoch
            best_test_auc, best_test_ap, best_test_f1 = test_auc, test_ap, test_f1
            
        user_train_total_time += time_train + time_valid
        user_epoch_num += 1
        if epoch > best_epoch + 30:
            break
        
        all_results['train_ap'].append(train_ap)
        all_results['valid_ap'].append(valid_ap)
        all_results['test_ap'].append(test_ap)
        
        all_results['valid_auc'].append(valid_auc)
        all_results['train_auc'].append(train_auc)
        all_results['test_auc'].append(test_auc)
        
        all_results['train_f1'].append(train_f1)
        all_results['valid_f1'].append(valid_f1)
        all_results['test_f1'].append(test_f1)
        
        all_results['train_loss'].append(train_loss)
        all_results['valid_loss'].append(valid_loss)
        all_results['test_loss'].append(test_loss)
        
        print(f"Train: AUROC1 {train_auc:.4f}, AUPRC {train_ap:.4f}, F1 {train_f1:.4f}, Loss {train_loss:.4f}")
        print(f"Valid: AUROC1 {valid_auc:.4f}, AUPRC {valid_ap:.4f}, F1 {valid_f1:.4f}, Loss {valid_loss:.4f}")
        print(f"Test: AUROC1 {test_auc:.4f}, AUPRC {test_ap:.4f}, F1 {test_f1:.4f}, Loss {test_loss:.4f}")
    
    print(f'Best Epoch: {best_epoch}, Best Test AUROC: {best_test_auc:.4f}, Best Test AUPRC: {best_test_ap:.4f}, Best Test F1: {best_test_f1:.4f}, Best Valid Loss: {low_loss:.4f}')
    
    # Save the best metrics and epoch information
    results = {
        'Total epochs': user_epoch_num,
        'best_epoch': best_epoch,
        'best_test_auc': best_test_auc,
        'best_test_ap': best_test_ap,
        'best_test_f1': best_test_f1,
        'lowest loss': low_loss,
        'Total train time': user_train_total_time
    }
    save_result_folder = Path("results") / f"no_memory_{str(args.no_memory).lower()}" / str(args.data)
    save_result_folder.mkdir(parents=True, exist_ok=True)
    save_result_path = save_result_folder / f"node{args.use_node_feats}_pair{args.use_pair_index}.json"
    with open(save_result_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results written to {save_result_path}")
    
    return best_auc_model

def compute_sign_feats(node_feats, df, start_i, num_links, root_nodes, args, num_nodes):
    root_nodes = torch.as_tensor(root_nodes, dtype=torch.long, device=args.device)
    num_duplicate = len(root_nodes) // num_links 
    # num_nodes = args.num_nodes

    root_inds = torch.arange(len(root_nodes)).view(num_duplicate, -1)
    root_inds = [arr.flatten() for arr in root_inds.chunk(1, dim=1)]

    output_feats = torch.zeros((len(root_nodes), node_feats.size(1))).to(args.device)
    i = start_i

    for _root_ind in root_inds:

        if i == 0 or args.structure_hops == 0:
            sign_feats = node_feats.clone()
        else:
            prev_i = max(0, i - args.structure_time_gap)
            cur_df = df[prev_i: i] # get adj's row, col indices (as undirected)
  
            src = torch.as_tensor(cur_df["src"].values, dtype=torch.long, device=args.device)
            dst = torch.as_tensor(cur_df["dst"].values, dtype=torch.long, device=args.device)
            edge_index = torch.stack([
                torch.cat([src, dst]), 
                torch.cat([dst, src])
            ])
            edge_index, edge_cnt = torch.unique(edge_index, dim=1, return_counts=True) 
            mask = edge_index[0]!=edge_index[1] 
            adj = SparseTensor(
                value = torch.ones_like(edge_cnt[mask]).float(),
                row = edge_index[0][mask].long(),
                col = edge_index[1][mask].long(),
                sparse_sizes=(num_nodes, num_nodes)
            )
            adj_norm = row_norm(adj).to(args.device)
            sign_feats = [node_feats]
            for _ in range(args.structure_hops):
                sign_feats.append(adj_norm@sign_feats[-1])
            sign_feats = torch.sum(torch.stack(sign_feats), dim=0)

        output_feats[_root_ind] = sign_feats[root_nodes[_root_ind]]

        i += len(_root_ind) // num_duplicate

    return output_feats
