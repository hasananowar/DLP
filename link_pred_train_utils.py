from tqdm import tqdm
import torch
import time
import copy
import numpy as np
from torch_sparse import SparseTensor
from data_process_utils import pre_compute_subgraphs, get_random_inds, get_subgraph_sampler
from construct_subgraph import construct_mini_batch_giant_graph, print_subgraph_data
from utils import row_norm
from torchmetrics.classification import MulticlassAUROC, MulticlassAveragePrecision
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from sklearn.preprocessing import MinMaxScaler


def run_dual(model, optimizer, args, subgraphs1, subgraphs2, df1, df2, node_feats, edge_feats1, edge_feats2, 
             MLAUROC1, MLAUPRC1, MLAUROC2, MLAUPRC2, mode):
    """
    Executes a training or evaluation epoch for dual link prediction tasks.
    """
    time_epoch = 0
    ###################################################
    # Setup modes
    if mode == 'train':
        model.train()
        cur_df1 = df1[:args.train_edge_end]
        cur_df2 = df2[:args.train_edge_end]
        neg_samples1 = args.neg_samples
        neg_samples2 = args.neg_samples  # Assuming same number of neg_samples for both
        cached_neg_samples1 = args.extra_neg_samples
        cached_neg_samples2 = args.extra_neg_samples
        cur_inds1 = 0
        cur_inds2 = 0
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
    else:
        raise ValueError("Mode should be 'train', 'valid', or 'test'")
    
    # Create separate loaders for subgraphs1 and subgraphs2
    train_loader1 = cur_df1.groupby(cur_df1.index // args.batch_size)
    train_loader2 = cur_df2.groupby(cur_df2.index // args.batch_size)
    
    # Ensure both loaders have the same number of batches
    assert len(train_loader1) == len(train_loader2), "Mismatch in number of batches between subgraphs1 and subgraphs2"
    
    pbar = tqdm(total=len(train_loader1))
    pbar.set_description('%s mode with negative samples1 %d ...'%(mode, neg_samples1))
        
    ###################################################
    # Initialize variables for loss and metrics
    subgraphs1, elabel1 = subgraphs1
    subgraphs2, elabel2 = subgraphs2
    loss_lst = []
    MLAUROC1.reset()
    MLAUPRC1.reset()
    MLAUROC2.reset()
    MLAUPRC2.reset()
    scaler1 = MinMaxScaler()
    scaler2= MinMaxScaler()
    
    for ind in range(len(train_loader1)):
        ###################################################
        if args.use_cached_subgraph == False and mode == 'train':
            subgraph_data_list1 = subgraphs1.all_root_nodes[ind]
            subgraph_data_list2 = subgraphs2.all_root_nodes[ind]
            
            mini_batch_inds1 = get_random_inds(len(subgraph_data_list1), cached_neg_samples1, neg_samples1)
            mini_batch_inds2 = get_random_inds(len(subgraph_data_list2), cached_neg_samples2, neg_samples2)
            
            subgraph_data1 = subgraphs1.mini_batch(ind, mini_batch_inds1)
            subgraph_data2 = subgraphs2.mini_batch(ind, mini_batch_inds2)
            
        else: # valid + test
            print('using cached subgraph')
            subgraph_data_list1 = subgraphs1[ind]
            subgraph_data_list2 = subgraphs2[ind]
            
            mini_batch_inds1 = get_random_inds(len(subgraph_data_list1), cached_neg_samples1, neg_samples1)
            mini_batch_inds2 = get_random_inds(len(subgraph_data_list2), cached_neg_samples2, neg_samples2)
            
            subgraph_data1 = [subgraph_data_list1[i] for i in mini_batch_inds1]
            subgraph_data2 = [subgraph_data_list2[i] for i in mini_batch_inds2]
        
        # Construct mini-batch giant graphs for both subgraphs
        subgraph_data1 = construct_mini_batch_giant_graph(subgraph_data1, args.max_edges)
        subgraph_data2 = construct_mini_batch_giant_graph(subgraph_data2, args.max_edges)
        
        
        ###################################################
        # Raw edge feats 
        subgraph_edge_feats1 = edge_feats1[subgraph_data1['eid']]
        subgraph_edts1 = torch.from_numpy(subgraph_data1['edts']).float()

        
        subgraph_edge_feats2 = edge_feats2[subgraph_data2['eid']]
        subgraph_edts2 = torch.from_numpy(subgraph_data2['edts']).float()
        
        ###################################################
        # Handle node features if required
        if args.use_graph_structure and node_feats:
            num_of_df_links1 = len(subgraph_data_list1) //  (cached_neg_samples1 + 2)   
            subgraph_node_feats1 = compute_sign_feats(node_feats, df1, cur_inds1, num_of_df_links1, subgraph_data1['root_nodes'], args)
            cur_inds1 += num_of_df_links1
            
            num_of_df_links2 = len(subgraph_data_list2) //  (cached_neg_samples2 + 2)   
            subgraph_node_feats2 = compute_sign_feats(node_feats, df2, cur_inds2, num_of_df_links2, subgraph_data2['root_nodes'], args)
            cur_inds2 += num_of_df_links2
        else:
            subgraph_node_feats1 = None
            subgraph_node_feats2 = None
        
        ###################################################
        # Scale edge timestamps
        scaler1.fit(subgraph_edts1.reshape(-1,1))
        subgraph_edts1 = scaler1.transform(subgraph_edts1.reshape(-1,1)).ravel().astype(np.float32) * 1000
        subgraph_edts1 = torch.from_numpy(subgraph_edts1)
        
        scaler2.fit(subgraph_edts2.reshape(-1,1))
        subgraph_edts2 = scaler2.transform(subgraph_edts2.reshape(-1,1)).ravel().astype(np.float32) * 1000
        subgraph_edts2 = torch.from_numpy(subgraph_edts2)



        # # Ensure subgraph_edts1 and subgraph_edts2 are properly handled
        # subgraph_edts1 = np.array(subgraph_edts1) if not isinstance(subgraph_edts1, np.ndarray) else subgraph_edts1
        # subgraph_edts2 = np.array(subgraph_edts2) if not isinstance(subgraph_edts2, np.ndarray) else subgraph_edts2

        # # Handle subgraph_edts1
        # if subgraph_edts1.size > 0:  # Ensure size attribute works correctly
        #     scaler1.fit(subgraph_edts1.reshape(-1, 1))  # Fit the scaler
        #     subgraph_edts1 = scaler1.transform(subgraph_edts1.reshape(-1, 1)).ravel().astype(np.float32) * 1000
        #     subgraph_edts1 = torch.from_numpy(subgraph_edts1)  # Convert to torch tensor
        # else:
        #     print("Warning: subgraph_edts1 is empty. Creating an empty tensor.")
        #     subgraph_edts1 = torch.tensor([], dtype=torch.float32)

        # # Handle subgraph_edts2
        # if subgraph_edts2.size > 0:  # Ensure size attribute works correctly
        #     scaler2.fit(subgraph_edts2.reshape(-1, 1))  # Fit the scaler
        #     subgraph_edts2 = scaler2.transform(subgraph_edts2.reshape(-1, 1)).ravel().astype(np.float32) * 1000
        #     subgraph_edts2 = torch.from_numpy(subgraph_edts2)  # Convert to torch tensor
        # else:
        #     print("Warning: subgraph_edts2 is empty. Creating an empty tensor.")
        #     subgraph_edts2 = torch.tensor([], dtype=torch.float32)



        ###################################################
        # Compute inds1 and inds2 based on all_edge_indptr
        all_inds1 = []
        has_temporal_neighbors1 = []
        all_edge_indptr1 = subgraph_data1['all_edge_indptr']
        for i in range(len(all_edge_indptr1) -1):
            num_edges1 = all_edge_indptr1[i+1] - all_edge_indptr1[i]
            all_inds1.extend([args.max_edges * i + j for j in range(num_edges1)])
            has_temporal_neighbors1.append(num_edges1 > 0)

        all_inds2 = []
        has_temporal_neighbors2 = []
        all_edge_indptr2 = subgraph_data2['all_edge_indptr']
        for i in range(len(all_edge_indptr2) -1):
            num_edges2 = all_edge_indptr2[i+1] - all_edge_indptr2[i]
            all_inds2.extend([args.max_edges * i + j for j in range(num_edges2)])
            has_temporal_neighbors2.append(num_edges2 > 0)
        
        ###################################################
        # Prepare inputs for the model
        model_inputs1 = [
            subgraph_edge_feats1.to(args.device),    # [number_of_edges, edge_dims]
            subgraph_edts1.to(args.device),          # [number_of_edges]
            len(has_temporal_neighbors1), # list of booleans
            torch.tensor(all_inds1).long()                    # [number_of_edges]
        ]
        
        model_inputs2 = [
            subgraph_edge_feats2.to(args.device),    # [number_of_edges, edge_dims]
            subgraph_edts2.to(args.device),          # [number_of_edges]
            len(has_temporal_neighbors2), # list of booleans
            torch.tensor(all_inds2).long()                    # [number_of_edges]
        ]
        
        start_time = time.time()
        
        # Forward pass through the model
        loss, preds1, edge_labels1, preds2, edge_labels2 = model(
            model_inputs1, 
            model_inputs2, 
            neg_samples=max(neg_samples1, neg_samples2),  # Ensure consistency
            node_feats=subgraph_node_feats1  # Assuming node_feats1 and node_feats2 are similar
        )
        
        if mode == 'train' and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        time_epoch += (time.time() - start_time)
        
        ###################################################
        # Directly pass logits to metrics; they will apply sigmoid internally
        MLAUROC1.update(preds1, edge_labels1)
        MLAUPRC1.update(preds1, edge_labels1)
        
        MLAUROC2.update(preds2, edge_labels2)
        MLAUPRC2.update(preds2, edge_labels2)
        
        # Accumulate loss
        loss_lst.append(float(loss))

        pbar.update(1)

    pbar.close()    
    
    # Compute final metrics
    total_auroc1 = MLAUROC1.compute()
    total_auprc1 = MLAUPRC1.compute()
    total_auroc2 = MLAUROC2.compute()
    total_auprc2 = MLAUPRC2.compute()
    
    print('%s mode with time %.4f, AUROC1 %.4f, AUPRC1 %.4f, AUROC2 %.4f, AUPRC2 %.4f, loss %.4f'%(mode, time_epoch, total_auroc1, total_auprc1, total_auroc2, total_auprc2, loss.item()))
    
    return_loss = np.mean(loss_lst)
    return total_auroc1, total_auprc1, total_auroc2, total_auprc2, return_loss, time_epoch


def link_pred_train_dual(model, args, g1, g2, df1, df2, node_feats, edge_feats1, edge_feats2):
    """
    Train the model for dual link prediction tasks.
    
    Args:
        model: The neural network model.
        args: Configuration arguments.
        g: Tuple of graphs (g1, g2).
        df: Tuple of DataFrames (df1, df2).
        node_feats: Node features tensor.
        edge_feats: Edge features tensor.
    
    Returns:
        best_auc_model: The best-performing model based on validation loss.
    """
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # data = g2
    # # Access the contents
    # indptr = data['indptr']
    # indices = data['indices']
    # ts = data['ts']
    # eid = data['eid']

    # # Print the data to verify
    # print("indptr:", indptr)
    # print("indices:", indices)
    # print("timestamps (ts):", ts)
    # print("edge IDs (eid):", eid)

    
    ###################################################
    # Get cached subgraphs
    if args.use_cached_subgraph:
        train_subgraphs1 = pre_compute_subgraphs(args, g1, df1, mode='train', input_data='edges1')
        train_subgraphs2 = pre_compute_subgraphs(args, g2, df2, mode='train', input_data="edges2")
    else:
        train_subgraphs1 = get_subgraph_sampler(args, g1, df1, mode='train')
        train_subgraphs2 = get_subgraph_sampler(args, g2, df2, mode='train')
    
    valid_subgraphs1 = pre_compute_subgraphs(args, g1, df1, mode='valid', input_data='edges1')
    valid_subgraphs2 = pre_compute_subgraphs(args, g2, df2, mode='valid', input_data="edges2")
    test_subgraphs1  = pre_compute_subgraphs(args, g1, df1, mode='test', input_data='edges1')
    test_subgraphs2  = pre_compute_subgraphs(args, g2, df2, mode='test', input_data="edges2")
    
    ###################################################
    # Initialize metrics
    all_results = {
        'train_ap1': [],
        'valid_ap1': [],
        'test_ap1' : [],
        'train_ap2': [],
        'valid_ap2': [],
        'test_ap2' : [],
        'train_auc1': [],
        'valid_auc1': [],
        'test_auc1' : [],
        'train_auc2': [],
        'valid_auc2': [],
        'test_auc2' : [],
        'train_loss': [],
        'valid_loss': [],
        'test_loss': [],
    }
    
    low_loss = 100000
    best_epoch = -1
    best_test_auc, best_test_ap = 0, 0
    
    if args.predict_class:
        num_classes = args.num_edgeType + 1
        # Metrics for dst1
        train_AUROC1 = MulticlassAUROC(num_classes, average="macro", thresholds=None)
        valid_AUROC1 = MulticlassAUROC(num_classes, average="macro", thresholds=None)
        test_AUROC1 = MulticlassAUROC(num_classes, average="macro", thresholds=None)
        
        train_AUPRC1 = MulticlassAveragePrecision(num_classes, average="macro", thresholds=None)
        valid_AUPRC1 = MulticlassAveragePrecision(num_classes, average="macro", thresholds=None)
        test_AUPRC1 = MulticlassAveragePrecision(num_classes, average="macro", thresholds=None)
        
        # Metrics for dst2
        train_AUROC2 = MulticlassAUROC(num_classes, average="macro", thresholds=None)
        valid_AUROC2 = MulticlassAUROC(num_classes, average="macro", thresholds=None)
        test_AUROC2 = MulticlassAUROC(num_classes, average="macro", thresholds=None)
        
        train_AUPRC2 = MulticlassAveragePrecision(num_classes, average="macro", thresholds=None)
        valid_AUPRC2 = MulticlassAveragePrecision(num_classes, average="macro", thresholds=None)
        test_AUPRC2 = MulticlassAveragePrecision(num_classes, average="macro", thresholds=None)
    else:
        # Binary metrics for dst1
        train_AUROC1 = BinaryAUROC(thresholds=None)
        valid_AUROC1 = BinaryAUROC(thresholds=None)
        test_AUROC1 = BinaryAUROC(thresholds=None)
        
        train_AUPRC1 = BinaryAveragePrecision(thresholds=None)
        valid_AUPRC1 = BinaryAveragePrecision(thresholds=None)
        test_AUPRC1 = BinaryAveragePrecision(thresholds=None)
        
        # Binary metrics for dst2
        train_AUROC2 = BinaryAUROC(thresholds=None)
        valid_AUROC2 = BinaryAUROC(thresholds=None)
        test_AUROC2 = BinaryAUROC(thresholds=None)
        
        train_AUPRC2 = BinaryAveragePrecision(thresholds=None)
        valid_AUPRC2 = BinaryAveragePrecision(thresholds=None)
        test_AUPRC2 = BinaryAveragePrecision(thresholds=None)
    
    ###################################################
    # Training loop
    for epoch in range(args.epochs):
        print(f'>>> Epoch {epoch + 1}')
        
        # Train
        train_auc1, train_ap1, train_auc2, train_ap2, train_loss, time_train = run_dual(
            model, optimizer, args, train_subgraphs1, train_subgraphs2, df1, df2, node_feats, edge_feats1, edge_feats2,
            train_AUROC1, train_AUPRC1, train_AUROC2, train_AUPRC2, mode='train'
        )
        
        # Validate
        with torch.no_grad():
            valid_auc1, valid_ap1, valid_auc2, valid_ap2, valid_loss, time_valid = run_dual(
                model, None, args, valid_subgraphs1, valid_subgraphs2, df1, df2, node_feats, edge_feats1, edge_feats2, 
                valid_AUROC1, valid_AUPRC1, valid_AUROC2, valid_AUPRC2, mode='valid'
            )
            
            # Test
            test_auc1, test_ap1, test_auc2, test_ap2, test_loss, time_test = run_dual(
                model, None, args, test_subgraphs1, test_subgraphs2, df1, df2, node_feats, edge_feats1, edge_feats2, 
                test_AUROC1, test_AUPRC1, test_AUROC2, test_AUPRC2, mode='test'
            )
        
        # Check for improvement
        if valid_loss < low_loss:
            best_auc_model = copy.deepcopy(model).cpu()
            low_loss = valid_loss
            best_epoch = epoch
            best_test_auc = ((test_auc1 + test_auc2) / 2)
            best_test_ap = ((test_ap1 + test_ap2) / 2)
        
        # Early stopping
        if epoch > best_epoch + 20:
            print("Early stopping triggered.")
            break
        
        # Aggregate metrics
        all_results['train_ap1'].append(train_ap1)
        all_results['valid_ap1'].append(valid_ap1)
        all_results['test_ap1'].append(test_ap1)
        
        all_results['train_ap2'].append(train_ap2)
        all_results['valid_ap2'].append(valid_ap2)
        all_results['test_ap2'].append(test_ap2)
        
        all_results['train_auc1'].append(train_auc1)
        all_results['valid_auc1'].append(valid_auc1)
        all_results['test_auc1'].append(test_auc1)
        
        all_results['train_auc2'].append(train_auc2)
        all_results['valid_auc2'].append(valid_auc2)
        all_results['test_auc2'].append(test_auc2)
        
        all_results['train_loss'].append(train_loss)
        all_results['valid_loss'].append(valid_loss)
        all_results['test_loss'].append(test_loss)
        
        print(f"Train: AUROC1 {train_auc1:.4f}, AUPRC1 {train_ap1:.4f}, AUROC2 {train_auc2:.4f}, AUPRC2 {train_ap2:.4f}, Loss {train_loss:.4f}")
        print(f"Valid: AUROC1 {valid_auc1:.4f}, AUPRC1 {valid_ap1:.4f}, AUROC2 {valid_auc2:.4f}, AUPRC2 {valid_ap2:.4f}, Loss {valid_loss:.4f}")
        print(f"Test: AUROC1 {test_auc1:.4f}, AUPRC1 {test_ap1:.4f}, AUROC2 {test_auc2:.4f}, AUPRC2 {test_ap2:.4f}, Loss {test_loss:.4f}")
    
    print(f'Best Test AUROC: {best_test_auc:.4f}, AUPRC: {best_test_ap:.4f}')
    return best_auc_model


def compute_sign_feats(node_feats, df, start_i, num_links, root_nodes, args):
    num_duplicate = len(root_nodes) // num_links 
    num_nodes = args.num_nodes

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
            src = torch.from_numpy(cur_df.src.values)
            dst = torch.from_numpy(cur_df.dst.values)
            edge_index = torch.stack([
                torch.cat([src, dst]), 
                torch.cat([dst, src])
            ])
            edge_index, edge_cnt = torch.unique(edge_index, dim=1, return_counts=True) 
            mask = edge_index[0]!=edge_index[1] # ignore self-loops
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