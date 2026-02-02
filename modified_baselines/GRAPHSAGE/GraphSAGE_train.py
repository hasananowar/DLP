import os
import json
import time
import random
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torchmetrics.functional.classification import binary_f1_score, binary_auroc, binary_average_precision

# --- CLI Arguments ---
parser = argparse.ArgumentParser(description="GraphSAGE Training Script")
parser.add_argument("--dataset_name", type=str, required=True, help="Path to input CSV file (e.g. flavanone/edges1.csv)")
args = parser.parse_args()

# --- Configuration ---
SEEDS = [0, 1, 2, 3, 4]
DATASET_PATH = f"DATA/{args.dataset_name}"

# Construct a clean identifier for filenames
# e.g., "flavanone/edges1.csv" -> "flavanone_edges1"
path_parts = os.path.normpath(args.dataset_name).split(os.sep)
if len(path_parts) > 1:
    dataset_identifier = f"{path_parts[-2]}_{os.path.splitext(path_parts[-1])[0]}"
else:
    dataset_identifier = os.path.splitext(path_parts[-1])[0]

# Output Directory
output_dir = f"temp_result"
os.makedirs(output_dir, exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Data (We load once, but process/split inside the loop)
print(f"Loading data from {DATASET_PATH}...")
raw_df = pd.read_csv(DATASET_PATH)

# --- Helper Functions ---
def generate_extra_negatives(count, used_edges, num_nodes):
    extra_src, extra_dst = [], []
    while len(extra_src) < count:
        u, v = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
        if u != v and (u, v) not in used_edges and (v, u) not in used_edges:
            extra_src.append(u)
            extra_dst.append(v)
            used_edges.add((u, v))
    return pd.DataFrame({'src': extra_src, 'dst': extra_dst, 'label': 0})

def make_edges_labels(df_in):
    edges = df_in[['src', 'dst']].values
    labels = df_in['label'].values
    undirected_edges = np.concatenate([edges, edges[:, [1, 0]]], axis=0)
    undirected_labels = np.concatenate([labels, labels], axis=0)
    ei = torch.tensor(undirected_edges.T, dtype=torch.long, device=device)
    lbl = torch.tensor(undirected_labels, dtype=torch.float32, device=device)
    return ei, lbl

# --- Models ---
class GraphSAGE(nn.Module):
    def __init__(self, in_ch, hid_ch=100, dropout=0.1):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, hid_ch)
        self.conv2 = SAGEConv(hid_ch, hid_ch)
        self.dropout = dropout

    def forward(self, x, ei):
        h = F.relu(self.conv1(x, ei))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, ei)
        return h

class LinkPredictor(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.lin = nn.Linear(in_ch, 1)

    def forward(self, h_i, h_j):
        return self.lin(h_i * h_j).view(-1)

@torch.no_grad()
def evaluate(model, predictor, x, ei, lbl, criterion):
    model.eval(); predictor.eval()
    h = model(x, ei)
    logits = predictor(h[ei[0]], h[ei[1]])
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()
    li = lbl.long()
    
    if len(li) == 0: return 0.0, 0.0, 0.0, 0.0
    
    return (
        binary_f1_score(preds, li).item(),
        binary_auroc(probs, li).item(),
        binary_average_precision(probs, li).item(),
        criterion(logits, lbl).item()
    )

# --- Main Execution Loop ---
def run_seed(seed):
    print(f"\n--- Starting Run for Seed {seed} ---")
    
    # 1. Set Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 2. Data Processing (Seed dependent)
    df = raw_df.copy()
    df = df.sort_values('time').drop_duplicates(subset=['src', 'dst'], keep='last').reset_index(drop=True)

    # Encode node IDs
    all_nodes = set(df['src']).union(set(df['dst']))
    node2id = {node: i for i, node in enumerate(all_nodes)}
    df['src'] = df['src'].map(node2id)
    df['dst'] = df['dst'].map(node2id)
    num_nodes = len(node2id)

    # Split positives and labeled negatives
    pos_df = df[df['label'] == 1].copy()
    neg_df = df[df['label'] == 0].copy()

    # Generate extra negative edges
    used_edges = set((u, v) for u, v in zip(df['src'], df['dst'])) | set((v, u) for u, v in zip(df['src'], df['dst']))
    extra_neg_df = generate_extra_negatives(len(pos_df), used_edges, num_nodes)

    # Combine and upsample positives
    df_combined = pd.concat([pos_df, neg_df, extra_neg_df], ignore_index=True)
    pos_df_final = df_combined[df_combined['label'] == 1].copy()
    neg_df_final = df_combined[df_combined['label'] == 0].copy()
    
    # Use 'seed' for pandas sampling to ensure reproducibility per run
    pos_df_upsampled = pos_df_final.sample(n=len(neg_df_final), replace=True, random_state=seed)
    df_balanced = pd.concat([pos_df_upsampled, neg_df_final], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    # Train/val/test split
    n = len(df_balanced)
    train_df = df_balanced.iloc[:int(0.7 * n)].reset_index(drop=True)
    val_df   = df_balanced.iloc[int(0.7 * n):int(0.85 * n)].reset_index(drop=True)
    test_df  = df_balanced.iloc[int(0.85 * n):].reset_index(drop=True)

    print(f"Train size: {len(train_df)} | Val size: {len(val_df)} | Test size: {len(test_df)}")

    # Identity features
    x = torch.eye(num_nodes, dtype=torch.float32).to(device)
    
    train_ei, train_lbl = make_edges_labels(train_df)
    val_ei, val_lbl = make_edges_labels(val_df)
    test_ei, test_lbl = make_edges_labels(test_df)

    # Initialize Model
    model = GraphSAGE(x.size(1)).to(device)
    predictor = LinkPredictor(100).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=5e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    num_params = sum(p.numel() for p in list(model.parameters()) + list(predictor.parameters()) if p.requires_grad)

    # Training Loop
    best_loss = float('inf')
    best_epoch = 0
    epochs = 300
    
    start_time = time.time() # Start timer

    for epoch in range(epochs):
        model.train(); predictor.train()
        optimizer.zero_grad()
        h = model(x, train_ei)
        logits = predictor(h[train_ei[0]], h[train_ei[1]])
        loss = criterion(logits, train_lbl)
        loss.backward()
        optimizer.step()

        val_f1, val_auroc, val_ap, val_loss = evaluate(model, predictor, x, val_ei, val_lbl, criterion)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch

        if epoch - best_epoch > 30:
            break

    train_valid_end = time.time()
    train_valid_seconds = train_valid_end - start_time

    # --- Save Test Predictions ---
    test_start = time.time()
    model.eval(); predictor.eval()
    with torch.no_grad():
        h = model(x, test_ei)
        logits = predictor(h[test_ei[0]], h[test_ei[1]])
        
        # Save output file
        # Format: outputs_{identifier}_seed{seed}.pt
        save_path = os.path.join(output_dir, f"outputs_{dataset_identifier}_seed{seed}.pt")
        torch.save(
            {'test_preds': logits.cpu(), 'test_labels': test_lbl.cpu()},
            save_path
        )

    test_end = time.time()
    test_time_sec = test_end - test_start

    # --- Save Summary ---
    summary = {
        "dataset": dataset_identifier,
        "model": f"GraphSAGE_seed{seed}",
        "train_validation_time_seconds": round(train_valid_seconds, 4),
        "test_inference_time_seconds": round(test_time_sec, 4),
        "Trainable params": num_params,
        "Trainable params (MB)": round(num_params * 4 / 1024 / 1024, 6)
    }

    # Save summary file
    summary_path = os.path.join(output_dir, f"summary_{dataset_identifier}_seed{seed}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Seed {seed} complete. Results saved to {output_dir}")

if __name__ == "__main__":
    for seed in SEEDS:
        run_seed(seed)
    print("\nAll seeds completed successfully.")