from __future__ import division, print_function
import time, argparse, numpy as np, copy, torch, os
import json
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils import data
from models import SkipGNN
from utils import (
    load_data_link_prediction_DDI, load_data_link_prediction_PPI,
    load_data_link_prediction_DTI, load_data_link_prediction_FILT_HB,
    load_data_link_prediction_GDI, Data_DDI, Data_PPI,
    Data_DTI, Data_FILT_HB, Data_GDI
)

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--fastmode', action='store_true', default=False)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--hidden1', type=int, default=64)
parser.add_argument('--hidden2', type=int, default=32)
parser.add_argument('--hidden_decode1', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--data_path', type=str, required=True, help="Path to data directory containing train.csv, val.csv, etc.")
parser.add_argument('--network_type', type=str, required=True)
parser.add_argument('--input_type', type=str, default=None) # Added default to avoid errors if not used
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

# --- Load Data Once (Structure is static, splits are file-based) ---
# We load the graph structure and dataframes once, but will re-create DataLoaders inside the loop
# to ensure shuffling respects the new seeds.

if args.network_type == 'DDI':
    adj, adj2, features, idx_map = load_data_link_prediction_DDI(args.data_path, args.input_type)
    Data_class = Data_DDI
elif args.network_type == 'PPI':
    adj, adj2, features, idx_map = load_data_link_prediction_PPI(args.data_path, args.input_type)
    Data_class = Data_PPI
elif args.network_type == 'DTI':
    adj, adj2, features, idx_map = load_data_link_prediction_DTI(args.data_path, args.input_type)
    Data_class = Data_DTI
elif args.network_type == 'GDI':
    adj, adj2, features, idx_map = load_data_link_prediction_GDI(args.data_path, args.input_type)
    Data_class = Data_GDI
# Grouping the FILT_HB based types
elif args.network_type in ['FILT_HB_DIST', 'FILT_HB_ANGLE', 'flavanone_DIST', 'flavanone_ANGLE', 'benzoin_DIST', 'benzoin_ANGLE']:
    adj, adj2, features, idx_map = load_data_link_prediction_FILT_HB(args.data_path, args.input_type)
    Data_class = Data_FILT_HB
else:
    raise ValueError(f"Unknown network_type: {args.network_type}")

# Load Split CSVs
df_train = pd.read_csv(f"{args.data_path}/train.csv")
df_val = pd.read_csv(f"{args.data_path}/val.csv")
df_test = pd.read_csv(f"{args.data_path}/test.csv")

# Ensure src and dst are integers
for df in [df_train, df_val, df_test]:
    df['src'] = df['src'].astype(int)
    df['dst'] = df['dst'].astype(int)

# Move static graph data to GPU if needed
if args.cuda:
    features = features.cuda()
    adj = adj.cuda()
    adj2 = adj2.cuda()

# --- Helper Function for Saving ---
def evaluate_and_save_logits(loader, model, output_path):
    model.eval()
    logits_list, labels_list = [], []
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with torch.no_grad():
        for labels, inp in loader:
            labels = labels.to(torch.float32)
            if args.cuda:
                labels = labels.cuda()
            out, _ = model(features, adj, adj2, (inp[0], inp[1]))
            logits = torch.squeeze(out)
            logits_list += logits.cpu().numpy().tolist()
            labels_list += labels.cpu().numpy().tolist()

    # Save as .pt file
    df_out = {
        "test_preds": torch.tensor(logits_list),
        "test_labels": torch.tensor(labels_list)
    }
    torch.save(df_out, output_path)
    # print(f"Saved predictions to {output_path}")

# --- Main Run Function ---
def run_seed(seed):
    print(f"\n--- Starting Run for Seed {seed} ---")
    
    # 1. Set Seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    
    # 2. Re-create DataLoaders (to reset shuffle with new seed)
    params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 0, 'drop_last': False}
    train_loader = data.DataLoader(Data_class(idx_map, df_train.label.values, df_train), **params)
    val_loader = data.DataLoader(Data_class(idx_map, df_val.label.values, df_val), **params)
    test_loader = data.DataLoader(Data_class(idx_map, df_test.label.values, df_test), **params)

    # 3. Initialize Model
    model = SkipGNN(
        nfeat=features.shape[1],
        nhid1=args.hidden1,
        nhid2=args.hidden2,
        nhid_decode1=args.hidden_decode1,
        dropout=args.dropout
    )

    if args.cuda:
        model.cuda()

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fct = torch.nn.BCEWithLogitsLoss()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 4. Training Loop
    start_time = time.time()
    
    best_model_state = copy.deepcopy(model.state_dict())
    lowest_val_loss = float('inf')
    patience = 30
    counter = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for labels, inp in train_loader:
            labels = labels.to(torch.float32)
            if args.cuda:
                labels = labels.cuda()
            optimizer.zero_grad()
            out, _ = model(features, adj, adj2, (inp[0], inp[1]))
            logits = torch.squeeze(out)
            loss = loss_fct(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Evaluate on val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for labels, inp in val_loader:
                labels = labels.to(torch.float32)
                if args.cuda:
                    labels = labels.cuda()
                out, _ = model(features, adj, adj2, (inp[0], inp[1]))
                logits = torch.squeeze(out)
                batch_loss = loss_fct(logits, labels)
                val_loss += batch_loss.item()

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1

        # Optional: Print less frequently to reduce clutter
        if epoch % 5 == 0:
            print(f"Seed {seed} | Epoch {epoch + 1:03d} | Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

        if counter >= patience:
            print(f"Seed {seed} | Early stopping at epoch {epoch + 1}")
            break

    end_time = time.time()
    train_val_time_sec = end_time - start_time

    # 5. Testing & Saving
    # Load best model
    model.load_state_dict(best_model_state)

    output_dir = f"temp_result"
    os.makedirs(output_dir, exist_ok=True)

    # Path for predictions
    output_pt_path = os.path.join(output_dir, f"outputs_{args.network_type}_seed{seed}.pt")

    test_start_time = time.time()
    evaluate_and_save_logits(test_loader, model, output_pt_path)
    test_end_time = time.time()
    test_time_sec = test_end_time - test_start_time

    # 6. Save Summary JSON
    summary_info = {
        "dataset": args.network_type,
        "model": f"SkipGNN_seed{seed}",
        "train_validation_time_seconds": round(train_val_time_sec, 4),
        "test_inference_time_seconds": round(test_time_sec, 4),
        "Trainable params": num_params,
        "Trainable params (MB)": round(num_params * 4 / 1024 / 1024, 6)
    }

    summary_json_path = os.path.join(output_dir, f"summary_{args.network_type}_seed{seed}.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_info, f, indent=4)

    print(f"Seed {seed} Completed. Files saved to: {output_dir}")

# --- Execute Loops ---
if __name__ == "__main__":
    SEEDS = [0, 1, 2, 3, 4]
    
    print(f"Running training for network: {args.network_type} on seeds: {SEEDS}")
    print(f"Data Path: {args.data_path}")
    
    for seed in SEEDS:
        run_seed(seed)
        
    print("\nAll seeds finished.")