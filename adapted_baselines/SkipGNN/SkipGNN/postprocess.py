import torch
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryF1Score
import json
import os
import argparse
import numpy as np
import time

parser = argparse.ArgumentParser()

parser.add_argument('--file1_pattern', type=str, required=True, help='Path pattern to prediction file 1 (e.g., path/to/preds_seed{}.pt)')
parser.add_argument('--file2_pattern', type=str, required=True, help='Path pattern to prediction file 2 (e.g., path/to/preds_seed{}.pt)')
parser.add_argument('--summary1_pattern', type=str, required=True, help='Path pattern to summary file 1 (e.g., path/to/summary_seed{}.json)')
parser.add_argument('--summary2_pattern', type=str, required=True, help='Path pattern to summary file 2 (e.g., path/to/summary_seed{}.json)')

# Allow user to specify which seeds to run (defaults to 0-4)
parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4], help='List of seeds to run')
args_cli = parser.parse_args()

def modified_link_pred_metrics(predicts1, predicts2, labels1, labels2):
    device = torch.device("cpu")

    min_len = min(predicts1.size(0), predicts2.size(0))

    predicts1, predicts2, labels1, labels2  = predicts1[:min_len], predicts2[:min_len], labels1[:min_len], labels2[:min_len]
    predicts1, predicts2 = predicts1.detach().cpu(), predicts2.detach().cpu()

    joint_label = (labels2 == 1).long()
    final_score = (predicts1 + predicts2) / 2.0
    final_preds = (final_score > 0.5).long()

    metric_auc = BinaryAUROC(thresholds=None).to(device)
    metric_ap = BinaryAveragePrecision(thresholds=None).to(device)
    metric_f1 = BinaryF1Score().to(device)

    metric_auc.update(final_score, joint_label)
    metric_ap.update(final_score, joint_label)
    metric_f1.update(final_preds, joint_label)

    return {
        "AUROC": metric_auc.compute().item(),
        "AUPRC": metric_ap.compute().item(),
        "F1": metric_f1.compute().item(),
    }

if __name__ == "__main__":
    
    all_metrics = []
    
    all_train_times = []
    all_inference_times = []
    all_params = []
    all_params_mb = []

    individual_results = {}
    
    start_global = time.time()

    # Loop through the seeds defined in arguments
    for seed in args_cli.seeds:
        f1_path = args_cli.file1_pattern.format(seed)
        f2_path = args_cli.file2_pattern.format(seed)
        s1_path = args_cli.summary1_pattern.format(seed)
        s2_path = args_cli.summary2_pattern.format(seed)

        print(f"Processing Seed {seed}...")
        
        # Check files existence
        if not os.path.exists(f1_path) or not os.path.exists(f2_path):
            print(f"Warning: Prediction files for seed {seed} not found. Skipping.")
            continue
        if not os.path.exists(s1_path) or not os.path.exists(s2_path):
            print(f"Warning: Summary files for seed {seed} not found. Skipping.")
            continue

        data1 = torch.load(f1_path)
        data2 = torch.load(f2_path)

        with open(s1_path, 'r') as f: summary1 = json.load(f)
        with open(s2_path, 'r') as f: summary2 = json.load(f)

        metrics = modified_link_pred_metrics(
            data1["test_preds"], data2["test_preds"],
            data1["test_labels"], data2["test_labels"]
        )
        
        sum_train_time = summary1.get("train_validation_time_seconds", 0) + summary2.get("train_validation_time_seconds", 0)
        sum_inf_time = summary1.get("test_inference_time_seconds", 0) + summary2.get("test_inference_time_seconds", 0)
        sum_params = summary1.get("Trainable params", 0) + summary2.get("Trainable params", 0)
        sum_params_mb = summary1.get("Trainable params (MB)", 0) + summary2.get("Trainable params (MB)", 0)

        all_metrics.append(metrics)
        all_train_times.append(sum_train_time)
        all_inference_times.append(sum_inf_time)
        all_params.append(sum_params)
        all_params_mb.append(sum_params_mb)

        individual_results[f"seed_{seed}"] = {
            "metrics": {k: f"{v:.4f}" for k, v in metrics.items()},
            "computational_sums": {
                "train_validation_time_seconds": f"{sum_train_time:.4f}",
                "test_inference_time_seconds": f"{sum_inf_time:.4f}",
                "Trainable params": sum_params,
                "Trainable params (MB)": f"{sum_params_mb:.4f}"
            }
        }

    end_global = time.time()
    post_processing_time = end_global - start_global
    print(f"Total post_processing_time: {post_processing_time:.4f} seconds")

    if not all_metrics:
        print("No metrics calculated. Check file paths.")
        exit()

    aggregated_results = {}

  
    for key in ["AUROC", "AUPRC", "F1"]:
        values = [m[key] for m in all_metrics]
        aggregated_results[f"{key}_mean"] = f"{np.mean(values):.4f}"
        aggregated_results[f"{key}_std"] = f"{np.std(values):.4f}"

   
    comp_map = {
        "train_validation_time_seconds": all_train_times,
        "test_inference_time_seconds": all_inference_times,
        "Trainable_params": all_params,
        "Trainable_params_MB": all_params_mb
    }

    for key, values in comp_map.items():
        aggregated_results[f"{key}_mean"] = f"{np.mean(values):.4f}"
        aggregated_results[f"{key}_std"] = f"{np.std(values, ddof=1):.4f}"


    result_json = {
        "source_pattern1": args_cli.file1_pattern,
        "source_pattern2": args_cli.file2_pattern,
        "summary_pattern1": args_cli.summary1_pattern,
        "summary_pattern2": args_cli.summary2_pattern,
        "aggregated_metrics": aggregated_results,
        "post_processing_time": round(post_processing_time, 4),
        "individual_runs": individual_results
    }

    final_json_str = json.dumps(result_json, indent=4)


    base1 = os.path.basename(args_cli.file1_pattern).replace('_{}.pt', '').replace('{}.pt', '').replace('{}', '')
    base2 = os.path.basename(args_cli.file2_pattern).replace('_{}.pt', '').replace('{}.pt', '').replace('{}', '')
    

    base1 = os.path.splitext(base1)[0]
    base2 = os.path.splitext(base2)[0]

    save_result_folder = f"./results"
    os.makedirs(save_result_folder, exist_ok=True)
    save_result_path = os.path.join(save_result_folder, f"{base1}_{base2}.json")

    with open(save_result_path, 'w') as file:
        file.write(final_json_str)

    print(f"Aggregated results saved to {save_result_path}")
