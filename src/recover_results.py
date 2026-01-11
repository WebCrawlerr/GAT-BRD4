import os
import re
import pandas as pd
import torch
import torch_geometric
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

# Imports from project
from src.config import *
from src.dataset import BRD4Dataset, building_block_split
from src.model import GATModel
from src.utils import set_seed, plot_loss_curve, plot_val_ap_curve, plot_confusion_matrix, plot_roc_curve, plot_pr_curve
from src.train import evaluate

# 1. LOG DATA (Paste your log here)
LOG_DATA = """
626.0s	    264	Class Distribution in Train: Pos=300687, Neg=896602
3505.6s 	265	Epoch: 001, Loss: 0.5603, Val AP: 0.8744, Val AUC: 0.9255
6381.3s	    266	Epoch: 002, Loss: 0.4079, Val AP: 0.8995, Val AUC: 0.9434
9261.6s 	267	Epoch: 003, Loss: 0.3881, Val AP: 0.8850, Val AUC: 0.9332
12137.1s	268	Epoch: 004, Loss: 0.3788, Val AP: 0.8811, Val AUC: 0.9309
15188.1s	269	Epoch: 005, Loss: 0.3722, Val AP: 0.9120, Val AUC: 0.9531
18050.4s	270	Epoch: 006, Loss: 0.3683, Val AP: 0.8896, Val AUC: 0.9366
20907.2s	271	Epoch: 007, Loss: 0.3651, Val AP: 0.8786, Val AUC: 0.9286
23783.1s	272	Epoch: 008, Loss: 0.3625, Val AP: 0.8751, Val AUC: 0.9252
26661.7s	273	Epoch: 009, Loss: 0.3610, Val AP: 0.8997, Val AUC: 0.9436
29578.9s	274	Epoch: 010, Loss: 0.3599, Val AP: 0.8993, Val AUC: 0.9442
32457.3s	275	Epoch: 011, Loss: 0.3589, Val AP: 0.8877, Val AUC: 0.9352
35343.5s	276	Epoch: 012, Loss: 0.3582, Val AP: 0.8954, Val AUC: 0.9425
38214.3s	277	Epoch: 013, Loss: 0.3557, Val AP: 0.8807, Val AUC: 0.9286
41065.4s	278	Epoch: 014, Loss: 0.3551, Val AP: 0.8808, Val AUC: 0.9269
"""

def parse_logs(log_text):
    data = []
    # Regex to capture metrics
    pattern = r"Epoch: (\d+), Loss: ([\d\.]+), Val AP: ([\d\.]+), Val AUC: ([\d\.]+)"
    
    for line in log_text.strip().split('\n'):
        match = re.search(pattern, line)
        if match:
            epoch, loss, val_ap, val_auc = match.groups()
            data.append({
                'Epoch': int(epoch),
                'Train_Loss': float(loss),
                'Val_AP': float(val_ap),
                'Val_AUC': float(val_auc)
            })
    return pd.DataFrame(data)

def main():
    print("Recovering results from logs...")
    
    # Create output dir
    output_dir = os.path.join(PLOTS_DIR, 'recovered_run')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Process Logs & History Plots
    df = parse_logs(LOG_DATA)
    if not df.empty:
        csv_path = os.path.join(output_dir, 'training_log.csv')
        df.to_csv(csv_path, index=False)
        print(f"Log saved to {csv_path}")
        
        plot_loss_curve(df['Train_Loss'].tolist(), os.path.join(output_dir, 'loss_curve.png'))
        plot_val_ap_curve(df['Val_AP'].tolist(), os.path.join(output_dir, 'val_ap_curve.png'))
        print("History plots generated.")
    else:
        print("No valid log lines found!")

    # 2. Evaluate Best Model (if exists)
    model_path = "/kaggle/input/v1-0/pytorch/default/1/best_model.pth" 
    if os.path.exists(model_path):
        print(f"Found saved model: {model_path}. Proceeding to evaluation...")
        
        set_seed(SEED)
        
        # Load Data (must match training setup)
        # Assuming args: processed_dir=DATA_PROCESSED_DIR, filtered_file=None (or default), limit=None
        # Note: If train used 'limit', we need to know. 
        # The logs show "Pos=300687, Neg=896602" (~1.2M total). 
        # Full dataset is usually larger. 
        # If user ran: --limit 1500000 --cv 0 (from the first message in conversation history)
        # Then we should use: limit=1500000
        
        # Checking user request from Step 0: "--limit 1500000"
        LIMIT = 1500000
        
        print(f"Loading dataset (limit={LIMIT})...")
        target_csv = /kaggle/input/gat-inz/leash_brd4_filtered.csv #os.path.join(DATA_PROCESSED_DIR, 'leash_brd4_filtered.csv')
            
        dataset = BRD4Dataset(root=DATA_PROCESSED_DIR, filtered_file=target_csv, limit=LIMIT)
        
        print("Splitting data...")
        train_dataset, val_dataset, test_dataset = building_block_split(dataset)
        
        # Eval Loader (Test if available, else Val)
        eval_dataset = test_dataset if test_dataset else val_dataset
        eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Model
        num_node_features = dataset[0].x.shape[1]
        num_edge_features = dataset[0].edge_attr.shape[1]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = GATModel(num_node_features, num_edge_features, 
                         hidden_dim=GAT_HIDDEN_DIM, heads=GAT_HEADS, 
                         layers=GAT_LAYERS, dropout=DROPOUT).to(device)
                         
        print("Loading weights...")
        model.load_state_dict(torch.load(model_path))
        
        print("Running evaluation...")
        metrics, y_true, y_pred = evaluate(model, eval_loader, device)
        print(f"Evaluation Metrics: {metrics}")
        
        # Plots
        plot_confusion_matrix(y_true, y_pred, os.path.join(output_dir, 'confusion_matrix.png'))
        plot_roc_curve(y_true, y_pred, os.path.join(output_dir, 'roc_curve.png'))
        plot_pr_curve(y_true, y_pred, os.path.join(output_dir, 'pr_curve.png'))
        print(f"Evaluation plots saved to {output_dir}")
        
    else:
        print("No 'best_model.pth' found. Skipping evaluation.")

if __name__ == "__main__":
    main()
