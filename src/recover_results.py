import os
import re
import pandas as pd
import torch
import torch_geometric
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt


# Add project root to path to allow imports from 'src'
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Imports from project
from src.config import *
from src.dataset import BRD4Dataset, building_block_split
from src.model import GATModel
from src.utils import set_seed, plot_loss_curve, plot_val_ap_curve, plot_confusion_matrix, plot_roc_curve, plot_pr_curve
from src.train import evaluate

# 1. LOG DATA (Paste your log here)
LOG_DATA = """
1894.6s	268	Epoch: 001, Loss: 0.1216, Val AP: 0.3959, Val AUC: 0.7005
3428.6s	269	Epoch: 002, Loss: 0.1163, Val AP: 0.4245, Val AUC: 0.7229
4956.4s	270	Epoch: 003, Loss: 0.1131, Val AP: 0.4635, Val AUC: 0.7478
6490.6s	271	Epoch: 004, Loss: 0.1110, Val AP: 0.4753, Val AUC: 0.7561
8023.7s	272	Epoch: 005, Loss: 0.1100, Val AP: 0.4852, Val AUC: 0.7632
9553.0s	273	Epoch: 006, Loss: 0.1084, Val AP: 0.4969, Val AUC: 0.7724
11080.3s	274	Epoch: 007, Loss: 0.1061, Val AP: 0.5028, Val AUC: 0.7792
12604.3s	275	Epoch: 008, Loss: 0.1037, Val AP: 0.5112, Val AUC: 0.7840
14133.1s	276	Epoch: 009, Loss: 0.1021, Val AP: 0.5140, Val AUC: 0.7846
15659.4s	277	Epoch: 010, Loss: 0.1011, Val AP: 0.5227, Val AUC: 0.7887
17183.3s	278	Epoch: 011, Loss: 0.1001, Val AP: 0.5269, Val AUC: 0.7911
18709.6s	279	Epoch: 012, Loss: 0.0993, Val AP: 0.5317, Val AUC: 0.7951
20228.0s	280	Epoch: 013, Loss: 0.0984, Val AP: 0.5407, Val AUC: 0.8004
21741.7s	281	Epoch: 014, Loss: 0.0972, Val AP: 0.5580, Val AUC: 0.8108
23255.7s	282	Epoch: 015, Loss: 0.0955, Val AP: 0.5783, Val AUC: 0.8224
24769.3s	283	Epoch: 016, Loss: 0.0943, Val AP: 0.5970, Val AUC: 0.8294
26295.5s	284	Epoch: 017, Loss: 0.0935, Val AP: 0.6079, Val AUC: 0.8338
27825.2s	285	Epoch: 018, Loss: 0.0929, Val AP: 0.6187, Val AUC: 0.8374
29362.1s	286	Epoch: 019, Loss: 0.0923, Val AP: 0.6265, Val AUC: 0.8410
30881.5s	287	Epoch: 020, Loss: 0.0918, Val AP: 0.6370, Val AUC: 0.8436
32409.0s	288	Epoch: 021, Loss: 0.0912, Val AP: 0.6462, Val AUC: 0.8462
33947.5s	289	Epoch: 022, Loss: 0.0906, Val AP: 0.6555, Val AUC: 0.8492
35477.0s	290	Epoch: 023, Loss: 0.0900, Val AP: 0.6682, Val AUC: 0.8536
36993.5s	291	Epoch: 024, Loss: 0.0891, Val AP: 0.6788, Val AUC: 0.8565
38516.5s	292	Epoch: 025, Loss: 0.0880, Val AP: 0.6908, Val AUC: 0.8606
40035.8s	293	Epoch: 026, Loss: 0.0869, Val AP: 0.7001, Val AUC: 0.8641
41557.0s	294	Epoch: 027, Loss: 0.0860, Val AP: 0.7090, Val AUC: 0.8669
43077.5s	295	Epoch: 028, Loss: 0.0852, Val AP: 0.7176, Val AUC: 0.8695
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
        target_csv = "/kaggle/input/gat-inz/leash_brd4_filtered.csv" #os.path.join(DATA_PROCESSED_DIR, 'leash_brd4_filtered.csv')
            
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
                         num_layers=GAT_LAYERS, dropout=DROPOUT).to(device)
                         
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
