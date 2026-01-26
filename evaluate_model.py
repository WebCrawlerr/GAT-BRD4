import argparse
import os
import torch
from torch_geometric.loader import DataLoader
from src.config import *
from src.dataset import BRD4Dataset, building_block_split
from src.model import GATModel
from src.train import evaluate
from src.utils import plot_confusion_matrix, plot_roc_curve, plot_pr_curve, set_seed

def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved GAT model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the .pth model file')
    parser.add_argument('--data_dir', type=str, default=DATA_PROCESSED_DIR, help='Directory containing the dataset')
    parser.add_argument('--filtered_file', type=str, default=None, help='Specific CSV file to load if data.pt is missing')
    parser.add_argument('--output_dir', type=str, default=os.path.join(PLOTS_DIR, 'evaluation'), help='Directory to save plots')
    parser.add_argument('--limit', type=int, default=None, help='Limit dataset size for quick testing')
    parser.add_argument('--raw_file', type=str, default=r'data/raw/leash-BELKA/train.csv', help='Path to raw CSV')
    parser.add_argument('--target', type=str, default='BRD4', help='Target protein name')
    parser.add_argument('--heads', type=int, default=GAT_HEADS, help='Number of attention heads')
    parser.add_argument('--hidden_dim', type=int, default=GAT_HIDDEN_DIM, help='Hidden dimension size')
    parser.add_argument('--log_path', type=str, default=None, help='Path to training_log.csv to plot history')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Optional: Plot History first if log provided
    if args.log_path and os.path.exists(args.log_path):
        import pandas as pd
        from src.utils import plot_loss_curve, plot_val_ap_curve
        
        print(f"Found log file at {args.log_path}. Generating history plots...")
        try:
            df_log = pd.read_csv(args.log_path)
            if 'Train_Loss' in df_log.columns:
                plot_loss_curve(df_log['Train_Loss'].tolist(), os.path.join(args.output_dir, 'loss_curve_recovered.png'))
            if 'Val_AP' in df_log.columns:
                plot_val_ap_curve(df_log['Val_AP'].tolist(), os.path.join(args.output_dir, 'val_ap_curve_recovered.png'))
            # Also plot Val Loss if available (new feature)
            if 'Val_Loss' in df_log.columns:
                # We can reuse plot_loss_curve but title might be wrong, or just plot it manually here or add a specific function.
                # For now let's just use plot_loss_curve logic but custom
                import matplotlib.pyplot as plt
                plt.figure(figsize=(6, 5))
                plt.plot(df_log['Val_Loss'], label='Val Loss', color='orange')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Validation Loss')
                plt.legend()
                plt.savefig(os.path.join(args.output_dir, 'val_loss_curve_recovered.png'))
                plt.close()
                
            print("History plots generated.")
        except Exception as e:
            print(f"Error generating history plots: {e}")
            
    # 1. Load Dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset = BRD4Dataset(
        root=args.data_dir, 
        raw_file=args.raw_file,
        filtered_file=args.filtered_file, 
        limit=args.limit,
        target_name=args.target
    )
    
    # 2. Split (use consistent split)
    # We evaluate on the TEST set (or Validation if Test is empty/0%)
    _, val_dataset, test_dataset = building_block_split(dataset, seed=SEED)
    
    usage_set_name = "Test"
    if len(test_dataset) == 0:
        print("Warning: Test dataset is empty (split=0.0). Using Validation set for evaluation.")
        test_dataset = val_dataset
        usage_set_name = "Validation"
        
    if len(test_dataset) == 0:
        print("Error: Validation dataset is also empty.")
        return

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"{usage_set_name} Set Size: {len(test_dataset)}")
    
    # 3. Load Model
    print(f"Loading model from {args.model_path}...")
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
        
    num_node_features = dataset[0].x.shape[1]
    num_edge_features = dataset[0].edge_attr.shape[1]
    
    model = GATModel(num_node_features, num_edge_features, 
                     hidden_dim=args.hidden_dim, heads=args.heads, 
                     num_layers=GAT_LAYERS, dropout=DROPOUT).to(device)
                     
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return

    # 4. Evaluate
    print("Running evaluation...")
    metrics, _, y_true, y_pred_logits = evaluate(model, test_loader, device)
    print(f"Metrics: {metrics}")
    
    # 5. Plot
    print(f"Generating plots in {args.output_dir}...")
    plot_confusion_matrix(y_true, y_pred_logits, os.path.join(args.output_dir, 'confusion_matrix.png'))
    plot_roc_curve(y_true, y_pred_logits, os.path.join(args.output_dir, 'roc_curve.png'))
    plot_pr_curve(y_true, y_pred_logits, os.path.join(args.output_dir, 'pr_curve.png'))
    
    print("Done.")

if __name__ == "__main__":
    main()
