import os
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.config import *
from src.dataset import BRD4Dataset, scaffold_split
from src.train import run_training
from src.model import GCNModel, GATModel

def run_learning_curve(dataset, fractions=[0.2, 0.4, 0.6, 0.8, 1.0]):
    print("\n=== Running Learning Curve Experiment ===")
    
    # Use a fixed split for consistency
    train_dataset_full, val_dataset, test_dataset = scaffold_split(dataset, seed=42)
    
    results = {'Fraction': [], 'Train_Size': [], 'Val_AP': [], 'Val_AUC': [], 'Val_F0.5': []}
    
    # Shuffle train dataset once to ensure random subsets are consistent if we took first N
    # But scaffold split already shuffles scaffolds.
    # We will take the first N samples from the shuffled train set.
    
    total_train_size = len(train_dataset_full)
    
    for frac in fractions:
        subset_size = int(total_train_size * frac)
        print(f"\nTraining with {frac*100}% data ({subset_size} samples)...")
        
        # Create subset
        train_subset = train_dataset_full[:subset_size]
        
        # Run training (suppress plotting for speed)
        metrics = run_training(train_subset, val_dataset, test_dataset=None, plot=False, verbose=False)
        
        results['Fraction'].append(frac)
        results['Train_Size'].append(subset_size)
        results['Val_AP'].append(metrics['AP'])
        results['Val_AUC'].append(metrics['AUC'])
        results['Val_F0.5'].append(metrics['F0.5'])
        
        print(f"  -> AP: {metrics['AP']:.4f}, AUC: {metrics['AUC']:.4f}, F0.5: {metrics['F0.5']:.4f}")
        
    # Save results
    df_results = pd.DataFrame(results)
    save_path = os.path.join(PLOTS_DIR, 'learning_curve_results.csv')
    df_results.to_csv(save_path, index=False)
    print(f"\nResults saved to {save_path}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['Fraction'], df_results['Val_AP'], marker='o', label='AP')
    plt.plot(df_results['Fraction'], df_results['Val_AUC'], marker='s', label='AUC')
    plt.plot(df_results['Fraction'], df_results['Val_F0.5'], marker='^', label='F0.5')
    plt.xlabel('Fraction of Training Data')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, 'learning_curve.png'))
    print("Plot saved to plots/learning_curve.png")

def run_ablation_study(dataset):
    print("\n=== Running Ablation Study (GAT vs GCN) ===")
    
    train_dataset, val_dataset, test_dataset = scaffold_split(dataset, seed=42)
    
    # 1. GAT (Baseline) - uses config parameters
    print("\nTraining GAT (Baseline)...")
    metrics_gat = run_training(train_dataset, val_dataset, test_dataset=None, plot=False, verbose=False)
    print(f"GAT Results: AP={metrics_gat['AP']:.4f}, AUC={metrics_gat['AUC']:.4f}")
    
    # 2. GCN
    print("\nTraining GCN...")
    # We need to manually initialize GCN and pass it to run_training if possible, 
    # OR modify run_training to accept a model class/instance.
    # Currently run_training initializes GATModel internally.
    # We will instantiate GCN here and pass it if we modify run_training, 
    # BUT run_training doesn't accept a model instance argument yet.
    # Let's check run_training signature.
    
    # Hack: We will temporarily patch the model initialization in run_training OR 
    # better, we instantiate the model here and pass it.
    # Let's check src/train.py content.
    # It instantiates: model = GATModel(...).to(device)
    
    # To avoid changing train.py too much, we can just copy the training loop here for GCN 
    # or make run_training accept a 'model_cls' argument.
    # Making run_training flexible is best.
    
    # For now, let's assume we update run_training to accept model_cls.
    # If not, I'll write a small training loop here for GCN.
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNModel(num_node_features=dataset.num_features, 
                     num_edge_features=dataset.num_edge_features,
                     hidden_dim=GAT_HIDDEN_DIM, # Use same dim
                     num_layers=GAT_LAYERS,
                     dropout=DROPOUT).to(device)
                     
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Simple training loop for GCN
    from src.train import train_epoch, evaluate
    
    best_val_ap = 0
    patience_counter = 0
    
    train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch_geometric.loader.DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        
        if val_metrics['AP'] > best_val_ap:
            best_val_ap = val_metrics['AP']
            patience_counter = 0
            best_metrics_gcn = val_metrics
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            break
            
    print(f"GCN Results: AP={best_metrics_gcn['AP']:.4f}, AUC={best_metrics_gcn['AUC']:.4f}")
    
    # Compare
    print("\n--- Comparison ---")
    print(f"GAT AP: {metrics_gat['AP']:.4f}")
    print(f"GCN AP: {best_metrics_gcn['AP']:.4f}")
    diff = metrics_gat['AP'] - best_metrics_gcn['AP']
    print(f"Difference (GAT - GCN): {diff:.4f}")
    
    # Save results
    with open(os.path.join(PLOTS_DIR, 'ablation_results.txt'), 'w') as f:
        f.write(f"GAT AP: {metrics_gat['AP']:.4f}\n")
        f.write(f"GCN AP: {best_metrics_gcn['AP']:.4f}\n")
        f.write(f"Difference: {diff:.4f}\n")

def main():
    parser = argparse.ArgumentParser(description="Run Experiments")
    parser.add_argument('--processed_dir', type=str, default=DATA_PROCESSED_DIR)
    parser.add_argument('--raw_file', type=str, default=os.path.join(DATA_RAW_DIR, BINDINGDB_FILENAME))
    parser.add_argument('--experiment', type=str, choices=['learning_curve', 'ablation', 'all'], default='all')
    args = parser.parse_args()
    
    print("Loading Dataset...")
    processed_path = os.path.join(args.processed_dir, 'data.pt')
    
    # Ensure processed directory exists
    os.makedirs(args.processed_dir, exist_ok=True)
    
    if os.path.exists(processed_path):
        print(f"Found processed data at {processed_path}. Loading...")
        dataset = BRD4Dataset(root=args.processed_dir)
    else:
        print("Processed data not found. Checking for cached CSV or raw file...")
        from src.data_processing import load_and_filter_data, clean_and_label_data
        
        # Check for cached CSV
        filtered_csv_path = os.path.join(args.processed_dir, 'brd4_filtered.csv')
        
        if os.path.exists(filtered_csv_path):
            print(f"Loading cached filtered data from {filtered_csv_path}...")
            df = pd.read_csv(filtered_csv_path)
            print(f"Loaded {len(df)} records.")
        else:
            if not os.path.exists(args.raw_file):
                print(f"ERROR: Raw data file not found at {args.raw_file}")
                print("Please provide path to BindingDB_All.tsv using --raw_file")
                return
                
            # Load and filter
            df = load_and_filter_data(args.raw_file)
            if df is None or df.empty:
                print("No data found or error loading data.")
                return
                
            print(f"Found {len(df)} records for BRD4.")
            print("Cleaning and Labeling...")
            df = clean_and_label_data(df)
            
            print(f"Saving filtered data to {filtered_csv_path}...")
            df.to_csv(filtered_csv_path, index=False)
            
        print(f"Creating Graph Dataset in {args.processed_dir}...")
        dataset = BRD4Dataset(root=args.processed_dir, df=df)
    
    if args.experiment in ['learning_curve', 'all']:
        run_learning_curve(dataset)
        
    if args.experiment in ['ablation', 'all']:
        run_ablation_study(dataset)

if __name__ == "__main__":
    import torch_geometric.loader # Import here to avoid issues if not installed in some envs
    main()
