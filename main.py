import os
import argparse
import pandas as pd
import numpy as np
import torch
from src.config import *
from src.data_processing import load_and_filter_data, clean_and_label_data
from src.dataset import BRD4Dataset, scaffold_split
from src.train import run_training
from src.dataset import BRD4Dataset, scaffold_split
from src.train import run_training

from src.utils import calculate_metrics, plot_loss_curve, plot_val_ap_curve, plot_confusion_matrix, plot_roc_curve, plot_pr_curve, set_seed
from src.config import *
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="GAT BRD4 Binding Prediction Pipeline")
    parser.add_argument('--raw_file', type=str, default=r'data/raw/leash-BELKA/train.csv',
                        help='Path to the raw Leash BELKA CSV file')
    parser.add_argument('--processed_dir', type=str, default=DATA_PROCESSED_DIR,
                        help='Directory to save/load processed data')
    parser.add_argument('--cv', type=int, default=0,
                        help='Number of folds for Cross-Validation (0 or 1 to disable)')
    parser.add_argument('--optimize', action='store_true',
                        help='Run hyperparameter optimization using Optuna')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of trials for optimization')
    args = parser.parse_args()

    args = parser.parse_args()

    # Set reproducibility
    set_seed(SEED)

    print("Starting GAT BRD4 Binding Prediction Pipeline...")
    
    # 1. Data Acquisition & Processing
    raw_path = args.raw_file
    processed_dir = args.processed_dir
    processed_path = os.path.join(processed_dir, 'data_leash.pt') 
    
    # Ensure processed directory exists if using a custom one
    os.makedirs(processed_dir, exist_ok=True)
    
    if os.path.exists(processed_path):
        print(f"Found processed data at {processed_path}. Loading...")
        if os.path.exists(os.path.join(processed_dir, 'data.pt')):
            print("WARNING: 'data.pt' exists. Assuming it is valid or the user handles conflicts.")
        dataset = BRD4Dataset(root=processed_dir)
    else:
        # Check for cached CSV
        filtered_csv_path = os.path.join(processed_dir, 'leash_brd4_filtered.csv')
        
        if not os.path.exists(filtered_csv_path):
             print(f"Cached filtered file {filtered_csv_path} not found.")
             if not os.path.exists(raw_path):
                print(f"ERROR: Raw data file not found at {raw_path}")
                print("Please download Leash BELKA data and place it in data/raw/leash-BELKA/ or specify path with --raw_file")
                return

             # Process Leash data
             # Two-step preparation
             intermediate_file = os.path.join(processed_dir, '..', 'intermediate', 'brd4_all.csv')
             
             # 1. Filter
             print("Step 1: Filtering Leash BELKA data for BRD4...")
             from filter_protein import filter_protein_records
             success = filter_protein_records(raw_path, intermediate_file, 'BRD4')
             if not success:
                 print("Filtering failed.")
                 return
                 
             # 2. Prepare (Shuffle/Format)
             print("Step 2: Preparing final dataset (Removing downsampling)...")
             from prepare_leash_data import prepare_training_data
             prepare_training_data(intermediate_file, filtered_csv_path, seed=42, chunk_size=1000000)
        
        if os.path.exists(filtered_csv_path):
            print(f"Loading cached filtered data from {filtered_csv_path}...")
            df = pd.read_csv(filtered_csv_path)
            print(f"Loaded {len(df)} records.")
            
            # RENAME COLUMNS
            if 'molecule_smiles' in df.columns:
                df.rename(columns={'molecule_smiles': 'Ligand SMILES'}, inplace=True)
            if 'binds' in df.columns:
                df.rename(columns={'binds': 'Label'}, inplace=True)
            
            print(f"Columns after renaming: {df.columns.tolist()}")

            print(f"Creating Graph Dataset in {processed_dir} (this may take a while)...")
            dataset = BRD4Dataset(root=processed_dir, df=df)
        else:
            print("Error: Failed to produce filtered CSV.")
            return
        
    # 2. Optimization or Split & Train
    if args.optimize:
        from src.optimize import run_optimization
        best_params = run_optimization(dataset, n_trials=args.n_trials)
        print("\nOptimization finished. You can now update src/config.py with these parameters.")
        
    elif args.cv > 1:
        print(f"Starting {args.cv}-Fold Scaffold Cross-Validation...")
        from src.dataset import scaffold_k_fold
        folds = scaffold_k_fold(dataset, k=args.cv)
        
        cv_metrics = {'AUC': [], 'AP': [], 'F1': []}
        
        for i, (train_dataset, val_dataset) in enumerate(folds):
            print(f"\n--- Fold {i+1}/{args.cv} ---")
            print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
            
            metrics = run_training(train_dataset, val_dataset, test_dataset=None, fold_idx=i)
            
            for k, v in metrics.items():
                cv_metrics[k].append(v)
                
        print("\n=== Cross-Validation Results ===")
        for k, v in cv_metrics.items():
            print(f"Mean {k}: {np.mean(v):.4f} +/- {np.std(v):.4f}")
            
    else:
        print("Splitting data (Scaffold Split)...")
        train_dataset, val_dataset, test_dataset = scaffold_split(dataset)
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # 3. Training
        print("Starting Training...")
        run_training(train_dataset, val_dataset, test_dataset)
    
    print("Pipeline Completed.")

if __name__ == "__main__":
    main()
