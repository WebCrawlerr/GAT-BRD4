import os
import argparse
from src.config import *
from src.dataset import BRD4Dataset, scaffold_split
from src.train import run_training
import os
import argparse
import numpy as np
import torch # Ensure torch is imported if needed for other parts, though mostly handled in other files
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
    parser.add_argument('--filtered_file', type=str, default=None,
                        help='Path to a pre-filtered CSV file (e.g., for Kaggle input). If provided, skips filtering step.')
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
    
    # Ensure processed directory exists
    os.makedirs(processed_dir, exist_ok=True)
    
    # 2. Initialize Dataset
    # The dataset class handles loading existing .pt files or processing the CSV if needed.
    # We pass the filtered_file path if provided.
    
    target_csv = None
    if args.filtered_file and os.path.exists(args.filtered_file):
        print(f"Using provided filtered file: {args.filtered_file}")
        target_csv = args.filtered_file
    elif os.path.exists(os.path.join(processed_dir, 'leash_brd4_filtered.csv')):
        target_csv = os.path.join(processed_dir, 'leash_brd4_filtered.csv')
        print(f"Using default filtered file: {target_csv}")
    
    # If explicit processed data exists, it will load that.
    # Otherwise it will try to process target_csv.
    # If neither exists, and no target_csv provided, it might fail (handled in dataset).
    
    print(f"Initializing Graph Dataset in {processed_dir}...")
    try:
        dataset = BRD4Dataset(root=processed_dir, filtered_file=target_csv)
    except FileNotFoundError as e:
        # Fallback to generation if not found (legacy path)
        print(f"Dataset not ready: {e}")
        # Here we could re-implement the filtering call if absolutely needed,
        # but for Kaggle usage the user usually provides the file.
        print("Please provide a valid --filtered_file or ensure processed data exists.")
        return
    except Exception as e:
        print(f"Error initializing dataset: {e}")
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
