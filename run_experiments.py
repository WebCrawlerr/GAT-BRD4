import os
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.config import *
from src.dataset import BRD4Dataset, building_block_split
from src.train import run_training
from src.model import GATModel
import torch_geometric.loader

def run_learning_curve(dataset, fractions=[0.2, 0.4, 0.6, 0.8, 1.0]):
    print("\n=== Running Learning Curve Experiment ===")
    
    # Use a fixed split for consistency
    train_dataset_full, val_dataset, test_dataset = building_block_split(
    dataset, 
    frac_train=0.8, 
    frac_val=0.1, 
    frac_test=0.1, 
    seed=42
)
    
    results = {'Fraction': [], 'Train_Size': [], 'Val_AP': [], 'Val_AUC': [], 'Val_F0.5': []}
    
    # Shuffle train dataset once to ensure random subsets are consistent if we took first N
    # But scaffold split already shuffles scaffolds.
    # We will take the first N samples from the shuffled train set.
    
    total_train_size = len(train_dataset_full)
    # Time tracking for info (optional, or just rely on standard logs)
    import time
    start_time = time.time()
    
    save_path = os.path.join(PLOTS_DIR, 'learning_curve_results.csv')

    for frac in fractions:
        # Check time for logging but don't break
        elapsed = time.time() - start_time
        
        subset_size = int(total_train_size * frac)
        print(f"\nTraining with {frac*100}% data ({subset_size} samples)... (Elapsed: {elapsed/3600:.2f}h)")
        
        # Create subset
        train_subset = train_dataset_full[:subset_size]
        
        # Run training (plot only for the last fraction to generate confusion matrix, etc.)
        is_last = (frac == fractions[-1])
        metrics = run_training(train_subset, val_dataset, test_dataset=None, plot=is_last, verbose=False)
        
        results['Fraction'].append(frac)
        results['Train_Size'].append(subset_size)
        results['Val_AP'].append(metrics['AP'])
        results['Val_AUC'].append(metrics['AUC'])
        results['Val_F0.5'].append(metrics['F0.5'])
        
        print(f"  -> AP: {metrics['AP']:.4f}, AUC: {metrics['AUC']:.4f}, F0.5: {metrics['F0.5']:.4f}")
        
        # Save results incrementally
        df_results = pd.DataFrame(results)
        df_results.to_csv(save_path, index=False)
        print(f"  [Progress Saved] to {save_path}")
        
    # Final Plotting
    print(f"\nResults saved to {save_path}")
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

def main():
    parser = argparse.ArgumentParser(description="Run Experiments")
    parser.add_argument('--processed_dir', type=str, default=DATA_PROCESSED_DIR)
    parser.add_argument('--raw_file', type=str, default=r'data/raw/leash-BELKA/train.csv')
    parser.add_argument('--target', type=str, default='BRD4', help='Target protein name (BRD4, HSA, sEH)')
    parser.add_argument('--filtered_file', type=str, default=None, help='Path to pre-filtered CSV')
    parser.add_argument('--experiment', type=str, choices=['learning_curve'], default='learning_curve')
    args = parser.parse_args()
    
    print("Loading Dataset...")
    processed_path = os.path.join(args.processed_dir, 'data_leash.pt') 
    
    # Ensure processed directory exists
    os.makedirs(args.processed_dir, exist_ok=True)
    
    # Initialize Dataset with fallback to raw file logic
    
    target_csv = None
    if args.filtered_file and os.path.exists(args.filtered_file):
        target_csv = args.filtered_file
        print(f"Using provided filtered file: {target_csv}")
    else:
        # Check for legacy filtered file (optional fallback)
        filtered_csv_path = os.path.join(args.processed_dir, f'leash_{args.target.lower()}_filtered.csv')
        if os.path.exists(filtered_csv_path):
            target_csv = filtered_csv_path
            print(f"Using found filtered file: {target_csv}")
    
    print(f"Initializing Graph Dataset in {args.processed_dir}...")
    try:
        dataset = BRD4Dataset(
            root=args.processed_dir, 
            raw_file=args.raw_file, 
            filtered_file=target_csv, 
            target_name=args.target
        )
    except FileNotFoundError as e:
        print(f"Dataset initialization failed: {e}")
        print("Please provide a valid --raw_file or ensure processed data exists.")
        return
            
    if args.experiment == 'learning_curve':
        run_learning_curve(dataset)

if __name__ == "__main__":
    main()
