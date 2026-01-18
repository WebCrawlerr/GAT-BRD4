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
        
        # Run training (suppress plotting for speed)
        metrics = run_training(train_subset, val_dataset, test_dataset=None, plot=False, verbose=False)
        
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
    parser.add_argument('--experiment', type=str, choices=['learning_curve'], default='learning_curve')
    args = parser.parse_args()
    
    print("Loading Dataset...")
    processed_path = os.path.join(args.processed_dir, 'data_leash.pt') 
    
    # Ensure processed directory exists
    os.makedirs(args.processed_dir, exist_ok=True)
    
    if os.path.exists(processed_path):
        print(f"Found processed data at {processed_path}. Loading...")
        # We need to manually set the processed path if it's non-standard, but BRD4Dataset expects 'data.pt' by default.
        # To avoid hacking BRD4Dataset too much, we will just instantiate it.
        # However, BRD4Dataset hardcodes 'data.pt'.
        # Let's rename 'data.pt' to 'data_old_bindingdb.pt' if it exists and we want to enforce new data?
        # A clearer way:
        if os.path.exists(os.path.join(args.processed_dir, 'data.pt')):
            print("WARNING: 'data.pt' exists. Assuming it is valid or the user handles conflicts.")
        
        dataset = BRD4Dataset(root=args.processed_dir)
    else:
        print("Processed data not found. Checking for cached filtered CSV from Leash...")
        
        # Check for cached CSV
        filtered_csv_path = os.path.join(args.processed_dir, 'leash_brd4_filtered.csv')
        
        if not os.path.exists(filtered_csv_path):
             print(f"Cached filtered file {filtered_csv_path} not found.")
             if not os.path.exists(args.raw_file):
                print(f"ERROR: Raw data file not found at {args.raw_file}")
                return

             # Two-step preparation
             intermediate_file = os.path.join(args.processed_dir, '..', 'intermediate', 'brd4_all.csv')
             
             # 1. Filter
             print("Step 1: Filtering Leash BELKA data for BRD4...")
             from filter_protein import filter_protein_records
             success = filter_protein_records(args.raw_file, intermediate_file, 'BRD4')
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
            
            # Columns should already be renamed by prepare_leash_data, but check just in case
            if 'molecule_smiles' in df.columns:
                df.rename(columns={'molecule_smiles': 'Ligand SMILES'}, inplace=True)
            if 'binds' in df.columns:
                df.rename(columns={'binds': 'Label'}, inplace=True)
                
            print(f"Columns verified: {df.columns.tolist()}")

            print(f"Creating Graph Dataset in {args.processed_dir}...")
            dataset = BRD4Dataset(root=args.processed_dir, df=df)
        else:
            print("Error: Failed to produce filtered CSV.")
            return
            
    if args.experiment == 'learning_curve':
        run_learning_curve(dataset)

if __name__ == "__main__":
    main()
