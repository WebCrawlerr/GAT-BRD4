import pandas as pd
import os
import numpy as np
import argparse
from tqdm import tqdm

def prepare_training_data(input_file, output_file, seed, chunk_size):
    print(f"--- Preparing Training Data ---")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return

    # Ensure output dir exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    total_processed = 0
    
    # Set seed
    np.random.seed(seed)
    
    first_chunk = True
    
    # Process in chunks to handle memory if intermediate file is huge
    try:
        with pd.read_csv(input_file, chunksize=chunk_size) as reader:
            for chunk in tqdm(reader, desc="Formatting and Shuffling"):
                
                # Check columns and Rename
                # We expect: molecule_smiles -> Ligand SMILES
                #            binds -> Label
                
                # Check if it was already renamed or needs renaming
                if 'molecule_smiles' in chunk.columns:
                    chunk.rename(columns={'molecule_smiles': 'Ligand SMILES'}, inplace=True)
                if 'binds' in chunk.columns:
                    chunk.rename(columns={'binds': 'Label'}, inplace=True)
                
                # Standardize columns? 
                # Ensure we have Ligand SMILES and Label
                if 'Ligand SMILES' not in chunk.columns or 'Label' not in chunk.columns:
                    print(f"Warning: standard columns not found in chunk. Columns: {chunk.columns}")
                    continue
                
                # Keeping all data (NO UNDERSAMPLING)
                chunk_to_save = chunk
                
                # Shuffle within chunk? 
                # ideally we want global shuffle. 
                # With huge data, global shuffle is hard without loading all. 
                # We can shuffle chunk here, and assume the input was random or valid enough.
                # Or better: We can add a random index column, save all, then sort by it? 
                # For now, local shuffle is better than nothing.
                chunk_to_save = chunk_to_save.sample(frac=1, random_state=seed).reset_index(drop=True)
                
                # Save
                mode = 'w' if first_chunk else 'a'
                header = first_chunk
                chunk_to_save.to_csv(output_file, mode=mode, header=header, index=False)
                
                total_processed += len(chunk_to_save)
                first_chunk = False
                
    except Exception as e:
        print(f"Error during preparation: {e}")
        return

    print(f"Preparation Complete!")
    print(f"Total records ready: {total_processed}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Leash-BELKA data for training (Format & Shuffle, No Sampling).")
    
    parser.add_argument('--input_file', type=str, default=r'data/intermediate/brd4_all.csv', help='Path to filtered CSV')
    parser.add_argument('--output_file', type=str, default=r'data/processed/leash_brd4_final.csv', help='Path to final CSV')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--chunk_size', type=int, default=1_000_000)
    
    args = parser.parse_args()
    
    prepare_training_data(
        input_file=args.input_file,
        output_file=args.output_file,
        seed=args.seed,
        chunk_size=args.chunk_size
    )
