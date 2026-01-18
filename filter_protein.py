import pandas as pd
import os
import argparse
from tqdm import tqdm

def filter_protein_records(input_file, output_file, target_protein, chunk_size=1_000_000):
    print(f"--- Starting Protein Filter ---")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Target: {target_protein}")
    
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return False

    # Ensure output dir exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    total_processed = 0
    saved_rows = 0
    first_chunk = True
    
    try:
        with pd.read_csv(input_file, chunksize=chunk_size) as reader:
            for chunk in tqdm(reader, desc="Filtering chunks"):
                if 'protein_name' not in chunk.columns:
                    print("Error: 'protein_name' column missing.")
                    return False
                
                # Filter rows
                chunk_filtered = chunk[chunk['protein_name'] == target_protein]
                
                if chunk_filtered.empty:
                    total_processed += len(chunk)
                    continue
                
                # OPTIMIZATION: Keep only necessary columns to save disk space
                # We need: molecule_smiles (for graph), binds (label), buildingblock3_smiles (for split)
                cols_to_keep = ['molecule_smiles', 'binds', 'buildingblock3_smiles']
                # Check which ones exist
                existing_cols = [c for c in cols_to_keep if c in chunk_filtered.columns]
                chunk_filtered = chunk_filtered[existing_cols]
                
                # Save
                mode = 'w' if first_chunk else 'a'
                header = first_chunk
                chunk_filtered.to_csv(output_file, mode=mode, header=header, index=False)
                
                saved_rows += len(chunk_filtered)
                total_processed += len(chunk)
                first_chunk = False
                
    except Exception as e:
        print(f"Error during filtering: {e}")
        return False

    print(f"Filtering Complete.")
    print(f"Total rows scanned: {total_processed}")
    print(f"Rows matching {target_protein}: {saved_rows}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Filter Leash Data by Protein")
    parser.add_argument('--input_file', type=str, default=r'data/raw/leash-BELKA/train.csv')
    parser.add_argument('--output_file', type=str, default=r'data/intermediate/brd4_all.csv')
    parser.add_argument('--protein', type=str, default='BRD4')
    parser.add_argument('--chunk_size', type=int, default=1_000_000)
    
    args = parser.parse_args()
    
    filter_protein_records(args.input_file, args.output_file, args.protein, args.chunk_size)
