import polars as pl
import os
import argparse
import time

def prepare_training_data(input_file, output_file, seed, chunk_size):
    print(f"--- Preparing Training Data (with Polars) ---")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return

    # Ensure output dir exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    start_time = time.time()
    
    try:
        # Scan the CSV lazily
        # specific low_memory=True might not be needed for polars as it is efficient, 
        # but let's rely on default optimization.
        q = pl.scan_csv(input_file)
        
        # Rename columns if they exist
        # Polars rename is strict by default, checking columns exists is harder in lazy mode 
        # without fetching schema. But usually we know the schema.
        # Let's inspect schema first to be safe, like the original code did.
        # Using read_csv with n_rows=0 or essentially fetching schema.
        schema = pl.scan_csv(input_file).collect_schema()
        
        rename_map = {}
        if 'molecule_smiles' in schema.names():
            rename_map['molecule_smiles'] = 'Ligand SMILES'
        if 'binds' in schema.names():
            rename_map['binds'] = 'Label'
            
        if rename_map:
            q = q.rename(rename_map)
            
        # Select only necessary columns
        # Original code checked for 'Ligand SMILES' and 'Label'
        # We can select them to prune other columns early
        q = q.select(['Ligand SMILES', 'Label'])
        
        # Shuffle
        # LazyFrame.sample is not essentially supported or failed in verification.
        # We skip global shuffle here. 
        # The main dataset loader 'src/dataset.py' handles sampling and shuffling.
        # q = q.sample(fraction=1.0, shuffle=True, seed=seed)
        
        # Streaming save
        print("Streaming to CSV...")
        q.sink_csv(output_file)
        
    except Exception as e:
        print(f"Error during preparation: {e}")
        return

    elapsed = time.time() - start_time
    print(f"Preparation Complete!")
    print(f"Time taken: {elapsed:.2f} seconds")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Leash-BELKA data for training (Format & Shuffle, No Sampling) using Polars.")
    
    parser.add_argument('--input_file', type=str, default=r'data/intermediate/brd4_all.csv', help='Path to filtered CSV')
    parser.add_argument('--output_file', type=str, default=r'data/processed/leash_brd4_final.csv', help='Path to final CSV')
    parser.add_argument('--seed', type=int, default=42)
    # chunk_size is generic in Polars sink_csv/streaming, not strictly exposed as arg here often needed, 
    # generally Polars handles it. We keep arg for compatibility but ignore valid usage or pass to streaming config if needed.
    parser.add_argument('--chunk_size', type=int, default=1_000_000)
    
    args = parser.parse_args()
    
    prepare_training_data(
        input_file=args.input_file,
        output_file=args.output_file,
        seed=args.seed,
        chunk_size=args.chunk_size
    )
