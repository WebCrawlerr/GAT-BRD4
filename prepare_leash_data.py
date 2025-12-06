import pandas as pd
import os
import numpy as np
import argparse
from tqdm import tqdm

def process_leash_data(input_file, output_dir, output_filename, target_protein, negative_rate, seed, chunk_size):
    print(f"Starting processing...")
    print(f"Input File: {input_file}")
    print(f"Output Directory: {output_dir}")
    print(f"Target Protein: {target_protein}")
    print(f"Negative Sample Rate: {negative_rate}")
    
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found at {input_file}")
        return

    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize counters
    total_processed = 0
    saved_rows = 0
    positives_saved = 0
    negatives_saved = 0
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    first_chunk = True
    
    # Process in chunks
    try:
        with pd.read_csv(input_file, chunksize=chunk_size) as reader:
            for chunk in tqdm(reader, desc="Processing chunks"):
                # Filter by protein
                if 'protein_name' in chunk.columns:
                    chunk_filtered = chunk[chunk['protein_name'] == target_protein].copy()
                else:
                    print("Warning: 'protein_name' column not found. Skipping filtering by protein.")
                    chunk_filtered = chunk.copy()
                
                if chunk_filtered.empty:
                    continue
                
                # Separate positives and negatives
                if 'binds' in chunk_filtered.columns:
                    positives = chunk_filtered[chunk_filtered['binds'] == 1]
                    negatives = chunk_filtered[chunk_filtered['binds'] == 0]
                    
                    # Sample negatives
                    if negative_rate < 1.0:
                        negatives = negatives.sample(frac=negative_rate, random_state=seed)
                    
                    # Combine
                    chunk_to_save = pd.concat([positives, negatives])
                    
                    # Update counters
                    positives_saved += len(positives)
                    negatives_saved += len(negatives)
                else:
                    chunk_to_save = chunk_filtered
                
                # Shuffle chunk to mix positives and negatives
                chunk_to_save = chunk_to_save.sample(frac=1, random_state=seed).reset_index(drop=True)
                
                # Save to CSV
                mode = 'w' if first_chunk else 'a'
                header = first_chunk
                chunk_to_save.to_csv(output_path, mode=mode, header=header, index=False)
                
                saved_rows += len(chunk_to_save)
                total_processed += len(chunk)
                first_chunk = False
                
    except PermissionError:
        print(f"\nERROR: Permission denied accessing {input_file}.")
        print("Please ensure the file is not open in another program.")
        return
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        return

    print("\nProcessing Complete!")
    print(f"Total rows processed: {total_processed}")
    print(f"Rows saved: {saved_rows}")
    print(f"Positives saved: {positives_saved}")
    print(f"Negatives saved: {negatives_saved}")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and prepare Leash-BELKA dataset.")
    
    # Paths
    parser.add_argument('--input_file', type=str, default=r'data/raw/leash-BELKA/train.csv', help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, default=r'data/processed', help='Directory to save output')
    parser.add_argument('--output_filename', type=str, default='leash_brd4_filtered.csv', help='Output filename')
    
    # Filtering
    parser.add_argument('--protein', type=str, default='BRD4', help='Target protein name (BRD4, HSA, sEH)')
    parser.add_argument('--negative_rate', type=float, default=0.05, help='Fraction of negative samples to keep')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--chunk_size', type=int, default=1_000_000, help='Chunk size for processing')
    
    args = parser.parse_args()
    
    process_leash_data(
        input_file=args.input_file,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        target_protein=args.protein,
        negative_rate=args.negative_rate,
        seed=args.seed,
        chunk_size=args.chunk_size
    )
