import pandas as pd
import argparse
import os
from tqdm import tqdm

def process_bindingdb_data(input_file, output_file, target_names, chunk_size=100000):
    """
    Reads BindingDB TSV file in chunks, filters for specific targets, and saves to CSV.
    """
    print(f"Starting BindingDB processing...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Targets: {target_names}")
    
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total_processed = 0
    saved_rows = 0
    
    # 1. Find the target column name
    try:
        header_df = pd.read_csv(input_file, sep='\t', nrows=0)
        target_col = 'Target Name'
        
        if target_col not in header_df.columns:
            # Try to find a similar column
            candidates = [c for c in header_df.columns if 'Target Name' in c]
            if candidates:
                target_col = candidates[0]
                print(f"Found target column: '{target_col}'")
            else:
                print("ERROR: Could not find 'Target Name' column in the file.")
                print(f"Available columns: {list(header_df.columns)}")
                return
    except Exception as e:
        print(f"Error reading header: {e}")
        return

    # 2. Process in chunks
    first_chunk = True
    
    try:
        with pd.read_csv(input_file, sep='\t', chunksize=chunk_size, on_bad_lines='skip', low_memory=False) as reader:
            for chunk in tqdm(reader, desc="Processing chunks"):
                # Filter
                # We check if ANY of the target names are present in the target column
                # We use str(x) to handle potential non-string data, and case-insensitive check could be added if needed
                # but BindingDB usually has consistent naming.
                
                mask = chunk[target_col].apply(lambda x: any(name in str(x) for name in target_names) if pd.notnull(x) else False)
                filtered_chunk = chunk[mask].copy()
                
                if not filtered_chunk.empty:
                    # Save to CSV
                    mode = 'w' if first_chunk else 'a'
                    header = first_chunk
                    filtered_chunk.to_csv(output_file, mode=mode, header=header, index=False)
                    
                    saved_rows += len(filtered_chunk)
                    first_chunk = False
                
                total_processed += len(chunk)

    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")
        return

    print(f"\nProcessing Complete!")
    print(f"Total rows scanned: {total_processed}")
    print(f"Rows saved (matching targets): {saved_rows}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter BindingDB data for specific targets.")
    
    parser.add_argument('--input', type=str, required=True, help='Path to BindingDB TSV file')
    parser.add_argument('--output', type=str, default='bindingdb_filtered.csv', help='Path to output CSV file')
    parser.add_argument('--targets', type=str, nargs='+', default=["BRD4", "Bromodomain-containing protein 4"], 
                        help='List of target names to filter for (default: BRD4 synonyms)')
    parser.add_argument('--chunk_size', type=int, default=100000, help='Rows per chunk')

    args = parser.parse_args()
    
    process_bindingdb_data(
        input_file=args.input,
        output_file=args.output,
        target_names=args.targets,
        chunk_size=args.chunk_size
    )
