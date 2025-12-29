import pandas as pd
import os

# Define path to Leash data
# Assuming the standard path used in prepare_leash_data.py
file_path = r'data/raw/leash-BELKA/train.csv'
output_path = 'leash_head.csv'

def inspect_data():
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        # Try to search for it or list dir? 
        # For now just fail gracefully
        return

    print(f"Reading first 10 rows from {file_path}...")
    try:
        # Read only first 10 rows
        df = pd.read_csv(file_path, nrows=10)
        
        print("Columns found:", df.columns.tolist())
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Saved first 10 rows to {output_path}")
        
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    inspect_data()
