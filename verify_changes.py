import os
import shutil
import pandas as pd
from src.dataset import BRD4Dataset
from filter_protein import filter_protein_records
from prepare_leash_data import prepare_training_data

# Define paths
RAW_FILE = 'leash_head.csv' # Created earlier, contains 10 rows
PROCESSED_DIR = 'data/processed_test_split'
INTERMEDIATE_FILE = os.path.join(PROCESSED_DIR, 'intermediate.csv')
FINAL_CSV = os.path.join(PROCESSED_DIR, 'leash_brd4_final.csv')

def test_pipeline():
    print("Testing SPLIT data pipeline...")
    
    # Cleanup previous test
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
    os.makedirs(PROCESSED_DIR)
    
    # ensure raw file exists
    if not os.path.exists(RAW_FILE):
        print(f"Error: {RAW_FILE} missing.")
        return

    # 1. Filter
    print("Step 1: Filtering...")
    success = filter_protein_records(RAW_FILE, INTERMEDIATE_FILE, 'BRD4', chunk_size=5)
    
    if not success or not os.path.exists(INTERMEDIATE_FILE):
        print("FAILED: Step 1 Filtering failed.")
        return
        
    print("Step 1 Done. Content of intermediate:")
    print(pd.read_csv(INTERMEDIATE_FILE).head())

    # 2. Prepare
    print("\nStep 2: Preparing...")
    prepare_training_data(INTERMEDIATE_FILE, FINAL_CSV, seed=42, chunk_size=5)
    
    if not os.path.exists(FINAL_CSV):
        print("FAILED: Step 2 Preparation failed.")
        return
        
    print("Step 2 Done. Content of final:")
    df = pd.read_csv(FINAL_CSV)
    print(df.head())
    print(f"Columns: {df.columns.tolist()}")
    
    # 3. Verify Dataset Loading
    print("\nStep 3: Creating BRD4Dataset...")
    try:
        dataset = BRD4Dataset(root=PROCESSED_DIR, df=df)
        print(f"Dataset created successfully. Size: {len(dataset)}")
    except Exception as e:
        print(f"FAILED to create dataset: {e}")
        return

    print("VERIFICATION SUCCESSFUL: Split pipeline works.")

if __name__ == "__main__":
    test_pipeline()
