import os
import pandas as pd
import shutil
import polars as pl
from src.dataset import BRD4Dataset
import torch

def create_mock_raw_data(filename):
    # Create a small mock dataset with two proteins
    data = {
        'id': range(10),
        'buildingblock1_smiles': ['C']*10,
        'buildingblock2_smiles': ['C']*10,
        'buildingblock3_smiles': ['C']*10,
        'molecule_smiles': ['C']*10,
        'protein_name': ['BRD4']*5 + ['HSA']*5,
        'binds': [1, 0, 1, 0, 0, 1, 0, 1, 0, 0]
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Created mock data at {filename}")
    return df

def test_raw_loading():
    mock_file = "mock_raw_data.csv"
    processed_dir = "test_processed_raw"
    
    # Cleanup previous run
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir, exist_ok=True)
    
    try:
        # Create mock data
        create_mock_raw_data(mock_file)
        
        # Test 1: Load BRD4
        print("\n--- Test 1: Loading BRD4 from raw ---")
        dataset_brd4 = BRD4Dataset(
            root=processed_dir,
            raw_file=mock_file,
            filtered_file=None, # Explicitly None to force raw usage
            target_name='BRD4'
        )
        print(f"Loaded BRD4 dataset size: {len(dataset_brd4)}")
        
        # We expect 5 samples (since we have 5 BRD4 records in mock data)
        # Note: Dataset logic might undersample negatives, so let's check processed file properties
        # But wait, logic is: all positives + 3 * positives negatives.
        # In mock: BRD4 has 2 positives (idx 0, 2) and 3 negatives (idx 1, 3, 4)
        # Target pos = 2. Target neg = min(3, 2*3) = 3.
        # Total expected = 5.
        
        assert len(dataset_brd4) == 5, f"Expected 5 samples for BRD4, got {len(dataset_brd4)}"
        print("Test 1 Passed!")
        
        # Cleanup processed file to force re-processing for next target
        processed_subdir = os.path.join(processed_dir, 'processed')
        if os.path.exists(processed_subdir):
            shutil.rmtree(processed_subdir)
        
        # Also clean root if any
        for f in os.listdir(processed_dir):
            p = os.path.join(processed_dir, f)
            if os.path.isfile(p):
                os.remove(p)
            
        # Test 2: Load HSA
        print("\n--- Test 2: Loading HSA from raw ---")
        dataset_hsa = BRD4Dataset(
            root=processed_dir,
            raw_file=mock_file,
            filtered_file=None,
            target_name='HSA'
        )
        print(f"Loaded HSA dataset size: {len(dataset_hsa)}")
        # HSA in mock: 2 positives (5, 7), 3 negatives (6, 8, 9)
        # Expected: 5
        assert len(dataset_hsa) == 5, f"Expected 5 samples for HSA, got {len(dataset_hsa)}"
        print("Test 2 Passed!")
        
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        if os.path.exists(mock_file):
            os.remove(mock_file)
        if os.path.exists(processed_dir):
            shutil.rmtree(processed_dir)

if __name__ == "__main__":
    test_raw_loading()
