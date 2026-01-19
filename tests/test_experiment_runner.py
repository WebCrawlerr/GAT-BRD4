import os
import pandas as pd
import shutil
import subprocess
import sys

def create_mock_raw_data(filename):
    data = {
        'id': range(20),
        'buildingblock1_smiles': ['C']*20,
        'buildingblock2_smiles': ['C']*20,
        'buildingblock3_smiles': ['C']*20,
        'molecule_smiles': ['C']*20,
        'protein_name': ['BRD4']*10 + ['HSA']*10,
        'binds': [1, 0]*10
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Created mock data at {filename}")

def test_run_experiments():
    mock_file = "mock_raw_exp.csv"
    processed_dir = "test_processed_exp"
    
    # Cleanup
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir, exist_ok=True)
    
    try:
        create_mock_raw_data(mock_file)
        
        print("\n--- Running run_experiments.py (dry run) ---")
        
        # We need to ensure we don't actually run a long training loop, 
        # but run_learning_curve calls run_training.
        # Ideally, run_experiments would have a dry-run flag, but we can just check if it initializes.
        # But wait, it will try to start training.
        # We can mock src.train.run_training? No, we are running subprocess.
        
        # Let's rely on the fact that with only 10 samples, it should be super fast or fail.
        # We want to check if it correctly loads the dataset from raw file.
        
        cmd = [
            sys.executable, 'run_experiments.py',
            '--raw_file', mock_file,
            '--processed_dir', processed_dir,
            '--target', 'BRD4',
            '--experiment', 'learning_curve'
        ]
        
        # Run for a short time or until completion
        # Since we have very few samples, it might crash in training due to batch size > sample size?
        # BATCH_SIZE is 4096 in config. Mock has 10.
        # PyTorch Geometric DataLoader handles this fine usually (just 1 batch).
        
        result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
        if result.returncode != 0:
            print("Command failed!")
            # Check if it failed due to something we expect (like 'plot settings' or similar)
            # But crucial part is: "Dataset loaded. Size: 10" (or similar)
            if "Dataset loaded" in result.stdout:
                print("However, Dataset loading seems to have succeeded.")
            else:
                raise Exception("Script failed before dataset loading.")
        else:
            print("Command succeeded!")
            
        # Check if processed file was created
        # PyG Dataset saves to os.path.join(root, 'processed')
        processed_file = os.path.join(processed_dir, 'processed', 'sampled_data_BRD4_full.pt')
        if os.path.exists(processed_file):
            print(f"Verified: Processed file created at {processed_file}")
            # Verify content size
            df = pd.read_pickle(processed_file)
            print(f"Processed file size: {len(df)}")
            # Should be roughly equal to BRD4 count (10) * sampling ratio logic
            # BRD4: 5 pos, 5 neg. Target: 5 pos, 15 neg (clipped to 5 neg). Total 10.
            assert len(df) > 0, "Dataset is empty!"
        else:
            print("Verified: Processed file NOT found!")
            print(f"Contents of {processed_dir}:")
            try:
                print(os.listdir(processed_dir))
                sub_proc = os.path.join(processed_dir, 'processed')
                if os.path.exists(sub_proc):
                    print(f"Contents of {sub_proc}:")
                    print(os.listdir(sub_proc))
            except:
                pass
            raise Exception("Processed file missing.")

    finally:
        if os.path.exists(mock_file):
            os.remove(mock_file)
        if os.path.exists(processed_dir):
            shutil.rmtree(processed_dir)

if __name__ == "__main__":
    test_run_experiments()
