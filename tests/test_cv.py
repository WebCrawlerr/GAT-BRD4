import sys
import os
import torch
import numpy as np
from collections import defaultdict

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import scaffold_k_fold, generate_scaffold

class MockData:
    def __init__(self, smiles, idx):
        self.smiles = smiles
        self.idx = idx

class MockDataset(list):
    def __getitem__(self, idx):
        if isinstance(idx, list) or isinstance(idx, np.ndarray):
            return MockDataset([super(MockDataset, self).__getitem__(i) for i in idx])
        return super(MockDataset, self).__getitem__(idx)

def test_scaffold_k_fold():
    print("Testing Scaffold K-Fold...")
    
    # Create dummy dataset with known scaffolds
    data_list = [
        MockData("C1CCCCC1", 0), MockData("C1CCCCC1O", 1), MockData("C1CCCCC1N", 2), MockData("C1CCCCC1Cl", 3), # Scaffold A
        MockData("c1ccccc1", 4), MockData("c1ccccc1O", 5), MockData("c1ccccc1N", 6), # Scaffold B
        MockData("C1CC1", 7), MockData("C1CC1O", 8), # Scaffold C
        MockData("C", 9) # Scaffold D
    ]
    dataset = MockDataset(data_list)
    
    k = 3
    folds = scaffold_k_fold(dataset, k=k, seed=42)
    
    assert len(folds) == k
    
    total_val_samples = 0
    val_indices = set()
    
    for i, (train_set, val_set) in enumerate(folds):
        print(f"Fold {i}: Train={len(train_set)}, Val={len(val_set)}")
        
        # Check no overlap between train and val
        train_idxs = {d.idx for d in train_set}
        val_idxs_fold = {d.idx for d in val_set}
        
        assert train_idxs.isdisjoint(val_idxs_fold)
        
        total_val_samples += len(val_set)
        val_indices.update(val_idxs_fold)
        
    # Check that all samples were used in validation exactly once
    assert total_val_samples == len(dataset)
    assert len(val_indices) == len(dataset)
    
    print("Scaffold K-Fold Test Passed.")

if __name__ == "__main__":
    try:
        test_scaffold_k_fold()
        print("All CV tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
