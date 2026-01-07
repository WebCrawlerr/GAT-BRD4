import torch
from torch_geometric.data import InMemoryDataset, Dataset
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import numpy as np
from src.features import smiles_to_graph
from tqdm import tqdm
import os
import pandas as pd

class BRD4Dataset(Dataset):
    """
    PyTorch Geometric Dataset for BRD4 binding affinity prediction.
    
    Implements on-the-fly graph generation and balanced sampling to handle 
    large datasets (98M+) within Kaggle constraints.
    """
    def __init__(self, root, filtered_file=None, limit=None, transform=None, pre_transform=None):
        self.filtered_file = filtered_file
        self.limit = limit
        super(BRD4Dataset, self).__init__(root, transform, pre_transform)
        
        # Load the sampled dataset (DataFrame)
        # We expect 'processed_file_names' to handle the creation if missing
        try:
            self.df = pd.read_pickle(self.processed_paths[0])
        except Exception as e:
            # Fallback for older .pt files if any, though unlikely
            try:
                self.df = torch.load(self.processed_paths[0], weights_only=False)
            except:
                raise e
            
        self._num_samples = len(self.df)
        print(f"Dataset loaded. Size: {self._num_samples}. Columns: {list(self.df.columns)}")

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        # We save the sampled dataframe as a .pt file (pickle format)
        file_name = f'sampled_data_limit_{self.limit}.pt' if self.limit else 'sampled_data_full.pt'
        return [file_name]

    def download(self):
        pass

    def process(self):
        if self.filtered_file is None:
            raise FileNotFoundError("Processed data not found and no 'filtered_file' provided.")

        print(f"Sampling from {self.filtered_file}...")
        
        # Pass 1: Count classes to determine sampling probabilities
        total_pos = 0
        total_neg = 0
        
        # Helper to detect column names for 'binds'
        label_col = 'binds' 
        
        # Quick peek to check columns
        peek = pd.read_csv(self.filtered_file, nrows=1)
        if 'Label' in peek.columns:
            label_col = 'Label'
        elif 'binds' in peek.columns:
            label_col = 'binds'
            
        print("Pass 1: Counting class distribution...")
        for chunk in pd.read_csv(self.filtered_file, usecols=[label_col], chunksize=500000):
            counts = chunk[label_col].value_counts()
            total_pos += counts.get(1, 0)
            total_neg += counts.get(0, 0)
            
        print(f"Found: Positives={total_pos}, Negatives={total_neg}")
        
        # Determine strict counts based on 1:3 ratio and limit
        # Strategy: Take ALL positives (up to limit/4), fill rest with negatives (up to 3*pos)
        
        if self.limit:
            n_pos_target = min(total_pos, self.limit // 4)
            n_neg_target = min(total_neg, self.limit - n_pos_target) # Fill remainder, but usually 3*pos
            
            # Refined strategy: First prioritize 1:3 ratio
            desired_neg = n_pos_target * 3
            if n_pos_target + desired_neg <= self.limit:
                n_neg_target = min(total_neg, desired_neg)
            else:
                # Limit is very tight? This shouldn't happen with limit formula min(pos, limit//4)
                pass
        else:
            # No limit -> Take all pos, 3x negs
            n_pos_target = total_pos
            n_neg_target = min(total_neg, total_pos * 3)
            
        print(f"Target: Positives={n_pos_target}, Negatives={n_neg_target}")
        
        p_pos = n_pos_target / total_pos if total_pos > 0 else 0
        p_neg = n_neg_target / total_neg if total_neg > 0 else 0
        
        print(f"Sampling Probabilities: Pos={p_pos:.4f}, Neg={p_neg:.4f}")
        
        # Pass 2: Extract Samples
        sampled_rows = []
        
        current_pos = 0
        current_neg = 0
        
        print("Pass 2: Extracting samples...")
        # Re-read full file
        for chunk in pd.read_csv(self.filtered_file, chunksize=100000):
            # Rename columns if raw file
            if 'molecule_smiles' in chunk.columns:
                chunk.rename(columns={'molecule_smiles': 'Ligand SMILES'}, inplace=True)
            if 'binds' in chunk.columns:
                chunk.rename(columns={'binds': 'Label'}, inplace=True)
            
            # Separate
            pos_chunk = chunk[chunk['Label'] == 1]
            neg_chunk = chunk[chunk['Label'] == 0]
            
            # Sample Positives
            if not pos_chunk.empty and p_pos > 0:
                if p_pos >= 1.0:
                    selected_pos = pos_chunk
                else:
                    selected_pos = pos_chunk.sample(frac=p_pos)
                sampled_rows.append(selected_pos)
                current_pos += len(selected_pos)
                
            # Sample Negatives
            if not neg_chunk.empty and p_neg > 0:
                if p_neg >= 1.0:
                    selected_neg = neg_chunk
                else:
                    selected_neg = neg_chunk.sample(frac=p_neg)
                sampled_rows.append(selected_neg)
                current_neg += len(selected_neg)
        
        # Combine
        if len(sampled_rows) > 0:
            final_df = pd.concat(sampled_rows, ignore_index=True)
            # Shuffle
            final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            final_df = pd.DataFrame(columns=['Ligand SMILES', 'Label'])
            
        print(f"Final Dataset: {len(final_df)} samples ({final_df['Label'].sum()} positive)")
        
        # Save using pandas pickle
        final_df.to_pickle(self.processed_paths[0])

    def len(self):
        return self._num_samples

    def get(self, idx):
        row = self.df.iloc[idx]
        smiles = row['Ligand SMILES']
        label = row['Label']
        
        # On-the-fly conversion
        data = smiles_to_graph(smiles, label)
        
        # Attach Building Block info if available (for splitting)
        if 'buildingblock1_smiles' in row:
            data.buildingblock1_smiles = row['buildingblock1_smiles']
            
        return data

def generate_scaffold(smiles, include_chirality=False):
    """
    Compute the Bemis-Murcko scaffold for a SMILES string.
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
    return scaffold

def building_block_split(dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1, seed=42):
    """
    Splits the dataset based on buildingblock1_smiles.
    This is faster than scaffold split and chemically relevant for DEL libraries.
    """
    np.random.seed(seed)
    
    # Check if dataset has a dataframe (our BRD4Dataset does)
    if hasattr(dataset, 'df') and 'buildingblock1_smiles' in dataset.df.columns:
        print("Performing fast Building Block split using DataFrame...")
        groups = dataset.df.groupby('buildingblock1_smiles').indices
        bb_indices = list(groups.values())
    else:
        print("Performing generic Building Block split (slower)...")
        # Fallback for generic datasets
        bb_groups = defaultdict(list)
        for idx, data in enumerate(dataset):
            bb = getattr(data, 'buildingblock1_smiles', None)
            if bb is None:
                # Fallback to scaffold if BB missing
                bb = generate_scaffold(data.smiles)
            bb_groups[bb].append(idx)
        bb_indices = list(bb_groups.values())

    np.random.shuffle(bb_indices)
    
    train_idxs, val_idxs, test_idxs = [], [], []
    
    train_cutoff = frac_train * len(dataset)
    val_cutoff = (frac_train + frac_val) * len(dataset)
    
    for indices in bb_indices:
        if len(train_idxs) + len(indices) <= train_cutoff:
            train_idxs.extend(indices)
        elif len(train_idxs) + len(val_idxs) + len(indices) <= val_cutoff:
            val_idxs.extend(indices)
        else:
            test_idxs.extend(indices)
            
    print(f"Split results: Train={len(train_idxs)}, Val={len(val_idxs)}, Test={len(test_idxs)}")
    
    # Return subsets (Lazy!)
    from torch.utils.data import Subset
    return Subset(dataset, train_idxs), Subset(dataset, val_idxs), Subset(dataset, test_idxs)

def scaffold_split(dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1, seed=42):
    """
    Splits the dataset based on scaffolds.
    """
    np.random.seed(seed)
    
    scaffolds = defaultdict(list)
    for idx, data in enumerate(dataset):
        scaffold = generate_scaffold(data.smiles)
        scaffolds[scaffold].append(idx)
        
    # Sort scaffolds by size (descending) to ensure balanced-ish splits if possible, 
    # but standard scaffold split usually just takes them in order or randomizes the groups.
    # Here we shuffle the scaffold groups.
    
    scaffold_sets = list(scaffolds.values())
    np.random.shuffle(scaffold_sets)
    
    train_idxs, val_idxs, test_idxs = [], [], []
    
    train_cutoff = frac_train * len(dataset)
    val_cutoff = (frac_train + frac_val) * len(dataset)
    
    for scaffold_set in scaffold_sets:
        if len(train_idxs) + len(scaffold_set) <= train_cutoff:
            train_idxs.extend(scaffold_set)
        elif len(train_idxs) + len(val_idxs) + len(scaffold_set) <= val_cutoff:
            val_idxs.extend(scaffold_set)
        else:
            test_idxs.extend(scaffold_set)
            
    return dataset[train_idxs], dataset[val_idxs], dataset[test_idxs]

def scaffold_k_fold(dataset, k=5, seed=42):
    """
    Splits the dataset into K folds based on scaffolds.
    Returns a list of (train_dataset, val_dataset) tuples.
    """
    np.random.seed(seed)
    
    scaffolds = defaultdict(list)
    for idx, data in enumerate(dataset):
        scaffold = generate_scaffold(data.smiles)
        scaffolds[scaffold].append(idx)
        
    scaffold_sets = list(scaffolds.values())
    np.random.shuffle(scaffold_sets)
    
    folds = [[] for _ in range(k)]
    
    # Distribute scaffolds to folds to ensure roughly equal size
    # A simple greedy approach: add next scaffold to the smallest fold
    for scaffold_set in scaffold_sets:
        # Find fold with minimum number of samples
        min_fold_idx = np.argmin([len(fold) for fold in folds])
        folds[min_fold_idx].extend(scaffold_set)
        
    # Create Train/Val splits for each fold
    # Fold i is Val, rest are Train
    k_fold_datasets = []
    
    for i in range(k):
        val_idxs = folds[i]
        train_idxs = []
        for j in range(k):
            if i != j:
                train_idxs.extend(folds[j])
                
        k_fold_datasets.append((dataset[train_idxs], dataset[val_idxs]))
        
    return k_fold_datasets
