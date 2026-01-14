import torch
from torch_geometric.data import InMemoryDataset, Dataset
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import numpy as np
from src.features import smiles_to_graph
from tqdm import tqdm
import os
import pandas as pd
import polars as pl

class BRD4Dataset(Dataset):
    """
    PyTorch Geometric Dataset for BRD4 binding affinity prediction.
    
    Implements on-the-fly graph generation and balanced sampling to handle 
    large datasets (98M+) within Kaggle constraints.
    """
    def __init__(self, root, filtered_file=None, limit=None, test_mode=False, transform=None, pre_transform=None):
        self.filtered_file = filtered_file
        self.limit = limit
        self.test_mode = test_mode
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
        if self.test_mode:
            return ['processed_test_data.pt']
        file_name = f'sampled_data_limit_{self.limit}.pt' if self.limit else 'sampled_data_full.pt'
        return [file_name]

    def download(self):
        pass

    def process(self):
        if self.filtered_file is None:
            raise FileNotFoundError("Processed data not found and no 'filtered_file' provided.")

        print(f"Sampling from {self.filtered_file} using Polars (Test Mode={self.test_mode})...")
        
        # Determine strict counts based on 1:3 ratio and limit behavior
        
        # Detect column names lazily
        # Use simple read to get schema is fast enough
        schema = pl.scan_csv(self.filtered_file).collect_schema()
        
        rename_map = {}
        if 'molecule_smiles' in schema.names():
            rename_map['molecule_smiles'] = 'Ligand SMILES'
        
        # In train mode, we need labels
        if not self.test_mode:
            if 'binds' in schema.names():
                rename_map['binds'] = 'Label'
            
        # Scan dataset
        q = pl.scan_csv(self.filtered_file)
        if rename_map:
            q = q.rename(rename_map)

        # --- TEST MODE ---
        if self.test_mode:
            print("Processing test data (preserving all rows, ignoring labels)...")
            # In test mode, we just take everything
            if self.limit:
                q = q.limit(self.limit)
            
            final_df_pl = q.collect()
            print(f"Final Test Dataset: {final_df_pl.height} samples")
            
            final_df_pd = final_df_pl.to_pandas()
            final_df_pd.to_pickle(self.processed_paths[0])
            return

        # --- TRAIN MODE ---
        # Pass 1: Count classes
        print("Pass 1: Counting class distribution...")
        counts = q.group_by("Label").len().collect()
        
        # Extract counts
        total_pos = 0
        total_neg = 0
        
        try:
            total_pos = counts.filter(pl.col("Label") == 1)["len"][0]
        except:
            total_pos = 0
            
        try:
            total_neg = counts.filter(pl.col("Label") == 0)["len"][0]
        except:
            total_neg = 0
            
        print(f"Found: Positives={total_pos}, Negatives={total_neg}")
        
        # NEW STRATEGY: All Positives + Downsampled Negatives
        # 1. We want ALL positives (unless limit is extremly small)
        n_pos_target = total_pos
        
        if self.limit:
            if self.limit < n_pos_target:
                print(f"Warning: Limit ({self.limit}) is smaller than total positives ({total_pos}). Clipping positives.")
                n_pos_target = self.limit
                n_neg_target = 0
            else:
                # We have space. Fill with negatives up to 3x positives (or until limit hit)
                # Standard downsampling ratio 1:3
                desired_neg = n_pos_target * 3
                
                # Check if we fit in limit
                if n_pos_target + desired_neg <= self.limit:
                    n_neg_target = desired_neg
                else:
                    # Fill remaining space
                    n_neg_target = self.limit - n_pos_target
        else:
            # No limit: Take all positives and 3x negatives
            n_neg_target = min(total_neg, n_pos_target * 3)
            
        print(f"Target: Positives={n_pos_target}, Negatives={n_neg_target}")
        
        p_pos = n_pos_target / total_pos if total_pos > 0 else 0.0
        p_neg = n_neg_target / total_neg if total_neg > 0 else 0.0
        
        print(f"Sampling Probabilities: Pos={p_pos:.4f}, Neg={p_neg:.4f}")
        
        # Pass 2: Extract Samples
        # We can do this efficiently with Polars using hash-based sampling
        # (Deterministic but effective for large datasets)
        
        print("Pass 2: Extracting samples...")
        
        # Scaling factor for hash modulo
        M = 1000000
        
        # For positives
        q_pos = q.filter(pl.col("Label") == 1)
        if p_pos < 1.0:
             # Hash of SMILES to determining inclusion
             # We use modulo arithmetic on hash
             threshold = int(p_pos * M)
             q_pos = q_pos.filter((pl.col("Ligand SMILES").hash() % M) < threshold)

        # For negatives
        q_neg = q.filter(pl.col("Label") == 0)
        if p_neg < 1.0:
            threshold = int(p_neg * M)
            q_neg = q_neg.filter((pl.col("Ligand SMILES").hash() % M) < threshold)
            
        # Combine
        # Using concat on the queries
        final_lazy = pl.concat([q_pos, q_neg])
        
        # Global shuffle (Dataset typically small enough to collect then shuffle? or use rand sort)
        # We collect first then shuffle in pandas/numpy to be safe and compatible?
        # Or shuffle in Polars eagerly.
        
        # Collect
        final_df_pl = final_lazy.collect()
        
        print(f"Final Dataset: {final_df_pl.height} samples ({final_df_pl['Label'].sum()} positive)")
        
        # Convert to pandas for compatibility with existing pickle saving/loading
        final_df_pd = final_df_pl.to_pandas()
        
        # Shuffle (since we didn't do it globally in lazy mode)
        final_df_pd = final_df_pd.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save using pandas pickle
        final_df_pd.to_pickle(self.processed_paths[0])

    def len(self):
        return self._num_samples

    def get(self, idx):
        row = self.df.iloc[idx]
        smiles = row['Ligand SMILES']
        
        if self.test_mode:
            # No label expected
            label = None
            data = smiles_to_graph(smiles, label=None)
            # Attach ID logic
            # test.csv has 'id' column
            if 'id' in row:
                data.id = row['id']
            else:
                # Fallback if no ID? Should usually be there
                data.id = idx
            
            # Attach BB info if available (so splitting works if we ever wanted to split test data)
            if 'buildingblock3_smiles' in row:
                data.buildingblock3_smiles = row['buildingblock3_smiles']
                
            return data
            
        # Train mode
        label = row['Label']
        
        # On-the-fly conversion
        data = smiles_to_graph(smiles, label)
        
        # Attach Building Block info if available (for splitting)
        if 'buildingblock3_smiles' in row:
            data.buildingblock3_smiles = row['buildingblock3_smiles']
            
        return data

def generate_scaffold(smiles, include_chirality=False):
    """
    Compute the Bemis-Murcko scaffold for a SMILES string.
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
    return scaffold

def building_block_split(dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1, seed=42):
    """
    Splits the dataset based on buildingblock3_smiles.
    This is faster than scaffold split and chemically relevant for DEL libraries.
    """
    np.random.seed(seed)
    
    # Check if dataset has a dataframe (our BRD4Dataset does)
    if hasattr(dataset, 'df') and 'buildingblock3_smiles' in dataset.df.columns:
        print("Performing fast Building Block split using DataFrame...")
        groups = dataset.df.groupby('buildingblock3_smiles').indices
        bb_indices = list(groups.values())
    else:
        print("Performing generic Building Block split (slower)...")
        # Fallback for generic datasets
        bb_groups = defaultdict(list)
        for idx, data in enumerate(dataset):
            bb = getattr(data, 'buildingblock3_smiles', None)
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
