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
    
    Refactored to handle large datasets by storing chunks on disk instead of loading
    everything into memory.
    """
    def __init__(self, root, filtered_file=None, chunk_size=100000, transform=None, pre_transform=None):
        self.filtered_file = filtered_file
        self.chunk_size = chunk_size
        super(BRD4Dataset, self).__init__(root, transform, pre_transform)
        
        # Load metadata if it exists
        if os.path.exists(self.metadata_path):
            self.metadata = torch.load(self.metadata_path)
            self._num_samples = self.metadata['num_samples']
            self.chunk_map = self.metadata['chunk_map']
        else:
            self._num_samples = 0
            self.chunk_map = {} # sample_idx -> (chunk_idx, local_idx)

        # Cache for loaded chunks to avoid hitting disk constantly
        self.loaded_chunks = {}
        self.max_cache_size = 5

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        # We can't list all chunk files easily here without knowing how many there are beforehand
        # So we just check for metadata and at least one chunk
        return ['metadata.pt', 'data_0.pt']

    @property
    def metadata_path(self):
        return os.path.join(self.processed_dir, 'metadata.pt')

    def download(self):
        pass

    def process(self):
        if self.filtered_file is None:
            # If we are here, it means proper processed files don't exist and we were not given a file to process
            raise FileNotFoundError("Processed data not found and no 'filtered_file' provided.")

        print(f"Processing {self.filtered_file} in chunks of {self.chunk_size}...")
        
        chunk_idx = 0
        total_samples = 0
        sample_map = {} # Map global_idx -> (chunk_idx, local_idx)
        
        # Read CSV in chunks
        for chunk_df in pd.read_csv(self.filtered_file, chunksize=self.chunk_size):
            data_list = []
            
            # Rename if needed (handling both conventions just in case)
            if 'molecule_smiles' in chunk_df.columns:
                chunk_df.rename(columns={'molecule_smiles': 'Ligand SMILES'}, inplace=True)
            if 'binds' in chunk_df.columns:
                chunk_df.rename(columns={'binds': 'Label'}, inplace=True)
            
            for _, row in tqdm(chunk_df.iterrows(), total=len(chunk_df), desc=f"Processing Chunk {chunk_idx}", leave=False):
                smiles = row['Ligand SMILES']
                label = row['Label']
                
                data = smiles_to_graph(smiles, label)
                
                if data is not None:
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                        
                    data_list.append(data)
                    
                    # Map this sample
                    sample_map[total_samples] = (chunk_idx, len(data_list) - 1)
                    total_samples += 1
            
            # Save chunk
            if len(data_list) > 0:
                chunk_path = os.path.join(self.processed_dir, f'data_{chunk_idx}.pt')
                torch.save(data_list, chunk_path)
                chunk_idx += 1
                
        # Save metadata
        metadata = {
            'num_samples': total_samples,
            'chunk_map': sample_map,
            'num_chunks': chunk_idx
        }
        torch.save(metadata, self.metadata_path)
        print(f"Processing complete. Total samples: {total_samples}. Chunks: {chunk_idx}.")

    def len(self):
        return self._num_samples

    def get(self, idx):
        if idx not in self.chunk_map:
            raise IndexError(f"Index {idx} out of range (0-{self._num_samples-1})")
            
        chunk_idx, local_idx = self.chunk_map[idx]
        
        # Check cache
        if chunk_idx in self.loaded_chunks:
            return self.loaded_chunks[chunk_idx][local_idx]
        
        # Load chunk
        chunk_path = os.path.join(self.processed_dir, f'data_{chunk_idx}.pt')
        data_list = torch.load(chunk_path, weights_only=False)
        
        # Update cache
        if len(self.loaded_chunks) >= self.max_cache_size:
            # Remove a random chunk (or oldest) - simple FIFO
            self.loaded_chunks.pop(next(iter(self.loaded_chunks)))
            
        self.loaded_chunks[chunk_idx] = data_list
        return data_list[local_idx]

def generate_scaffold(smiles, include_chirality=False):
    """
    Compute the Bemis-Murcko scaffold for a SMILES string.
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
    return scaffold

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
