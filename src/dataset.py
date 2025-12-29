import torch
from torch_geometric.data import InMemoryDataset
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import numpy as np
from src.features import smiles_to_graph
from tqdm import tqdm

class BRD4Dataset(InMemoryDataset):
    """
    PyTorch Geometric Dataset for BRD4 binding affinity prediction.
    
    This dataset loads processed graph data from a .pt file. If the .pt file
    does not exist, it processes a pandas DataFrame containing SMILES strings
    and labels, converting them into graph objects suitable for GNNs.
    
    Args:
        root (str): Root directory where the dataset should be saved.
        df (pd.DataFrame, optional): DataFrame containing 'Ligand SMILES' and 'Label' columns.
                                     Required for 'process' step if data is not already cached.
        transform (callable, optional): A function/transform that takes in an
            torch_geometric.data.Data object and returns a transformed version.
        pre_transform (callable, optional): A function/transform that takes in
            an torch_geometric.data.Data object and returns a transformed version.
    """
    def __init__(self, root, df=None, transform=None, pre_transform=None):
        self.df = df
        super(BRD4Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        if self.df is None:
            raise ValueError("Dataframe must be provided for processing.")
        
        data_list = []
        failed_count = 0
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing Molecules"):
            smiles = row['Ligand SMILES']
            label = row['Label']
            
            data = smiles_to_graph(smiles, label)
            if data is not None:
                data_list.append(data)
            else:
                failed_count += 1
                
        print(f"Processing complete. Successfully processed: {len(data_list)}. Failed: {failed_count}.")
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

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
