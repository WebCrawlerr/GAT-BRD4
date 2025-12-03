import torch
from torch_geometric.data import InMemoryDataset
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import numpy as np
from src.features import smiles_to_graph
from tqdm import tqdm

class BRD4Dataset(InMemoryDataset):
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
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing Molecules"):
            smiles = row['Ligand SMILES']
            label = row['Label']
            
            data = smiles_to_graph(smiles, label)
            if data is not None:
                data_list.append(data)
        
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
