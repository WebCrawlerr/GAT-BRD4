import torch
from rdkit import Chem
from rdkit.Chem import rdmolops
from torch_geometric.data import Data
import numpy as np

# Allowed atom types for one-hot encoding
ALLOWED_ATOMS = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
    'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
    'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
]

def one_hot_encoding(x, permitted_list):
    """
    Maps input x to a one-hot vector based on permitted_list.
    If x is not in the list, maps to the last element (Unknown).
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(x == item) for item in permitted_list]
    return binary_encoding

def get_atom_features(atom):
    """
    Extracts features for a single atom.
    Features:
    - Atomic Symbol (one-hot)
    - Degree (one-hot)
    - Formal Charge (integer)
    - Hybridization (one-hot)
    - Aromaticity (boolean)
    - Total Num Hs (one-hot)
    """
    # Atomic Symbol
    atom_feature = one_hot_encoding(atom.GetSymbol(), ALLOWED_ATOMS)
    
    # Degree
    atom_feature += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # Formal Charge
    atom_feature += [atom.GetFormalCharge()]
    
    # Hybridization
    atom_feature += one_hot_encoding(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        'other'
    ])
    
    # Aromaticity
    atom_feature += [1 if atom.GetIsAromatic() else 0]
    
    # Total Num Hs
    atom_feature += one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    
    return np.array(atom_feature, dtype=np.float32)

def get_bond_features(bond):
    """
    Extracts features for a single bond.
    Features:
    - Bond Type (one-hot)
    - Conjugated (boolean)
    - In Ring (boolean)
    """
    bond_type = bond.GetBondType()
    bond_feature = one_hot_encoding(bond_type, [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ])
    
    bond_feature += [1 if bond.GetIsConjugated() else 0]
    bond_feature += [1 if bond.IsInRing() else 0]
    
    return np.array(bond_feature, dtype=np.float32)

def smiles_to_graph(smiles, label=None):
    """
    Converts a SMILES string to a PyTorch Geometric Data object.
    """
    # Fix for BELKA dataset: Replace Dysprosium [Dy] token with Carbon [C]
    # RDKit fails or produces invalid graphs for [Dy] in this context.
    if '[Dy]' in smiles:
        smiles = smiles.replace('[Dy]', 'C')
        
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Node features
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(get_atom_features(atom))
    
    x = torch.tensor(np.array(atom_features_list), dtype=torch.float)
    
    # Edge features and connectivity
    edge_indices = []
    edge_features_list = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        edge_feature = get_bond_features(bond)
        
        # Add both directions for undirected graph
        edge_indices.append([i, j])
        edge_features_list.append(edge_feature)
        
        edge_indices.append([j, i])
        edge_features_list.append(edge_feature)
        
    if len(edge_indices) == 0:
        # Handle single atom molecules or no bonds (rare but possible)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(get_bond_features(Chem.MolFromSmiles('CC').GetBondWithIdx(0)))), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.float)
        
    y = None
    if label is not None:
        y = torch.tensor([label], dtype=torch.float)
        
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles)
    return data
