import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.data import Data
import numpy as np
from typing import List, Optional, Tuple


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.
    
    Args:
        smiles: SMILES string representation of a molecule
        
    Returns:
        PyTorch Geometric Data object with node features and edge indices
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Get node features (atom features)
    node_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(),
            int(atom.IsInRing()),
        ]
        node_features.append(features)
    
    # Convert to tensor
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Get edge indices
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])  # Undirected graph
    
    if len(edge_indices) == 0:
        # Single atom molecule
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)


def get_node_feature_dim() -> int:
    """Return the dimension of node features."""
    return 7  # atomic_num, degree, formal_charge, hybridization, is_aromatic, total_hs, is_in_ring


def split_data(dataset, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42) -> Tuple[List, List, List]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: List of data samples
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    np.random.seed(seed)
    n_samples = len(dataset)
    indices = np.random.permutation(n_samples)
    
    train_end = int(train_ratio * n_samples)
    val_end = train_end + int(val_ratio * n_samples)
    
    train_indices = indices[:train_end].tolist()
    val_indices = indices[train_end:val_end].tolist()
    test_indices = indices[val_end:].tolist()
    
    return train_indices, val_indices, test_indices


def calculate_molecular_descriptors(smiles: str) -> dict:
    """
    Calculate basic molecular descriptors from SMILES.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary of molecular descriptors
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    
    return {
        'mol_weight': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'num_atoms': mol.GetNumAtoms(),
        'num_bonds': mol.GetNumBonds(),
        'num_rings': Descriptors.RingCount(mol),
        'tpsa': Descriptors.TPSA(mol),
    } 