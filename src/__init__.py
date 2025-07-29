"""
GNN Solubility Prediction Package

Core modules for molecular solubility prediction using Graph Neural Networks.
"""

from .model import SolubilityGNN, MyGNN
from .dataset import SolubilityDataset, create_dataloaders
from .utils import smiles_to_graph, get_node_feature_dim, split_data, calculate_molecular_descriptors

__version__ = "1.0.0"
__author__ = "GNN Solubility Team"

__all__ = [
    "SolubilityGNN",
    "MyGNN", 
    "SolubilityDataset",
    "create_dataloaders",
    "smiles_to_graph",
    "get_node_feature_dim",
    "split_data",
    "calculate_molecular_descriptors"
] 