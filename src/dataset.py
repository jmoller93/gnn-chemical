import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import List, Optional, Tuple
import numpy as np
from .utils import smiles_to_graph, split_data


class SolubilityDataset(Dataset):
    """Dataset for molecular solubility prediction using SMILES strings."""
    
    def __init__(self, csv_path: str, smiles_col: str = "SMILES", target_col: str = "Solubility"):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to the CSV file
            smiles_col: Name of the column containing SMILES strings
            target_col: Name of the column containing solubility labels
        """
        self.df = pd.read_csv(csv_path)
        self.smiles_col = smiles_col
        self.target_col = target_col
        
        # Remove rows with missing SMILES or target values
        self.df = self.df.dropna(subset=[smiles_col, target_col])
        
        # Convert SMILES to graphs and filter out invalid molecules
        self.valid_indices = []
        self.graphs = []
        self.targets = []
        
        print(f"Processing {len(self.df)} molecules...")
        for idx, row in self.df.iterrows():
            smiles = row[smiles_col]
            target = int(row[target_col])
            
            graph = smiles_to_graph(smiles)
            if graph is not None:
                self.valid_indices.append(idx)
                self.graphs.append(graph)
                self.targets.append(target)
        
        print(f"Successfully processed {len(self.graphs)} valid molecules")
        
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Tuple[Data, int]:
        """
        Get a single data sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (graph_data, target_label)
        """
        graph = self.graphs[idx]
        target = self.targets[idx]
        
        # Add target to the graph data object
        graph.y = torch.tensor([target], dtype=torch.long)
        
        return graph, target
    
    def get_class_counts(self) -> dict:
        """Get the count of each class for balancing."""
        unique, counts = np.unique(self.targets, return_counts=True)
        return dict(zip(unique, counts))
    
    def get_splits(self, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
        """
        Get train/validation/test split indices.
        
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation  
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        return split_data(self.graphs, train_ratio, val_ratio, seed)


def create_dataloaders(cfg) -> Tuple:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, dataset)
    """
    from torch_geometric.loader import DataLoader
    
    # Create dataset
    dataset = SolubilityDataset(
        csv_path=cfg.data.path,
        smiles_col=cfg.data.smiles_col,
        target_col=cfg.data.target_col
    )
    
    # Get splits
    train_indices, val_indices, test_indices = dataset.get_splits(
        train_ratio=cfg.data.train_split,
        val_ratio=cfg.data.val_split,
        seed=cfg.seed
    )
    
    # Create subset datasets
    train_dataset = [dataset[i][0] for i in train_indices]  # Only graph data for DataLoader
    val_dataset = [dataset[i][0] for i in val_indices]
    test_dataset = [dataset[i][0] for i in test_indices]
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.system.num_workers,
        persistent_workers=True if cfg.system.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.system.num_workers,
        persistent_workers=True if cfg.system.num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.system.num_workers,
        persistent_workers=True if cfg.system.num_workers > 0 else False
    )
    
    # Print dataset statistics
    class_counts = dataset.get_class_counts()
    print(f"Dataset statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    print(f"  Class distribution: {class_counts}")
    
    return train_loader, val_loader, test_loader, dataset 