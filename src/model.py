import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MyGNN(torch.nn.Module):
    """Graph Neural Network for molecular property prediction."""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Final prediction layer
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, batch):
        """
        Forward pass through the GNN.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment vector [num_nodes]
            
        Returns:
            Graph-level predictions [batch_size, out_channels]
        """
        # Apply graph convolutions with ReLU activation and dropout
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # Don't apply activation after last conv
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pool node embeddings to get graph-level representation
        x = global_mean_pool(x, batch)
        
        # Final classification
        x = self.classifier(x)
        
        return x


class SolubilityGNN(pl.LightningModule):
    """PyTorch Lightning module for molecular solubility prediction."""
    
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        # Model architecture
        self.gnn = MyGNN(
            in_channels=cfg.model.in_channels,
            hidden_channels=cfg.model.hidden_channels,
            out_channels=2,  # Binary classification
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, batch):
        """Forward pass through the model."""
        return self.gnn(batch.x, batch.edge_index, batch.batch)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        logits = self(batch)
        loss = self.criterion(logits, batch.y.squeeze())
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y.squeeze()).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        logits = self(batch)
        loss = self.criterion(logits, batch.y.squeeze())
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y.squeeze()).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        logits = self(batch)
        loss = self.criterion(logits, batch.y.squeeze())
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y.squeeze()).float().mean()
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = Adam(self.parameters(), lr=self.cfg.training.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        } 