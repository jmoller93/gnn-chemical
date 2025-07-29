#!/usr/bin/env python3
"""
Training script for molecular solubility prediction using Graph Neural Networks.
"""

import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import random
import hydra
from omegaconf import DictConfig

# Add src to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import create_dataloaders
from src.model import SolubilityGNN


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)


@hydra.main(version_base=None, config_path="../", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Set random seed
    set_seed(cfg.seed)
    
    # Create output directory
    os.makedirs(cfg.output.save_dir, exist_ok=True)
    
    print("🧪 Starting GNN Solubility Prediction Training")
    print(f"📊 Configuration:")
    print(f"  - Data path: {cfg.data.path}")
    print(f"  - Hidden channels: {cfg.model.hidden_channels}")
    print(f"  - Number of layers: {cfg.model.num_layers}")
    print(f"  - Batch size: {cfg.training.batch_size}")
    print(f"  - Learning rate: {cfg.training.learning_rate}")
    print(f"  - Max epochs: {cfg.training.max_epochs}")
    print(f"  - Random seed: {cfg.seed}")
    
    # Create data loaders
    print("\n📁 Loading and processing data...")
    train_loader, val_loader, test_loader, dataset = create_dataloaders(cfg)
    
    # Initialize model
    print("\n🤖 Initializing model...")
    model = SolubilityGNN(cfg)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.output.save_dir, "checkpoints"),
        filename=f"{cfg.output.model_name}-{{epoch:02d}}-{{val_loss:.2f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=cfg.training.patience,
        verbose=True
    )
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=cfg.output.save_dir,
        name="logs",
        version=None
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.system.accelerator,
        devices=cfg.system.devices,
        precision=cfg.system.precision,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train the model
    print("\n🚀 Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Test the model
    print("\n🧪 Running final evaluation...")
    if trainer.checkpoint_callback.best_model_path:
        print(f"Loading best model from: {trainer.checkpoint_callback.best_model_path}")
        # Load best model for testing
        best_model = SolubilityGNN.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            cfg=cfg
        )
        trainer.test(best_model, test_loader)
    else:
        trainer.test(model, test_loader)
    
    print("\n✅ Training completed!")
    print(f"📁 Results saved to: {cfg.output.save_dir}")
    print(f"📈 TensorBoard logs: {logger.log_dir}")
    print(f"💾 Best model: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main() 