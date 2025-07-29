#!/usr/bin/env python3
"""
Inference script for molecular solubility prediction.
"""

import os
import sys
import torch
import pandas as pd
import argparse
from typing import List, Union
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import SolubilityGNN
from src.utils import smiles_to_graph


class SolubilityPredictor:
    """Wrapper class for making solubility predictions."""
    
    def __init__(self, checkpoint_path: str, config_path: str = "config.yaml"):
        """
        Initialize the predictor.
        
        Args:
            checkpoint_path: Path to the trained model checkpoint
            config_path: Path to the YAML configuration file
        """
        # Load configuration
        self.cfg = OmegaConf.load(config_path)
        
        # Load model from checkpoint
        self.model = SolubilityGNN.load_from_checkpoint(checkpoint_path, cfg=self.cfg)
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def predict_smiles(self, smiles: Union[str, List[str]]) -> np.ndarray:
        """
        Predict solubility for SMILES string(s).
        
        Args:
            smiles: Single SMILES string or list of SMILES strings
            
        Returns:
            Numpy array of predictions (probabilities for each class)
        """
        if isinstance(smiles, str):
            smiles = [smiles]
        
        predictions = []
        
        with torch.no_grad():
            for smile in smiles:
                # Convert SMILES to graph
                graph = smiles_to_graph(smile)
                
                if graph is None:
                    # Invalid SMILES - return neutral prediction
                    predictions.append([0.5, 0.5])
                    continue
                
                # Move to device
                graph = graph.to(self.device)
                
                # Add batch dimension
                graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=self.device)
                
                # Get prediction
                logits = self.model(graph)
                probs = torch.softmax(logits, dim=1)
                
                predictions.append(probs.cpu().numpy()[0])
        
        return np.array(predictions)
    
    def predict_soluble(self, smiles: Union[str, List[str]]) -> np.ndarray:
        """
        Predict if molecules are soluble (binary prediction).
        
        Args:
            smiles: Single SMILES string or list of SMILES strings
            
        Returns:
            Numpy array of binary predictions (1 = soluble, 0 = insoluble)
        """
        probs = self.predict_smiles(smiles)
        return (probs[:, 1] > 0.5).astype(int)
    
    def predict_solubility_score(self, smiles: Union[str, List[str]]) -> np.ndarray:
        """
        Get solubility scores (probability of being soluble).
        
        Args:
            smiles: Single SMILES string or list of SMILES strings
            
        Returns:
            Numpy array of solubility scores (0-1)
        """
        probs = self.predict_smiles(smiles)
        return probs[:, 1]


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Predict molecular solubility from SMILES")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default="../config.yaml", help="Path to config YAML file")
    parser.add_argument("--smiles", help="Single SMILES string to predict")
    parser.add_argument("--input_file", help="CSV file with SMILES column")
    parser.add_argument("--smiles_col", default="SMILES", help="Name of SMILES column")
    parser.add_argument("--output_file", help="Output CSV file for predictions")
    
    args = parser.parse_args()
    
    # Initialize predictor
    print(f"Loading model from: {args.checkpoint}")
    print(f"Using config: {args.config}")
    predictor = SolubilityPredictor(args.checkpoint, args.config)
    
    if args.smiles:
        # Single SMILES prediction
        print(f"Predicting solubility for: {args.smiles}")
        
        score = predictor.predict_solubility_score(args.smiles)[0]
        prediction = predictor.predict_soluble(args.smiles)[0]
        
        print(f"Solubility Score: {score:.4f}")
        print(f"Prediction: {'Soluble' if prediction == 1 else 'Insoluble'}")
        
    elif args.input_file:
        # Batch prediction from file
        print(f"Reading SMILES from: {args.input_file}")
        df = pd.read_csv(args.input_file)
        
        if args.smiles_col not in df.columns:
            raise ValueError(f"Column '{args.smiles_col}' not found in input file")
        
        smiles_list = df[args.smiles_col].tolist()
        print(f"Predicting solubility for {len(smiles_list)} molecules...")
        
        # Get predictions
        scores = predictor.predict_solubility_score(smiles_list)
        predictions = predictor.predict_soluble(smiles_list)
        
        # Add predictions to dataframe
        df['solubility_score'] = scores
        df['solubility_prediction'] = predictions
        df['solubility_label'] = df['solubility_prediction'].map({1: 'Soluble', 0: 'Insoluble'})
        
        # Save results
        if args.output_file:
            df.to_csv(args.output_file, index=False)
            print(f"Results saved to: {args.output_file}")
        else:
            print("\nPrediction Results:")
            print(df[['SMILES', 'solubility_score', 'solubility_label']].head(10))
            
        # Summary statistics
        soluble_count = (predictions == 1).sum()
        total_count = len(predictions)
        print(f"\nSummary:")
        print(f"Total molecules: {total_count}")
        print(f"Predicted soluble: {soluble_count} ({soluble_count/total_count*100:.1f}%)")
        print(f"Predicted insoluble: {total_count - soluble_count} ({(total_count - soluble_count)/total_count*100:.1f}%)")
        print(f"Average solubility score: {scores.mean():.4f}")
        
    else:
        print("Please provide either --smiles or --input_file")
        print("\nExamples:")
        print("  python inference.py --checkpoint ../outputs/checkpoints/best.ckpt --smiles 'CCO'")
        print("  python inference.py --checkpoint ../outputs/checkpoints/best.ckpt --input_file molecules.csv --output_file predictions.csv")


if __name__ == "__main__":
    main() 