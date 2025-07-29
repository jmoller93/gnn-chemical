# 🧪 GNN Solubility Prediction

A Graph Neural Network pipeline for molecular solubility prediction using PyTorch Lightning, PyTorch Geometric, Hydra, and Docker.

## 📁 Project Structure

```
chemical-gnn/
├── solubility_classification.csv    # Dataset (9,982 molecules)
├── requirements.txt                 # Python dependencies
├── config.yaml                      # Hydra configuration file
├── .gitignore                       # Git ignore rules
├── src/                             # Core library code
│   ├── __init__.py                  # Package initialization
│   ├── model.py                     # GNN model with PyTorch Lightning
│   ├── dataset.py                   # PyTorch dataset with graph processing
│   └── utils.py                     # SMILES-to-graph utilities
├── scripts/                         # Executable scripts
│   ├── train.py                     # Main training script with Hydra
│   └── inference.py                 # Prediction script
├── Dockerfile                       # Container definition (CPU)
├── Dockerfile.cuda                  # Container definition (GPU)
├── docker-compose.yml               # Multi-service orchestration (CPU)
├── docker-compose.cuda.yml          # Multi-service orchestration (GPU)
├── Makefile                         # Convenient commands
└── README.md                        # This file
```

## 🚀 Quick Start

### CPU Training (Default)

```bash
# Show all available commands
make help

# Build and train on CPU
make build
make train

# Monitor training (in a separate terminal)
make tensorboard

# Make predictions
make inference ARGS="--checkpoint outputs/checkpoints/best.ckpt --smiles 'CCO'"

# Interactive development
make dev
```

### 🚀 GPU Training (CUDA)

**Prerequisites**: NVIDIA Docker support required

```bash
# Build and train on GPU
make train-cuda

# Monitor training (in a separate terminal)
make tensorboard-cuda

# Make predictions with GPU acceleration
make inference-cuda ARGS="--checkpoint outputs/checkpoints/best.ckpt --smiles 'CCO'"

# Interactive development with GPU access
make dev-cuda

# Test CUDA setup
make test-cuda
```

### Configuration with Hydra

The project uses [Hydra](https://hydra.cc/) for configuration management:

```yaml
# config.yaml
data:
  path: "solubility_classification.csv"
  smiles_col: "SMILES"
  target_col: "Solubility"

model:
  hidden_channels: 64
  num_layers: 2
  dropout: 0.2

training:
  batch_size: 32
  learning_rate: 0.001
  max_epochs: 100
```

**Override parameters from command line:**
```bash
# Change batch size and learning rate
python scripts/train.py training.batch_size=64 training.learning_rate=0.01

# Use different model architecture
python scripts/train.py model.hidden_channels=128 model.num_layers=3

# Train with different data splits
python scripts/train.py data.train_split=0.9 data.val_split=0.05
```

### Using Docker Compose Directly

#### CPU Commands
```bash
# Build the image
docker compose build

# Run training
docker compose --profile train up train

# Start TensorBoard
docker compose --profile tensorboard up tensorboard

# Run inference
docker compose --profile inference run --rm inference python scripts/inference.py --help
```

#### GPU Commands
```bash
# Run CUDA training
docker compose -f docker-compose.cuda.yml --profile cuda-train up train-cuda

# Start TensorBoard for CUDA training
docker compose -f docker-compose.cuda.yml --profile cuda-tensorboard up tensorboard-cuda

# Run CUDA inference
docker compose -f docker-compose.cuda.yml --profile cuda-inference run --rm inference-cuda python scripts/inference.py --help

# Interactive CUDA shell
docker compose -f docker-compose.cuda.yml --profile cuda-dev run --rm dev-cuda
```

## 📊 Dataset

The pipeline works with `solubility_classification.csv` containing:
- **9,982 molecules** with SMILES strings
- **Binary solubility labels** (0 = insoluble, 1 = soluble)
- **Molecular descriptors** (MolWt, LogP, TPSA, etc.)

## 🤖 Model Architecture

- **Graph Neural Network** using GCN convolutions
- **Node features**: Atomic number, degree, formal charge, hybridization, aromaticity
- **Graph pooling**: Global mean pooling for molecule-level predictions
- **Binary classification** with cross-entropy loss

## 🎯 Usage Examples

### CPU Training
```bash
# Standard training
make train

# Monitor progress
make tensorboard
# Visit http://localhost:6006
```

### ⚡ GPU Training
```bash
# GPU-accelerated training (much faster!)
make train-cuda

# Monitor progress
make tensorboard-cuda
# Visit http://localhost:6006
```

### Inference
```bash
# CPU inference
make inference ARGS="--checkpoint outputs/checkpoints/best.ckpt --smiles 'CCO'"

# GPU inference (faster for large batches)
make inference-cuda ARGS="--checkpoint outputs/checkpoints/best.ckpt --input_file molecules.csv --output_file predictions.csv"

# Use custom config for inference
make inference ARGS="--checkpoint outputs/checkpoints/best.ckpt --config config.yaml --smiles 'CCO'"
```

### Development
```bash
# CPU development
make dev

# GPU development
make dev-cuda

# Inside the container:
python scripts/train.py                    # Run training
python scripts/train.py model.hidden_channels=128  # Override config
python scripts/inference.py --help         # Get inference help
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"  # Check GPU

# Direct Python usage
cd scripts/
python train.py model.hidden_channels=64   # Run from scripts directory
python inference.py --checkpoint ../outputs/checkpoints/best.ckpt --smiles 'CCO'
```

### Configuration Examples

```bash
# Experiment with different architectures
python scripts/train.py model.hidden_channels=32 model.num_layers=3 model.dropout=0.3

# Try different batch sizes and learning rates
python scripts/train.py training.batch_size=16 training.learning_rate=0.0001

# Use different data splits
python scripts/train.py data.train_split=0.7 data.val_split=0.15 data.test_split=0.15

# Change system settings
python scripts/train.py system.num_workers=8 system.precision=32

# Set custom output directory
python scripts/train.py output.save_dir=experiments/run_001 output.model_name=gnn_v2
```

## 📈 Expected Performance

### CPU Training
- **Training Time**: ~2-4 hours (depending on hardware)
- **Model Size**: ~3,000-5,000 parameters
- **Expected Accuracy**: 75-85% on test set

### ⚡ GPU Training
- **Training Time**: ~30-60 minutes (with CUDA)
- **Speedup**: 3-5x faster than CPU
- **Same accuracy as CPU training**

## 🔧 Configuration

### YAML Configuration Structure

```yaml
data:           # Data loading and preprocessing
model:          # Model architecture parameters
training:       # Training hyperparameters
system:         # System and hardware settings
output:         # Output directories and naming
seed:           # Random seed for reproducibility
```

### Hydra Features
- **Command-line overrides**: Change any parameter without editing files
- **Experiment management**: Automatic output directory creation
- **Configuration composition**: Combine multiple config files
- **Type checking**: Automatic validation of parameter types

## 📁 Output Structure

After training, `outputs/` contains:
```
outputs/
├── checkpoints/           # Model checkpoints
│   ├── best.ckpt         # Best model by validation loss
│   └── last.ckpt         # Latest checkpoint
└── logs/                 # TensorBoard logs
    └── version_0/        # Training metrics and graphs
```

## 🛠️ Development

The Docker Compose setup supports live development:

```bash
# CPU development with live code mounting
make dev

# GPU development with live code mounting
make dev-cuda
```

### Local Development

For local development without Docker:

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
cd scripts/
python train.py

# Run inference
python inference.py --checkpoint ../outputs/checkpoints/best.ckpt --smiles 'CCO'

# Use the package directly
from src.model import SolubilityGNN
from src.dataset import create_dataloaders
from src.utils import smiles_to_graph
```

## 🧹 Cleanup

```bash
# Remove containers and volumes
make clean

# Or manually
docker compose down --volumes --remove-orphans
```

## 📝 Requirements

### Basic Requirements
- Docker and Docker Compose
- ~4GB disk space for images and data

### For GPU Training
- NVIDIA Docker runtime (`nvidia-docker2`)
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed on host

### Installing NVIDIA Docker (Ubuntu/Debian)
```bash
# Install NVIDIA Docker
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

## 🔬 Pipeline Components

1. **Data Loading**: SMILES → Molecular graphs using RDKit
2. **Model Training**: GNN with PyTorch Lightning
3. **Configuration**: Hydra for flexible parameter management
4. **Evaluation**: Comprehensive metrics and logging
5. **Inference**: Flexible prediction interface
6. **Monitoring**: TensorBoard integration
7. **GPU Acceleration**: CUDA support for faster training

---

🚀 Ready to predict molecular solubility with Graph Neural Networks!
⚡ Use GPU acceleration for faster training!
🔧 Use Hydra for flexible configuration management!
