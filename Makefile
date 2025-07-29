# GNN Solubility Prediction - Docker Compose Commands

.PHONY: help build train tensorboard inference shell dev clean train-cuda tensorboard-cuda inference-cuda dev-cuda

# Default target
help:
	@echo "🧪 GNN Solubility Prediction - Available Commands:"
	@echo ""
	@echo "CPU Commands:"
	@echo "  make build        - Build the Docker image"
	@echo "  make train        - Run training (CPU)"
	@echo "  make tensorboard  - Start TensorBoard (visit http://localhost:6006)"
	@echo "  make inference    - Show inference help (CPU)"
	@echo "  make shell        - Open interactive shell"
	@echo "  make dev          - Open development shell with live code mounting"
	@echo "  make clean        - Clean up Docker resources"
	@echo ""
	@echo "GPU Commands (requires NVIDIA Docker):"
	@echo "  make train-cuda        - Run training (GPU accelerated)"
	@echo "  make tensorboard-cuda  - Start TensorBoard for CUDA training"
	@echo "  make inference-cuda    - Show inference help (GPU accelerated)"
	@echo "  make dev-cuda          - Open development shell with GPU access"
	@echo ""
	@echo "Examples:"
	@echo "  make build && make train-cuda"
	@echo "  make inference-cuda ARGS='--checkpoint outputs/checkpoints/best.ckpt --smiles CCO'"
	@echo "  make tensorboard-cuda  # In a separate terminal after CUDA training starts"

# Build the Docker image
build:
	@echo "🔨 Building Docker image..."
	docker compose build

# CPU Commands
# Run training
train:
	@echo "🚀 Starting GNN training (CPU)..."
	@mkdir -p outputs
	docker compose --profile train up --build train

# Start TensorBoard
tensorboard:
	@echo "📈 Starting TensorBoard server..."
	@echo "🌐 Open http://localhost:6006 in your browser"
	docker compose --profile tensorboard up tensorboard

# Open interactive shell
shell:
	@echo "🐚 Opening interactive shell..."
	docker compose --profile shell run --rm shell

# Open development shell with live code mounting
dev:
	@echo "🛠️ Opening development shell..."
	docker compose --profile dev run --rm dev

# CUDA Commands
# Run CUDA training
train-cuda:
	@echo "🚀 Starting GNN training (GPU accelerated)..."
	@echo "⚡ Checking NVIDIA Docker support..."
	@docker run --rm --gpus all ubuntu:20.04 nvidia-smi || (echo "❌ NVIDIA Docker not available. Please install nvidia-docker2." && exit 1)
	@mkdir -p outputs
	docker compose -f docker-compose.cuda.yml --profile cuda-train up --build train-cuda

# Start TensorBoard for CUDA training
tensorboard-cuda:
	@echo "📈 Starting TensorBoard server for CUDA training..."
	@echo "🌐 Open http://localhost:6006 in your browser"
	docker compose -f docker-compose.cuda.yml --profile cuda-tensorboard up tensorboard-cuda

# Open CUDA development shell
dev-cuda:
	@echo "🛠️ Opening development shell with GPU access..."
	@echo "⚡ Checking NVIDIA Docker support..."
	@docker run --rm --gpus all ubuntu:20.04 nvidia-smi || (echo "❌ NVIDIA Docker not available. Please install nvidia-docker2." && exit 1)
	docker compose -f docker-compose.cuda.yml --profile cuda-dev run --rm dev-cuda

# Clean up Docker resources
clean:
	@echo "🧹 Cleaning up Docker resources..."
	docker compose down --volumes --remove-orphans
	docker compose -f docker-compose.cuda.yml down --volumes --remove-orphans
	docker system prune -f

# Quick test pipeline (build + minimal training test)
test:
	@echo "🧪 Running quick test..."
	@mkdir -p outputs
	docker compose --profile train run --rm train python -c "import sys; sys.path.append('/app'); from src.dataset import create_dataloaders; from src.model import SolubilityGNN; from omegaconf import OmegaConf; cfg = OmegaConf.load('/app/config.yaml'); print('✅ Pipeline test passed!')"

# Quick CUDA test
test-cuda:
	@echo "🧪 Running quick CUDA test..."
	@echo "⚡ Checking NVIDIA Docker support..."
	@docker run --rm --gpus all ubuntu:20.04 nvidia-smi || (echo "❌ NVIDIA Docker not available. Please install nvidia-docker2." && exit 1)
	@mkdir -p outputs
	docker compose -f docker-compose.cuda.yml --profile cuda-train run --rm train-cuda python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); import sys; sys.path.append('/app'); from omegaconf import OmegaConf; cfg = OmegaConf.load('/app/config.yaml'); from src.model import SolubilityGNN; print('✅ CUDA Pipeline test passed!')"

# Run inference (with custom arguments)
inference:
	@echo "🔮 Running inference (CPU)..."
	@if [ -n "$(ARGS)" ]; then \
		docker compose --profile inference run --rm inference python scripts/inference.py $(ARGS); \
	else \
		docker compose --profile inference run --rm inference python scripts/inference.py --help; \
	fi

# Run CUDA inference
inference-cuda:
	@echo "🔮 Running inference (GPU accelerated)..."
	@if [ -n "$(ARGS)" ]; then \
		docker compose -f docker-compose.cuda.yml --profile cuda-inference run --rm inference-cuda python scripts/inference.py $(ARGS); \
	else \
		docker compose -f docker-compose.cuda.yml --profile cuda-inference run --rm inference-cuda python scripts/inference.py --help; \
	fi
