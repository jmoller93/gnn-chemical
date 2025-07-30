# GNN Solubility Prediction - Docker Compose Commands

.PHONY: help build train tensorboard inference shell dev clean train-cuda tensorboard-cuda inference-cuda dev-cuda check-port kill-tensorboard

# Default target
help:
	@echo "🧪 GNN Solubility Prediction - Available Commands:"
	@echo ""
	@echo "CPU Commands:"
	@echo "  make build        - Build the Docker image"
	@echo "  make train        - Run training (CPU)"
	@echo "  make tensorboard  - Start TensorBoard for CPU training logs"
	@echo "  make inference    - Show inference help (CPU)"
	@echo "  make shell        - Open interactive shell"
	@echo "  make dev          - Open development shell with live code mounting"
	@echo "  make clean        - Clean up Docker resources"
	@echo ""
	@echo "GPU Commands (requires NVIDIA Docker):"
	@echo "  make train-cuda        - Run training (GPU accelerated)"
	@echo "  make tensorboard-cuda  - Start TensorBoard for CUDA training logs"
	@echo "  make inference-cuda    - Show inference help (GPU accelerated)"
	@echo "  make dev-cuda          - Open development shell with GPU access"
	@echo ""
	@echo "TensorBoard Access:"
	@echo "  After training, visit http://localhost:6006 to view logs"
	@echo "  Use 'tensorboard' for CPU training logs, 'tensorboard-cuda' for GPU logs"
	@echo "  Both commands now use explicit port forwarding (-p 6006:6006)"
	@echo ""
	@echo "Port Management:"
	@echo "  make check-port       - Check if port 6006 is available"
	@echo "  make kill-tensorboard - Stop all running TensorBoard instances"
	@echo ""
	@echo "Examples:"
	@echo "  make build && make train-cuda"
	@echo "  make inference-cuda ARGS='--checkpoint outputs/checkpoints/best.ckpt --smiles CCO'"
	@echo "  make tensorboard-cuda  # View GPU training progress"
	@echo "  make check-port && make tensorboard  # Check port before starting"

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
	@echo "📈 Starting TensorBoard server for CPU training logs..."
	@echo "🌐 Open http://localhost:6006 in your browser"
	@echo "📁 Serving logs from: ./outputs/logs"
	@echo "🔌 Using explicit port forwarding: -p 6006:6006"
	@mkdir -p outputs/logs
	docker run --rm -p 6006:6006 -v "$(PWD)/outputs:/app/outputs" gnn-solubility:latest \
		tensorboard --logdir=/app/outputs/logs --host=0.0.0.0 --port=6006

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
	@echo "📈 Starting TensorBoard server for CUDA training logs..."
	@echo "🌐 Open http://localhost:6006 in your browser"
	@echo "📁 Serving logs from: ./outputs/logs"
	@echo "🔌 Using explicit port forwarding: -p 6006:6006"
	@mkdir -p outputs/logs
	docker run --rm -p 6006:6006 -v "$(PWD)/outputs:/app/outputs" --gpus all gnn-solubility:cuda \
		tensorboard --logdir=/app/outputs/logs --host=0.0.0.0 --port=6006

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

# Port checking utilities
check-port:
	@echo "🔍 Checking if port 6006 is available..."
	@if lsof -i :6006 >/dev/null 2>&1; then \
		echo "⚠️  Port 6006 is already in use:"; \
		lsof -i :6006; \
		echo "💡 Use 'make kill-tensorboard' to stop existing TensorBoard instances"; \
	else \
		echo "✅ Port 6006 is available"; \
	fi

kill-tensorboard:
	@echo "🛑 Stopping any running TensorBoard instances..."
	@docker ps --filter "ancestor=gnn-solubility:latest" --filter "ancestor=gnn-solubility:cuda" --format "table {{.ID}}\t{{.Image}}\t{{.Command}}" | grep tensorboard | awk '{print $$1}' | xargs -r docker stop
	@docker compose down --remove-orphans >/dev/null 2>&1 || true
	@docker compose -f docker-compose.cuda.yml down --remove-orphans >/dev/null 2>&1 || true
	@echo "✅ TensorBoard cleanup complete"
