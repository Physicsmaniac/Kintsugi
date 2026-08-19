# =============================================================================
# Makefile — Turnkey commands for Forensic Document Reconstructor
#
# Usage:
#   make help          — Show all available targets
#   make setup         — One-shot environment setup
#   make train-seam    — Train the seam compatibility model
#   make train-embedder— Train the page embedding model
#   make app           — Launch the Streamlit app
#   make benchmark     — Run the full benchmark suite
#   make pipeline      — End-to-end: train → benchmark → app
# =============================================================================

SHELL := /bin/bash
PYTHON := .venv/bin/python3
PIP := .venv/bin/pip
STREAMLIT := .venv/bin/streamlit

# Directories
CHECKPOINT_DIR := checkpoints
SEAM_DIR := $(CHECKPOINT_DIR)/seam
EMBEDDER_DIR := $(CHECKPOINT_DIR)/embedder
BENCHMARK_DIR := benchmark_results
DATA_DIR := data

# Model paths (defaults, override with: make train-seam SEAM_MODEL=path/to/model.pth)
SEAM_MODEL := $(SEAM_DIR)/best_seam_model.pth
PAGE_MODEL := $(EMBEDDER_DIR)/best_page_embedder.pth

# Training defaults (override on command line: make train-seam EPOCHS=20)
EPOCHS ?= 10
BATCH_SIZE ?= 256
LR ?= 1e-4
STEPS_PER_EPOCH ?= 3000
VAL_STEPS ?= 400
NUM_WORKERS ?= 0

# Embedder training defaults
EMB_EPOCHS ?= 15
EMB_PAGES_PER_BATCH ?= 8
EMB_STRIPS_PER_PAGE ?= 4
EMB_STEPS_PER_EPOCH ?= 1000

# Dataset defaults
LOCAL ?= 0
LOCAL_FLAG = $(if $(filter 1,$(LOCAL)),--local,)

# Benchmark defaults
PDF ?= $(DATA_DIR)/206-10001-10017.pdf
NUM_PAGES ?= 2 5 10
NUM_STRIPS ?= 10
STRATEGIES ?= greedy atsp hdbscan+greedy hdbscan+atsp

# =============================================================================
# PHONY targets
# =============================================================================
.PHONY: help setup train-seam train-embedder train-all app benchmark pipeline \
        shred evaluate clean verify test

# =============================================================================
# Help
# =============================================================================
help: ## Show this help message
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║   Forensic Document Reconstructor — Makefile             ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  Setup:"
	@echo "    make setup              Create venv & install dependencies"
	@echo "    make verify             Verify installation is working"
	@echo ""
	@echo "  Training:"
	@echo "    make train-seam         Train seam compatibility model (Phase 1)"
	@echo "    make train-embedder     Train page embedding model (Phase 2)"
	@echo "    make train-all          Train both models sequentially"
	@echo ""
	@echo "  Inference:"
	@echo "    make app                Launch Streamlit web app"
	@echo "    make shred PDF=file.pdf Shred a PDF into test strips"
	@echo ""
	@echo "  Evaluation:"
	@echo "    make benchmark          Run benchmark suite"
	@echo "    make evaluate           Quick single-config evaluation"
	@echo ""
	@echo "  Pipeline:"
	@echo "    make pipeline           Full end-to-end pipeline"
	@echo ""
	@echo "  Other:"
	@echo "    make test               Run smoke tests"
	@echo "    make clean              Remove generated files"
	@echo ""
	@echo "  Override defaults with: make train-seam EPOCHS=20 BATCH_SIZE=128"
	@echo ""

# =============================================================================
# Setup
# =============================================================================
setup: ## Create venv and install all dependencies
	@chmod +x setup.sh && ./setup.sh

verify: ## Verify the installation is working
	@PYTHONPATH=. $(PYTHON) -c "import torch; from src.models.seam_model import SeamResNet; from src.models.page_embedder import PageEmbeddingNet; from src.solver.greedy import solve_greedy; from src.solver.scoring import compute_score_matrix; from src.evaluation.metrics import evaluate_reconstruction; m = SeamResNet(); out = m(torch.randn(1,3,224,224)); print('✅ All core modules working. PyTorch', torch.__version__, '| CUDA' if torch.cuda.is_available() else '| CPU')"

# =============================================================================
# Training
# =============================================================================
$(SEAM_DIR):
	@mkdir -p $(SEAM_DIR)

$(EMBEDDER_DIR):
	@mkdir -p $(EMBEDDER_DIR)

train-seam: ## Train the seam compatibility model from scratch
	@echo "🚀 Training SeamResNet from scratch..."
	@mkdir -p $(SEAM_DIR)
	$(PYTHON) -m src.training.train_seam \
		--batch-size $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--steps-per-epoch $(STEPS_PER_EPOCH) \
		--val-steps $(VAL_STEPS) \
		--lr $(LR) \
		--num-workers $(NUM_WORKERS) \
		--output-dir $(SEAM_DIR) \
		$(LOCAL_FLAG)

train-seam-resume: $(SEAM_DIR) ## Resume seam model training from latest checkpoint
	@echo "🔄 Resuming SeamResNet training..."
	$(PYTHON) -m src.training.train_seam \
		--batch-size $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--steps-per-epoch $(STEPS_PER_EPOCH) \
		--val-steps $(VAL_STEPS) \
		--lr $(LR) \
		--num-workers $(NUM_WORKERS) \
		--output-dir $(SEAM_DIR) \
		--resume $(SEAM_DIR)/latest_checkpoint.pth \
		$(LOCAL_FLAG)

train-embedder: $(EMBEDDER_DIR) ## Train the page embedding model
	@echo "🔥 Training PageEmbeddingNet ($(EMB_EPOCHS) epochs)..."
	$(PYTHON) -m src.training.train_page_embedder \
		--epochs $(EMB_EPOCHS) \
		--lr $(LR) \
		--pages-per-batch $(EMB_PAGES_PER_BATCH) \
		--strips-per-page $(EMB_STRIPS_PER_PAGE) \
		--train-steps-per-epoch $(EMB_STEPS_PER_EPOCH) \
		--output-dir $(EMBEDDER_DIR) \
		$(LOCAL_FLAG)

train-all: train-seam train-embedder ## Train both models sequentially
	@echo "✅ Both models trained."

# =============================================================================
# Inference / App
# =============================================================================
app: ## Launch the Streamlit web app
	@echo "🚀 Launching Forensic Document Reconstructor..."
	PYTHONPATH=. $(STREAMLIT) run src/app/app.py

shred: ## Shred a PDF into test strips (set PDF=path/to/file.pdf)
	@echo "✂️  Shredding $(PDF)..."
	$(PYTHON) -c " \
	import sys; sys.path.insert(0, '.'); \
	from src.data.shredder import shred_pdf; \
	gt = shred_pdf('$(PDF)', 'shredded_output', num_strips=$(NUM_STRIPS), max_pages=None, shuffle=True, seed=42); \
	print(f'✅ Shredded into {len(gt)} strips → shredded_output/'); \
	"

# =============================================================================
# Evaluation
# =============================================================================
benchmark: ## Run the full benchmark suite (set PDF=path/to/file.pdf)
	@echo "📊 Running benchmark on $(PDF)..."
	@mkdir -p $(BENCHMARK_DIR)
	PYTHONPATH=. $(PYTHON) scripts/benchmark.py \
		--pdf "$(PDF)" \
		--seam-model $(SEAM_MODEL) \
		$(if $(wildcard $(PAGE_MODEL)),--page-model $(PAGE_MODEL),) \
		--num-pages $(NUM_PAGES) \
		--num-strips $(NUM_STRIPS) \
		--strategies $(STRATEGIES) \
		--output-dir $(BENCHMARK_DIR)
	@echo "✅ Results saved to $(BENCHMARK_DIR)/"

evaluate: ## Quick single evaluation (greedy baseline)
	@echo "🧪 Quick evaluation (greedy, 5 pages, 10 strips)..."
	@mkdir -p $(BENCHMARK_DIR)
	PYTHONPATH=. $(PYTHON) scripts/benchmark.py \
		--pdf $(PDF) \
		--seam-model $(SEAM_MODEL) \
		--num-pages 5 \
		--num-strips 10 \
		--strategies greedy \
		--output-dir $(BENCHMARK_DIR)

# =============================================================================
# Pipeline (end-to-end)
# =============================================================================
pipeline: ## Full pipeline: train both models → benchmark → launch app
	@echo "╔══════════════════════════════════════════════════╗"
	@echo "║   🚀 Running Full Pipeline                       ║"
	@echo "╚══════════════════════════════════════════════════╝"
	@echo ""
	@echo "Step 1/4: Training seam model..."
	@$(MAKE) train-seam
	@echo ""
	@echo "Step 2/4: Training page embedder..."
	@$(MAKE) train-embedder
	@echo ""
	@echo "Step 3/4: Running benchmark..."
	@$(MAKE) benchmark
	@echo ""
	@echo "Step 4/4: Launching app..."
	@$(MAKE) app

# =============================================================================
# Testing
# =============================================================================
test: ## Run smoke tests on all modules
	@echo "🧪 Running smoke tests..."
	@PYTHONPATH=. $(PYTHON) scripts/smoke_test.py

# =============================================================================
# Cleanup
# =============================================================================
clean: ## Remove generated files (checkpoints, benchmarks, cache)
	@echo "🧹 Cleaning..."
	rm -rf $(CHECKPOINT_DIR) $(BENCHMARK_DIR) shredded_output __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Clean"

clean-all: clean ## Remove everything including .venv
	rm -rf .venv
	@echo "✅ Full clean (including .venv)"
