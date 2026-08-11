#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — End-to-end thesis experiment runner
#
# This script runs the complete pipeline:
#   1. Train seam model (with cross-document negatives)
#   2. Train page embedder (contrastive learning)
#   3. Run benchmarks comparing all solver strategies
#   4. Generate comparison report
#
# Usage:
#   ./scripts/run_pipeline.sh                          # Use defaults
#   ./scripts/run_pipeline.sh --pdf my_doc.pdf         # Custom test PDF
#   ./scripts/run_pipeline.sh --skip-training          # Benchmark only
#   ./scripts/run_pipeline.sh --seam-only              # Train seam model only
#   ./scripts/run_pipeline.sh --quick                  # Quick run (2 epochs)
#
# =============================================================================
set -e

# --- Resolve project root ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# --- Defaults ---
PYTHON=".venv/bin/python3"
SEAM_DIR="checkpoints/seam"
EMBEDDER_DIR="checkpoints/embedder"
BENCHMARK_DIR="benchmark_results"
PDF="data/206-10001-10017.pdf"
SEAM_EPOCHS=10
EMBEDDER_EPOCHS=15
SEAM_BATCH_SIZE=256
SEAM_LR=1e-4
SEAM_STEPS=3000
NUM_WORKERS=8
SKIP_TRAINING=false
SEAM_ONLY=false
QUICK=false

# --- Parse CLI args ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --pdf)             PDF="$2"; shift ;;
        --seam-epochs)     SEAM_EPOCHS="$2"; shift ;;
        --embedder-epochs) EMBEDDER_EPOCHS="$2"; shift ;;
        --batch-size)      SEAM_BATCH_SIZE="$2"; shift ;;
        --lr)              SEAM_LR="$2"; shift ;;
        --skip-training)   SKIP_TRAINING=true ;;
        --seam-only)       SEAM_ONLY=true ;;
        --quick)           QUICK=true ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --pdf PATH              Test PDF for benchmarks (default: data/206-10001-10017.pdf)"
            echo "  --seam-epochs N         Seam training epochs (default: 10)"
            echo "  --embedder-epochs N     Embedder training epochs (default: 15)"
            echo "  --batch-size N          Seam batch size (default: 256)"
            echo "  --lr RATE               Learning rate (default: 1e-4)"
            echo "  --skip-training         Skip training, run benchmarks only"
            echo "  --seam-only             Only train seam model (skip embedder)"
            echo "  --quick                 Quick run (2 epochs, 500 steps)"
            echo "  -h, --help              Show this help"
            exit 0
            ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
    shift
done

# Quick mode overrides
if [ "$QUICK" = true ]; then
    SEAM_EPOCHS=2
    EMBEDDER_EPOCHS=2
    SEAM_STEPS=500
    echo -e "${YELLOW}⚡ Quick mode: 2 epochs, 500 steps${NC}"
fi

# --- Pre-flight checks ---
if [ ! -f "$PYTHON" ]; then
    echo -e "${RED}❌ Virtual environment not found. Run: make setup${NC}"
    exit 1
fi

echo -e "${BOLD}${CYAN}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   🔬 Forensic Document Reconstructor — Full Pipeline    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "  PDF:             ${PDF}"
echo -e "  Seam epochs:     ${SEAM_EPOCHS}"
echo -e "  Embedder epochs: ${EMBEDDER_EPOCHS}"
echo -e "  Batch size:      ${SEAM_BATCH_SIZE}"
echo -e "  Skip training:   ${SKIP_TRAINING}"
echo ""

TIMER_START=$SECONDS

# =============================================================================
# PHASE 1: Train Seam Model
# =============================================================================
if [ "$SKIP_TRAINING" = false ]; then
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  Phase 1: Training SeamResNet (cross-document negatives) ${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    mkdir -p "$SEAM_DIR"

    PYTHONPATH=. $PYTHON -m src.training.train_seam \
        --batch-size "$SEAM_BATCH_SIZE" \
        --epochs "$SEAM_EPOCHS" \
        --steps-per-epoch "$SEAM_STEPS" \
        --val-steps 400 \
        --lr "$SEAM_LR" \
        --num-workers "$NUM_WORKERS" \
        --output-dir "$SEAM_DIR"

    echo ""
    echo -e "${GREEN}✅ Seam model saved to ${SEAM_DIR}/${NC}"
    echo ""

    # =========================================================================
    # PHASE 2: Train Page Embedder
    # =========================================================================
    if [ "$SEAM_ONLY" = false ]; then
        echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${BOLD}  Phase 2: Training PageEmbeddingNet (contrastive)       ${NC}"
        echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""

        mkdir -p "$EMBEDDER_DIR"

        PYTHONPATH=. $PYTHON -m src.training.train_page_embedder \
            --epochs "$EMBEDDER_EPOCHS" \
            --lr "$SEAM_LR" \
            --pages-per-batch 8 \
            --strips-per-page 4 \
            --train-steps-per-epoch 1000 \
            --output-dir "$EMBEDDER_DIR"

        echo ""
        echo -e "${GREEN}✅ Page embedder saved to ${EMBEDDER_DIR}/${NC}"
        echo ""
    fi
fi

# =============================================================================
# PHASE 3: Benchmark
# =============================================================================
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}  Phase 3: Running Benchmark Suite                        ${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

mkdir -p "$BENCHMARK_DIR"

# Build benchmark command
BENCHMARK_CMD="PYTHONPATH=. $PYTHON scripts/benchmark.py \
    --pdf $PDF \
    --seam-model ${SEAM_DIR}/best_seam_model.pth \
    --num-pages 2 5 10 \
    --num-strips 10 \
    --output-dir $BENCHMARK_DIR"

# Add page model if it exists
if [ -f "${EMBEDDER_DIR}/best_page_embedder.pth" ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --page-model ${EMBEDDER_DIR}/best_page_embedder.pth"
    BENCHMARK_CMD="$BENCHMARK_CMD --strategies greedy atsp hdbscan+greedy hdbscan+atsp"
else
    echo -e "${YELLOW}⚠️  No page embedder found. Running greedy and ATSP only.${NC}"
    BENCHMARK_CMD="$BENCHMARK_CMD --strategies greedy atsp"
fi

eval "$BENCHMARK_CMD"

echo ""
echo -e "${GREEN}✅ Benchmark results saved to ${BENCHMARK_DIR}/${NC}"
echo ""

# =============================================================================
# Summary
# =============================================================================
ELAPSED=$(( SECONDS - TIMER_START ))
MINUTES=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo -e "${BOLD}${GREEN}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   ✅ Pipeline Complete!                                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "  ⏱️  Total time: ${MINUTES}m ${SECS}s"
echo ""
echo -e "  📁 Outputs:"
echo -e "    Seam model:    ${SEAM_DIR}/best_seam_model.pth"
if [ -f "${EMBEDDER_DIR}/best_page_embedder.pth" ]; then
echo -e "    Page embedder: ${EMBEDDER_DIR}/best_page_embedder.pth"
fi
echo -e "    Benchmarks:    ${BENCHMARK_DIR}/"
echo -e "    Training logs: ${SEAM_DIR}/training_log.csv"
echo ""
echo -e "  🚀 Launch the app:"
echo -e "    ${CYAN}make app${NC}"
echo ""
