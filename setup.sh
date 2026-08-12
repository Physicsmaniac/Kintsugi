#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-shot environment setup for Forensic Document Reconstructor
#
# Usage:
#   chmod +x setup.sh && ./setup.sh
#
# This script:
#   1. Creates a Python virtual environment (.venv/)
#   2. Installs all dependencies (CPU or GPU PyTorch)
#   3. Verifies the installation
#   4. Prints next steps
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   Forensic Document Reconstructor — Setup        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# --- Detect GPU ---
USE_GPU=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        USE_GPU=true
        echo -e "${GREEN}🖥️  NVIDIA GPU detected — will install CUDA PyTorch${NC}"
    fi
fi

if [ "$USE_GPU" = false ]; then
    echo -e "${YELLOW}💻 No GPU detected — will install CPU-only PyTorch${NC}"
fi
echo ""

# --- Check Python ---
PYTHON=""
for candidate in python3.11 python3.12 python3.10 python3; do
    if command -v "$candidate" &> /dev/null; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "${RED}❌ Python 3 not found. Install Python 3.10+ first.${NC}"
    exit 1
fi

PY_VERSION=$($PYTHON --version 2>&1)
echo -e "Using: ${GREEN}${PY_VERSION}${NC} (${PYTHON})"
echo ""

# --- Create venv ---
if [ -d ".venv" ]; then
    echo -e "${YELLOW}⚠️  Existing .venv directory found. Reusing .venv...${NC}"
else
    echo -e "${CYAN}📦 Creating virtual environment (.venv)...${NC}"
    $PYTHON -m venv .venv
fi

if [ ! -d ".venv" ]; then
    echo -e "${CYAN}📦 Creating virtual environment...${NC}"
    $PYTHON -m venv .venv
fi

# Activate
source .venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"
echo ""

# --- Install PyTorch ---
echo -e "${CYAN}🔥 Installing PyTorch...${NC}"
if [ "$USE_GPU" = true ]; then
    CUDA_VER=$(nvidia-smi | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | awk '{print $3}')
    if [ -n "$CUDA_VER" ]; then
        echo -e "${CYAN}🔥 Detected Max CUDA Version: ${CUDA_VER}${NC}"
        MAJOR=$(echo $CUDA_VER | cut -d. -f1)
        MINOR=$(echo $CUDA_VER | cut -d. -f2)
        
        if [ "$MAJOR" -lt 12 ] || ( [ "$MAJOR" -eq 12 ] && [ "$MINOR" -lt 1 ] ); then
            echo -e "${YELLOW}⚠️  Driver supports < CUDA 12.1. Installing PyTorch cu118 fallback...${NC}"
            pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu118
        elif [ "$MAJOR" -eq 12 ] && [ "$MINOR" -lt 4 ]; then
            echo -e "${YELLOW}⚠️  Driver supports < CUDA 12.4. Installing PyTorch cu121 fallback...${NC}"
            pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121
        else
            echo -e "${GREEN}✅ Driver supports latest CUDA. Installing default PyTorch...${NC}"
            pip install --quiet torch torchvision
        fi
    else
        echo -e "${YELLOW}⚠️  Could not detect CUDA version. Installing default PyTorch...${NC}"
        pip install --quiet torch torchvision
    fi
else
    pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# --- Install project dependencies ---
echo -e "${CYAN}📚 Installing project dependencies...${NC}"
pip install --quiet -r requirements.txt

# --- Install optional thesis dependencies ---
echo -e "${CYAN}🧪 Installing thesis dependencies (HDBSCAN, UMAP, python-tsp)...${NC}"
pip install --quiet hdbscan umap-learn python-tsp 2>/dev/null || {
    echo -e "${YELLOW}⚠️  Some optional deps failed to install. Clustering/ATSP may not work.${NC}"
    echo -e "${YELLOW}   This is normal on some Python versions. You can try: pip install hdbscan umap-learn python-tsp${NC}"
}

# --- System dependencies ---
echo ""
echo -e "${CYAN}📋 Checking system dependencies...${NC}"
if command -v pdftoppm &> /dev/null; then
    echo -e "${GREEN}✅ poppler-utils installed${NC}"
else
    echo -e "${YELLOW}⚠️  poppler-utils not found. PDF shredding requires it.${NC}"
    echo -e "${YELLOW}   Install: sudo apt-get install poppler-utils (Ubuntu)${NC}"
    echo -e "${YELLOW}            brew install poppler (macOS)${NC}"
fi

# --- Verify ---
echo ""
echo -e "${CYAN}🔍 Verifying installation...${NC}"
python -c "
import sys
sys.path.insert(0, '.')
errors = []

try:
    import torch
    print(f'  ✅ PyTorch {torch.__version__}', '(CUDA)' if torch.cuda.is_available() else '(CPU)')
except ImportError as e:
    errors.append(f'PyTorch: {e}')

try:
    from src.models.seam_model import SeamResNet
    from src.models.page_embedder import PageEmbeddingNet
    print('  ✅ Models (SeamResNet, PageEmbeddingNet)')
except ImportError as e:
    errors.append(f'Models: {e}')

try:
    from src.data.preprocessing import preprocess_pair
    print('  ✅ Preprocessing pipeline')
except ImportError as e:
    errors.append(f'Preprocessing: {e}')

try:
    from src.solver.greedy import solve_greedy
    from src.solver.scoring import compute_score_matrix
    print('  ✅ Solver (greedy, scoring)')
except ImportError as e:
    errors.append(f'Solver: {e}')

try:
    from src.solver.atsp import solve_atsp
    print('  ✅ Solver (ATSP)')
except ImportError:
    print('  ⚠️  ATSP solver (install python-tsp)')

try:
    from src.solver.clustering import cluster_strips_by_page
    import hdbscan, umap
    print('  ✅ Clustering (HDBSCAN + UMAP)')
except ImportError:
    print('  ⚠️  Clustering (install hdbscan umap-learn)')

try:
    from src.evaluation.metrics import evaluate_reconstruction
    print('  ✅ Evaluation metrics')
except ImportError as e:
    errors.append(f'Evaluation: {e}')

if errors:
    print()
    for err in errors:
        print(f'  ❌ {err}')
    sys.exit(1)
else:
    print()
    print('  🎉 Core installation verified!')
"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   ✅ Setup complete!                             ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Next steps:"
echo -e "  ${CYAN}source .venv/bin/activate${NC}           # Activate the environment"
echo -e "  ${CYAN}make train-seam${NC}                     # Train the seam model"
echo -e "  ${CYAN}make train-embedder${NC}                 # Train the page embedder"
echo -e "  ${CYAN}make app${NC}                            # Launch the Streamlit app"
echo -e "  ${CYAN}make benchmark PDF=your_file.pdf${NC}    # Run benchmarks"
echo -e ""
echo -e "Or run the full pipeline:"
echo -e "  ${CYAN}make pipeline PDF=your_file.pdf${NC}"
echo ""
