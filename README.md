# Forensic Document Reconstructor

A deep learning system for reconstructing shredded multi-page documents. Given a pile of randomized vertical strips from multiple pages, the system clusters strips by page and reconstructs each page's original strip order.

## 🏗️ Architecture

```
Input Strips ──→ SeamResNet ──→ Score Matrix (n×n)
     │                               │
     └──→ PageEmbeddingNet ──→ UMAP + HDBSCAN ──→ Page Clusters
                                                       │
                                              ┌────────┴─────────┐
                                              │   Per-cluster    │
                                              │   ATSP Solver    │
                                              └────────┬─────────┘
                                                       │
                                              Reconstructed Pages
```

**Two models, three solver strategies:**
- **SeamResNet** — Binary classifier: "does strip B go right of strip A?" (ResNet18)
- **PageEmbeddingNet** — Contrastive embeddings: "which page does this strip belong to?" (ResNet18 + SupCon)
- **Solvers**: Greedy (baseline), ATSP (optimized), HDBSCAN+ATSP (thesis pipeline)

## 🚀 Quick Start

### One-Command Setup

```bash
git clone <repo-url> && cd Compvis_Thesis
chmod +x setup.sh && ./setup.sh
```

This auto-detects your GPU, creates a virtual environment, installs all dependencies, and verifies the installation.

### Full Pipeline (Train → Benchmark → App)

```bash
make pipeline PDF=path/to/test.pdf
```

Or step by step:

```bash
source .venv/bin/activate

# 1. Train the seam model (with cross-document negatives)
make train-seam

# 2. Train the page embedder (contrastive learning)
make train-embedder

# 3. Benchmark all strategies against each other
make benchmark PDF=data/206-10001-10017.pdf

# 4. Launch the interactive app
make app
```

### Quick Test Run

```bash
make pipeline PDF=data/206-10001-10017.pdf QUICK=true   # 2 epochs, fast
```

## 📋 Available Commands

| Command | Description |
|---|---|
| `make setup` | Create venv & install all dependencies |
| `make verify` | Verify installation works |
| `make train-seam` | Train seam compatibility model |
| `make train-embedder` | Train page embedding model |
| `make train-all` | Train both models sequentially |
| `make app` | Launch Streamlit web app |
| `make benchmark` | Run full benchmark suite |
| `make evaluate` | Quick single evaluation |
| `make shred PDF=file.pdf` | Shred a document into test strips |
| `make test` | Run smoke tests |
| `make clean` | Remove generated files |

**Override any default:**
```bash
make train-seam EPOCHS=20 BATCH_SIZE=128 LR=5e-5
make benchmark PDF=my_doc.pdf NUM_PAGES="2 5 10" STRATEGIES="greedy atsp"
```

## 📁 Project Structure

```
Compvis_Thesis/
├── src/
│   ├── models/
│   │   ├── seam_model.py          # SeamResNet — strip adjacency classifier
│   │   └── page_embedder.py       # PageEmbeddingNet — contrastive page embeddings
│   ├── data/
│   │   ├── preprocessing.py       # Shared preprocessing pipeline
│   │   └── shredder.py            # PDF/image → randomized strips
│   ├── training/
│   │   ├── dataset.py             # StreamingShredDataset (40/20/15/25% pair distribution)
│   │   ├── train_seam.py          # SeamResNet trainer (argparse, CSV logging)
│   │   └── train_page_embedder.py # PageEmbedder contrastive trainer
│   ├── solver/
│   │   ├── scoring.py             # n×n pairwise score matrix computation
│   │   ├── greedy.py              # Baseline greedy chain builder
│   │   ├── atsp.py                # ATSP solver (simulated annealing)
│   │   └── clustering.py          # HDBSCAN clustering + block detection
│   ├── evaluation/
│   │   └── metrics.py             # ARI, NMI, pairwise accuracy, Kendall's τ
│   └── app/
│       └── app.py                 # Streamlit web interface
├── scripts/
│   ├── benchmark.py               # Multi-strategy benchmark runner
│   └── run_pipeline.sh            # End-to-end pipeline script
├── data/                          # PDFs, images, model weights (gitignored)
├── legacy/                        # Original monolithic scripts (reference)
├── setup.sh                       # One-shot environment setup
├── Makefile                       # Turnkey command runner
└── requirements.txt               # Python dependencies
```

## 🧪 Training Details

### SeamResNet (Strip Adjacency)

Trained on `chainyo/rvl-cdip` (streamed from HuggingFace) with four pair types:

| Pair Type | Ratio | Description |
|---|---|---|
| Positive | 40% | Adjacent strips from the same image |
| Hard Negative | 20% | Adjacent strips with vertical shift (±10-30px) |
| Easy Negative | 15% | Non-adjacent strips from the same image |
| **Cross-Document** | **25%** | Strips from **different images** in the buffer |

The cross-document negatives are the key thesis improvement — without them, the model can't distinguish strips from different pages.

### PageEmbeddingNet (Page Identity)

Trained with Supervised Contrastive Loss (SupCon). Each batch samples P pages × S strips, where all strips from the same page share a label. The loss pulls same-page embeddings together and pushes different-page embeddings apart.

## 📊 Benchmark Strategies

| Strategy | Description |
|---|---|
| `greedy` | Baseline: greedily chain highest-scoring neighbors |
| `atsp` | ATSP solver on the full strip pool (no clustering) |
| `hdbscan+greedy` | Cluster by page first, then greedy within clusters |
| `hdbscan+atsp` | Cluster by page first, then ATSP within clusters |

## 🔧 Advanced Usage

### Pipeline Script

The `scripts/run_pipeline.sh` script runs the full experiment with fine-grained control:

```bash
./scripts/run_pipeline.sh --help

# Examples:
./scripts/run_pipeline.sh --pdf my_doc.pdf --seam-epochs 20
./scripts/run_pipeline.sh --skip-training           # Benchmark existing models
./scripts/run_pipeline.sh --seam-only               # Skip embedder training
./scripts/run_pipeline.sh --quick                   # 2 epochs, 500 steps
```

### Using Existing Models

If you have pre-trained models (e.g., `best_seam_model_v2.pth` from the original project):

```bash
# Copy to expected location
cp best_seam_model_v2.pth checkpoints/seam/best_seam_model.pth

# Benchmark with existing model (skip training)
make benchmark PDF=data/206-10001-10017.pdf

# Or override the model path directly
make benchmark SEAM_MODEL=path/to/your/model.pth
```

### Custom Training Configurations

```bash
# Long training run with higher resolution
make train-seam EPOCHS=50 BATCH_SIZE=128 LR=5e-5 STEPS_PER_EPOCH=5000

# Resume interrupted training
make train-seam-resume

# More pages per batch for better contrastive learning
make train-embedder EMB_PAGES_PER_BATCH=16 EMB_STRIPS_PER_PAGE=6 EMB_EPOCHS=30
```

## 📝 Requirements

- Python 3.10+
- CUDA GPU recommended (CPU works but is slow)
- `poppler-utils` for PDF processing

## License

[GPL v3](LICENSE)
