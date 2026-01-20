# BOND: Biomedical Ontology Normalization and Disambiguation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![GitHub](https://img.shields.io/github/stars/Aronow-Lab/BOND?style=social)](https://github.com/Aronow-Lab/BOND)
[![HuggingFace Dataset](https://img.shields.io/badge/dataset-huggingface-blue)](https://huggingface.co/datasets/AronowLab/bond-czi-benchmark)
[![HuggingFace Model](https://img.shields.io/badge/model-huggingface-blue)](https://huggingface.co/pankajrajdeo/bond-embed-v1-fp16)
[![HuggingFace Reranker](https://img.shields.io/badge/reranker-huggingface-blue)](https://huggingface.co/AronowLab/BOND-reranker)

BOND is a system for mapping free-text biological terms to standardized ontology identifiers. It combines hybrid retrieval (exact matching, BM25, dense embeddings), reciprocal rank fusion, cross-encoder reranker, graph-based expansion, and LLM-powered disambiguation to achieve high-accuracy ontology normalization for biomedical metadata.

## 📓 Example Notebooks

- **Reranker Trained Notebook**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1USKLEtyWmbvzOM3ZlkUYDg3EUcMRC1MQ#scrollTo=QRdu0nhobRLA) - Fine-tuned BOND reranker model
- **Train Your Own Reranker**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Qq0FolMY8GfLdxXnfvgdPJnx1rCzUgUy?usp=sharing) - Complete guide with dataset structure, configuration, and training instructions

## 🎯 Overview

BOND addresses the critical challenge of harmonizing diverse biological terminology across research studies and datasets. When researchers submit data to public repositories like GEO, ArrayExpress, or CELLxGENE, they often use inconsistent or non-standard terminology. BOND automatically maps these "author terms" to standardized ontology identifiers, enabling:

- **Metadata harmonization** across datasets
- **Semantic interoperability** for cross-study analysis
- **FAIR compliance** through ontology-linked metadata
- **Scalable curation** of large biomedical repositories

## ✨ Key Features

- **Hybrid Search Architecture**: Combines exact matching, BM25 keyword search, and dense semantic search (FAISS) for comprehensive retrieval
- **Reciprocal Rank Fusion (RRF)**: Intelligently combines results from multiple retrieval methods
- **Cross-Encoder Reranker**: Fine-tuned biomedical reranker model improves accuracy by 10-15% by re-ranking candidates
- **LLM-Powered Expansion**: Uses large language models to generate query expansions and context-aware synonyms
- **Graph-Based Expansion**: Leverages ontology hierarchies to discover related terms
- **Context-Aware Disambiguation**: LLM reasoning to select the correct ontology ID from candidate matches
- **Multi-Ontology Support**: Works with Cell Ontology (CL), UBERON, MONDO, EFO, PATO, HANCESTRO, and more
- **Organism-Aware Routing**: Automatically selects appropriate ontologies based on organism and field type
- **RESTful API**: FastAPI-based service for easy integration
- **CLI Tool**: Command-line interface for batch processing

## 📊 Supported Fields and Ontologies

### Supported Fields

- `cell_type`: Cell types and classifications (Cell Ontology)
- `tissue`: Anatomical structures (UBERON)
- `disease`: Disease conditions (MONDO)
- `development_stage`: Developmental stages (organism-specific)
- `sex`: Biological sex (PATO)
- `self_reported_ethnicity`: Ethnicity/ancestry (HANCESTRO)
- `assay`: Experimental methods (EFO)
- `organism`: Taxonomic classification (NCBI Taxonomy)

### Supported Organisms

- `Homo sapiens`
- `Mus musculus`
- `Danio rerio` (zebrafish)
- `Drosophila melanogaster` (fruit fly)
- `Caenorhabditis elegans` (C. elegans)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Aronow-Lab/BOND.git
cd BOND

# Create and activate virtual environment
python3.11 -m venv bond_venv
source bond_venv/bin/activate  # On Windows: bond_venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -e .
```

### Prerequisites

1. **Ontology Database**: You need an SQLite database containing ontology terms. See [Installation Guide](INSTALLATION.md) for details.
2. **FAISS Index**: Build a FAISS index for dense semantic search:
   ```bash
   bond-build-faiss --sqlite_path assets/ontologies.sqlite --assets_path assets
   ```
3. **Environment Variables**: Configure LLM providers and embedding models (see Configuration below)

### Basic Usage

#### CLI Example

```bash
bond-query \
  --query "T-cell" \
  --field cell_type \
  --organism "Homo sapiens" \
  --tissue "blood" \
  --verbose
```

#### Python API Example

```python
from bond import BondMatcher
from bond.config import BondSettings

# Initialize matcher
settings = BondSettings()
matcher = BondMatcher(settings)

# Query
result = matcher.query(
    query="T-cell",
    field_name="cell_type",
    organism="Homo sapiens",
    tissue="blood"
)

print(f"Matched: {result['chosen']['label']}")
print(f"Ontology ID: {result['chosen']['id']}")
print(f"Confidence: {result['chosen']['llm_confidence']}")
```

#### API Server

```bash
# Start server (with anonymous access for development)
BOND_ALLOW_ANON=1 bond-serve

# Or with API key authentication
export BOND_API_KEY=your-secret-key
bond-serve
```

Then query the API:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "query": "T-cell",
    "field_name": "cell_type",
    "organism": "Homo sapiens",
    "tissue": "blood"
  }'
```

## 🔧 Configuration

BOND uses environment variables for configuration. Create a `.env` file:

```bash
# Embedding Model (choose ONE of the options below)

# Option A: Use your Ollama encoder (recommended for local)
# Make sure the model is available in Ollama first:
#   ollama pull rajdeopankaj/bond-embed-v1-fp16
BOND_EMBED_MODEL=ollama:rajdeopankaj/bond-embed-v1-fp16
# If running remote Ollama, set the API base (default: http://localhost:11434)
# OLLAMA_API_BASE=http://your-ollama-host:11434

# Option B: Use a LiteLLM-compatible hosted embedding endpoint
# Example: OpenAI, Azure OpenAI, Together, Groq, etc.
# BOND_EMBED_MODEL=litellm:text-embedding-3-small

# Option C: Use Hugging Face TEI (Text Embeddings Inference)
# Deploy your HF model (pankajrajdeo/bond-embed-v1-fp16) behind a LiteLLM-compatible endpoint,
# then set the model name accordingly.
# Example if exposed as huggingface/teimodel (via LiteLLM routing):
# BOND_EMBED_MODEL=litellm:huggingface/teimodel

# LLM Providers (for expansion and disambiguation)
BOND_EXPANSION_LLM=anthropic/claude-3-5-sonnet-20241022
BOND_DISAMBIGUATION_LLM=anthropic/claude-3-5-sonnet-20241022


# Or use OpenAI
# BOND_EXPANSION_LLM=openai/gpt-4o
# BOND_DISAMBIGUATION_LLM=openai/gpt-4o

# API Keys (set as environment variables or in .env)
ANTHROPIC_API_KEY=your-key
OPENAI_API_KEY=your-key

# Paths
BOND_ASSETS_PATH=assets
BOND_SQLITE_PATH=assets/ontologies.sqlite

# Reranker Configuration
# Path to trained reranker model (cross-encoder)
# Default: Uses pre-trained model from reranker-model/ directory
# To use custom reranker: BOND_RERANKER_PATH=/path/to/your/reranker-model
# To disable reranker: BOND_ENABLE_RERANKER=0
BOND_RERANKER_PATH=./reranker-model
BOND_ENABLE_RERANKER=1

# Retrieval Parameters
BOND_TOPK_EXACT=5
BOND_TOPK_BM25=20
BOND_TOPK_DENSE=50
BOND_TOPK_FINAL=20

# Optional: Disable LLM stages for retrieval-only mode
# BOND_RETRIEVAL_ONLY=1
```

### Using Published Encoders

- Hugging Face model: `pankajrajdeo/bond-embed-v1-fp16`See model card and direct usage: https://huggingface.co/pankajrajdeo/bond-embed-v1-fp16
- Ollama model: `rajdeopankaj/bond-embed-v1-fp16`
  Pull first, then set `BOND_EMBED_MODEL=ollama:rajdeopankaj/bond-embed-v1-fp16` and (optionally) `OLLAMA_API_BASE`.

> Note: BOND’s embedding provider currently supports LiteLLM-style endpoints and Ollama natively. If you prefer running the Hugging Face model locally with SentenceTransformers, use it to precompute embeddings for your own pipelines; for BOND’s FAISS build and runtime embedding, route the HF model via a LiteLLM-compatible endpoint (e.g., TEI behind a gateway) or use the Ollama variant.

## 📖 Documentation

- [Installation Guide](INSTALLATION.md) - Detailed setup instructions
- [Hybrid Search Guide](Miscellaneous/README_hybrid_search.md) - Advanced search features
- [Reranker Training Guide](reranker-model/README.md) - Training custom rerankers. See [notebooks/](notebooks/) for training code and example notebooks
- [Benchmark Dataset](https://huggingface.co/datasets/AronowLab/bond-czi-benchmark) - Context-aware benchmark for biomedical entity normalization

## 🏗️ Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────┐
│ STAGE 1: QUERY EXPANSION                        │
├─────────────────────────────────────────────────┤
│ LLM generates synonyms, abbreviations,         │
│ and context-aware expansions                   │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ STAGE 2: HYBRID RETRIEVAL                      │
├─────────────────────────────────────────────────┤
│ • Exact Match (SQLite FTS) → Top-K             │
│ • BM25 Search (SQLite FTS) → Top-K             │
│ • Dense Search (FAISS) → Top-K                │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ STAGE 3: RECIPROCAL RANK FUSION                │
├─────────────────────────────────────────────────┤
│ Combine rankings from all methods using RRF    │
│ with field-aware weighting                     │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ STAGE 4: CROSS-ENCODER RERANKER                │
├─────────────────────────────────────────────────┤
│ Re-rank top candidates using fine-tuned        │
│ biomedical reranker model (bioformer-16L)      │
│ Improves accuracy by 10-15%                    │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ STAGE 5: GRAPH EXPANSION (Optional)            │
├─────────────────────────────────────────────────┤
│ Expand ontology neighbors if confidence low    │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ STAGE 6: LLM DISAMBIGUATION                    │
├─────────────────────────────────────────────────┤
│ LLM selects best match from candidates with    │
│ reasoning and confidence scoring                │
└─────────────────────────────────────────────────┘
```

### Key Components

- **`bond/pipeline.py`**: Core `BondMatcher` class implementing the full pipeline
- **`bond/retrieval/`**: Retrieval modules (BM25, FAISS)
- **`bond/fusion.py`**: Reciprocal rank fusion implementation
- **`bond/rerank.py`**: Cross-encoder reranker integration (loads and applies fine-tuned reranker model)
- **`bond/graph_utils.py`**: Ontology graph traversal
- **`bond/server.py`**: FastAPI REST service
- **`bond/cli.py`**: Command-line interface

## 📊 Benchmark Dataset

BOND includes a comprehensive, context-aware benchmark dataset derived from CELLxGENE metadata:

- **25,416 normalization pairs** across 7 field types (cell_type, tissue, disease, assay, sex, development_stage, self_reported_ethnicity)
- **85 high-quality datasets** from CELLxGENE Census (72 single-cell, 13 spatial)
- **186 unique tissues** across _Homo sapiens_ and _Mus musculus_
- **Stratified 80/10/10 splits**: 20,332 train / 2,542 validation / 2,542 test

Unlike traditional string-matching benchmarks, BOND requires **reasoning over biological context**—including tissue, disease, organism, and development stage information—to accurately map author-provided annotations to standardized ontology labels.

**Dataset**: [AronowLab/bond-czi-benchmark](https://huggingface.co/datasets/AronowLab/bond-czi-benchmark)

### Usage

```python
from datasets import load_dataset

dataset = load_dataset("AronowLab/bond-czi-benchmark")

# Access splits
train = dataset["train"]
val = dataset["validation"]
test = dataset["test"]

# Example
sample = train[0]
print(f"Input: {sample['author_term']}")
print(f"Context: {sample['tissue']}, {sample['disease']}")
print(f"Target: {sample['ontology_label']} ({sample['ontology_id']})")
```

## 🔬 Evaluation

BOND has been evaluated on the BOND-CZI benchmark dataset (25,416 pairs from 85 datasets). Performance highlights:

- **Embedding Model**: Custom fine-tuned encoder (`bond-embed-v1-fp16`) achieves **92.7% accuracy@10** on ontology evaluation
- **Hybrid Retrieval**: Combines exact match, BM25, and dense semantic search with Reciprocal Rank Fusion
- **Cross-Encoder Reranker**: Fine-tuned biomedical reranker (`bioformers/bioformer-16L`) improves Hit@10 accuracy from ~75-80% (retrieval only) to **85-90%** (with reranker)
- **LLM Disambiguation**: Uses large language models for context-aware term selection
- **Multi-field Support**: Handles 7 biological field types (cell_type, tissue, disease, etc.) across 5 organisms

See the [benchmark-metrics/README.md](benchmark-metrics/README.md) for evaluation results and scripts.

## 🔄 Current Work

Active development and research efforts:

- **Harmonized Knowledge Graph**: Building a harmonized transcriptomic knowledge graph from BOND-normalized metadata to enable advanced querying, cross-dataset analysis, and relationship discovery in single-cell transcriptomics data.

## 🛠️ Development

### Running Tests

```bash
make test
# or
pytest
```

### Code Quality

```bash
make lint
# or
ruff check .
black .
```

### Building FAISS Index

```bash
bond-build-faiss \
  --sqlite_path assets/ontologies.sqlite \
  --assets_path assets
```

### Generating Ontology Database

```bash
bond-generate-sqlite
```

## 📝 Citation

If you use BOND in your research, please cite:

**BOND System:**

```bibtex
@software{bond_2026,
  title={BOND: Biomedical Ontology Normalization and Disambiguation},
  author={Rajdeo, Pankaj and Gelal, Rupesh},
  year={2026},
  url={https://github.com/Aronow-Lab/BOND}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

- **Repository**: [github.com/Aronow-Lab/BOND](https://github.com/Aronow-Lab/BOND)
- **Benchmark Dataset**: [huggingface.co/datasets/AronowLab/bond-czi-benchmark](https://huggingface.co/datasets/AronowLab/bond-czi-benchmark)
- **Reranker Model**: [huggingface.co/AronowLab/BOND-reranker](https://huggingface.co/AronowLab/BOND-reranker)
- **Authors**: [Pankaj Rajdeo](https://github.com/pankajrajdeo), [Rupesh Gelal](https://github.com/rgrupesh)
