# BOND Installation Guide

This guide provides detailed instructions for installing and configuring BOND.

## System Requirements

- **Python**: 3.11 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 8GB RAM minimum (16GB+ recommended)
- **Disk Space**: ~5GB for ontology database and FAISS indices
- **Optional**: GPU for faster embedding inference (CUDA-compatible)

## Step 1: Clone the Repository

```bash
git clone https://github.com/Aronow-Lab/BOND.git
cd BOND
```

## Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv bond_venv

# Activate (Linux/macOS)
source bond_venv/bin/activate

# Activate (Windows)
bond_venv\Scripts\activate
```

## Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install BOND package (editable mode)
pip install -e .
```

Or install with development dependencies:

```bash
pip install -e ".[dev]"
```

## Step 4: Obtain Ontology Database

You need an SQLite database containing ontology terms. Detailed information on how to create, update, and manage the ontology database can be found in [assets/README.md](assets/README.md).

You have two options:

### Option A: Use Pre-built Database

If you have access to a pre-built `ontologies.sqlite` file:

```bash
mkdir -p assets
cp /path/to/ontologies.sqlite assets/ontologies.sqlite
```

### Option B: Generate Database from Ontology Files

```bash
# Generate SQLite database from OBO/OWL files
bond-generate-sqlite \
  --input_dir /path/to/ontology/files \
  --output_path assets/ontologies.sqlite
```

The script supports:

- OBO format files (`.obo`)
- OWL format files (`.owl`)
- JSON-LD format

Required ontologies for full functionality:

- Cell Ontology (CL)
- UBERON
- MONDO Disease Ontology
- Experimental Factor Ontology (EFO)
- PATO
- HANCESTRO
- NCBI Taxonomy
- Organism-specific development stage ontologies (HsapDv, MmusDv, etc.)

## Step 4.5: Create Abbreviations Dictionary (Optional but Recommended)

Create or update the abbreviations dictionary at `assets/abbreviations.json` to improve query matching for abbreviated terms. This file is optional but recommended:

```bash
# Create abbreviations file
mkdir -p assets
cat > assets/abbreviations.json << 'EOF'
{
  "cell_type": {
    "t": "t cell",
    "nk": "natural killer cell",
    "dc": "dendritic cell",
    "b": "b cell",
    "mono": "monocyte",
    "mφ": "macrophage",
    "neu": "neutrophil"
  },
  "tissue": {
    "bm": "bone marrow",
    "ln": "lymph node",
    "spl": "spleen"
  }
}
EOF
```


## Step 5: Configure Embedding Model (Before Building FAISS)

Before building the FAISS index, you need to configure your embedding model. The FAISS index must be built with the same embedding model you'll use at runtime.

**Important**: Configure your embedding model in Step 6 (Environment Configuration) before building FAISS in Step 7.

See the [Selecting Your Encoder](#selecting-your-encoder-hf-or-ollama) section below for detailed options.

## Step 6: Build FAISS Index

Build the FAISS index for dense semantic search:

```bash
bond-build-faiss \
  --sqlite_path assets/ontologies.sqlite \
  --assets_path assets \
  --embed_model st:all-MiniLM-L6-v2
```

**Note**: This step requires:

- Embedding model configured in `.env` file (see Step 5 and Step 7)
- Several hours for large ontology databases
- Sufficient disk space (~2-5GB)

**Important**: Make sure you've configured your embedding model in the `.env` file (Step 7) before running this command, as the FAISS index must match your runtime embedding model.

## Step 7: Configure Environment

Create a `.env` file in the project root:

```bash
# Embedding Model Configuration
# Options:
# - st:all-MiniLM-L6-v2 (Sentence Transformers, default)
# - st:sentence-transformers/all-mpnet-base-v2
# - litellm/http://your-embedding-service

BOND_EMBED_MODEL=ollama/rajdeopankaj/bond-embed-v1-fp16:latest

# LLM Providers for Expansion and Disambiguation
# You need at least one configured

# Option 1: Anthropic Claude
BOND_EXPANSION_LLM=anthropic/claude-3-5-sonnet-20241022
BOND_DISAMBIGUATION_LLM=anthropic/claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=your-anthropic-api-key

# Option 2: OpenAI GPT
# BOND_EXPANSION_LLM=openai/gpt-4o
# BOND_DISAMBIGUATION_LLM=openai/gpt-4o
# OPENAI_API_KEY=your-openai-api-key

# Option 3: Other LiteLLM-compatible providers
# BOND_EXPANSION_LLM=cohere/command-r-plus
# BOND_DISAMBIGUATION_LLM=cohere/command-r-plus
# COHERE_API_KEY=your-cohere-api-key

# Paths (defaults shown)
BOND_ASSETS_PATH=assets
BOND_SQLITE_PATH=assets/ontologies.sqlite
BOND_RERANKER_PATH=reranker-model/

# Optional: Retrieval-only mode (skip LLM stages)
# BOND_RETRIEVAL_ONLY=1

# Optional: API Authentication
# BOND_API_KEY=your-secret-api-key
# BOND_ALLOW_ANON=1  # Allow anonymous access (development only)
```

## Step 8: Download Reranker Model (Optional)

The reranker model improves accuracy by 10-15%. It's optional but recommended:

1. **Download from Hugging Face**: https://huggingface.co/AronowLab/BOND-reranker
2. **Extract model files** to `reranker-model/` directory:
   ```bash
   mkdir -p reranker-model
   # Download and extract model files to reranker-model/
   ```
3. **Verify files**: The directory should contain:
   - `config.json`
   - `model.safetensors` (or `pytorch_model.bin`)
   - `tokenizer_config.json`
   - `vocab.txt`
   - Other tokenizer files

**Note**: The `BOND_RERANKER_PATH` in your `.env` file should point to this directory (default: `reranker-model/`). See [reranker-model/README.md](reranker-model/README.md) for detailed instructions.

## Step 9: Verify Installation

Verify that all components are properly installed:

### 1. Verify Assets

```bash
# Check SQLite database exists
ls -lh assets/ontologies.sqlite

# Check FAISS index exists
ls -lh assets/faiss_store/embeddings.faiss
ls -lh assets/faiss_store/id_map.npy

# Check abbreviations file (optional)
ls -lh assets/abbreviations.json

# Check reranker model (optional)
ls -lh reranker-model/config.json
```

### 2. Test CLI

```bash
# Check CLI works
bond-query --help

# Test query (requires database and FAISS index)
bond-query \
  --query "T-cell" \
  --field cell_type \
  --organism "Homo sapiens" \
  --tissue "blood"
```

### 3. Test Python API

```python
from bond import BondMatcher
from bond.config import BondSettings

# Initialize matcher
settings = BondSettings()
matcher = BondMatcher(settings)

# Test query
result = matcher.query(
    query="T-cell",
    field_name="cell_type",
    organism="Homo sapiens",
    tissue="blood"
)

print(f"Matched: {result['chosen']['label']}")
print(f"Ontology ID: {result['chosen']['id']}")
```

### 4. Test Server (Optional)

```bash
# Start server (if API key is set)
bond-serve
# In another terminal:
curl http://localhost:8000/health
```

## Docker Installation (Alternative)

A Dockerfile is provided for containerized deployment:

```bash
# Build image
docker build -t bond:latest .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/assets:/app/assets \
  -e BOND_API_KEY=your-key \
  -e ANTHROPIC_API_KEY=your-key \
  bond:latest
```

## Troubleshooting

### Issue: "Database not found: assets/ontologies.sqlite"

**Solution**: Ensure the ontology database exists at the specified path:

```bash
ls -lh assets/ontologies.sqlite
```

### Issue: "FAISS index not found"

**Solution**: Build the FAISS index:

```bash
bond-build-faiss --sqlite_path assets/ontologies.sqlite --assets_path assets
```

### Issue: LLM API errors

**Solutions**:

1. Verify API keys are set correctly
2. Check API key permissions (write access required)
3. Ensure sufficient API credits/quota
4. Try a different LLM provider

### Issue: Out of memory during FAISS build

**Solutions**:

1. Build index with smaller batch size
2. Use CPU-only FAISS (faiss-cpu) instead of GPU version
3. Process ontologies in chunks

### Issue: Import errors

**Solution**: Ensure virtual environment is activated and dependencies installed:

```bash
source bond_venv/bin/activate
pip install -e .
```

## Next Steps

- Read the [README.md](README.md) for usage examples
- Explore [Hybrid Search Guide](Miscellaneous/README_hybrid_search.md) for advanced features
- Review [Reranker Training Guide](reranker-model/README.md) for custom model training. See [notebooks/](notebooks/) for training code and example notebooks

## Getting Help

For questions and support, see the resources below.

## Additional Resources

- **Benchmark Dataset**: [HuggingFace Dataset](https://huggingface.co/datasets/AronowLab/bond-czi-benchmark)
- **Paper**: [Multi-agent AI System for High Quality Metadata Curation at Scale](https://www.biorxiv.org/content/10.1101/2025.06.10.658658v1) - Related multi-agent curation system
- **Issues**: [GitHub Issues](https://github.com/Aronow-Lab/BOND/issues)

## Selecting Your Encoder (HF or Ollama)

**Important**: Configure your embedding model before building the FAISS index (Step 6). The FAISS index must be built with the same embedding model you'll use at runtime.

You can use your published encoders with BOND.

### Option A: Ollama (local)

1) Pull the model:

```bash
ollama pull rajdeopankaj/bond-embed-v1-fp16
```

2) Set the env var (e.g., in `.env`):

```bash
BOND_EMBED_MODEL=ollama:rajdeopankaj/bond-embed-v1-fp16
# OLLAMA_API_BASE=http://localhost:11434  # if remote, set your host
```

3) Build FAISS:

```bash
bond-build-faiss --sqlite_path assets/ontologies.sqlite --assets_path assets
```

### Option B: Hugging Face TEI (hosted)

1) Deploy `pankajrajdeo/bond-embed-v1-fp16` behind a LiteLLM-compatible endpoint (e.g., TEI + gateway).
2) Set the env var to the routed model name, for example:

```bash
BOND_EMBED_MODEL=litellm:huggingface/teimodel
```

3) Build FAISS as usual.

References:

- HF model: https://huggingface.co/pankajrajdeo/bond-embed-v1-fp16
- Ollama model: https://ollama.com/rajdeopankaj/bond-embed-v1-fp16
