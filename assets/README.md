# BOND Assets Directory

This directory contains all the pre-computed data structures required for the BOND (Biomedical Ontology Normalization and Disambiguation) pipeline to function. This README explains the history, structure, usage, and how to reproduce these assets with your own data.

## 📁 Directory Structure

```
assets/
├── ontologies.sqlite          # SQLite database containing ontology terms, synonyms, definitions, and relationships
├── ontology.db                # Legacy database file (if present)
├── abbreviations.json         # Field-specific abbreviation mappings for query normalization
└── faiss_store/              # FAISS index for dense semantic search
    ├── embeddings.faiss       # Binary FAISS index (compressed embeddings)
    ├── id_map.npy            # Mapping from FAISS index positions to ontology CURIE IDs
    ├── rescore_vectors.npy   # Int8 quantized vectors for precise rescoring
    ├── embedding_signature.json  # Metadata about the embedding model used
    └── meta.json             # Index metadata (creation date, statistics, etc.)
```

## 🎯 Purpose and History

### Background

The BOND assets directory was created to provide a **pre-computed, optimized knowledge base** for biomedical ontology normalization. The system was designed to handle the challenge of mapping free-text biological terms (e.g., "T-cell", "lymphocyte") to standardized ontology identifiers (e.g., `CL:0000084`) across multiple ontologies.

### Why These Assets Are Needed

1. **Performance**: Pre-computed indices enable fast retrieval (milliseconds) instead of real-time computation (seconds/minutes)
2. **Consistency**: All components use the same ontology data, ensuring consistent results
3. **Scalability**: Optimized data structures (FAISS, SQLite FTS) handle millions of ontology terms efficiently
4. **Reproducibility**: Same assets produce same results across different runs

### Creation History

The assets were originally created through a multi-step process:

1. **Ontology Collection**: Ontology files (OBO/OWL format) were collected from various sources:

   - Cell Ontology (CL) from OBO Foundry
   - UBERON (anatomy) from OBO Foundry
   - MONDO (diseases) from OBO Foundry
   - EFO, PATO, HANCESTRO, and organism-specific ontologies
2. **SQLite Database Generation**: The `scripts/sqlite_generator.py` script processed raw ontology JSON files to create a normalized SQLite database with:

   - Terms, synonyms, definitions
   - Hierarchical relationships (parent/child)
   - Cross-references and metadata
   - Full-text search (FTS) indices for BM25 retrieval
3. **FAISS Index Building**: The `scripts/build_faiss_from_sqlite.py` script:

   - Generated embeddings for all ontology terms using a biomedical embedding model
   - Created a binary FAISS index for fast approximate nearest neighbor search
   - Quantized vectors to int8 for memory-efficient rescoring
4. **Abbreviation Dictionary**: Manually curated field-specific abbreviations (e.g., "T" → "T cell", "NK" → "natural killer cell") to improve query matching

## 🔧 How Assets Are Used in BOND Pipeline

### 1. SQLite Database (`ontologies.sqlite`)

**Purpose**: Stores all ontology terms, synonyms, definitions, and relationships.

**Usage in Pipeline**:

- **Exact Matching**: Fast lookup of exact term matches using SQL queries
- **BM25 Retrieval**: Full-text search using SQLite FTS5 for keyword-based retrieval
- **Metadata Access**: Retrieving term definitions, synonyms, and relationships for LLM disambiguation
- **Ontology Filtering**: Filtering results by ontology namespace (CL, UBERON, etc.)

**Key Tables**:

- `ontology_terms`: Main table with term labels, CURIE IDs, definitions
- `ontology_terms_fts`: Full-text search index (FTS5) for BM25 retrieval
- `ontology_edges`: Hierarchical relationships (parent/child) for graph expansion
- `term_synonym`: Structured synonyms (exact, narrow, broad, related)

**Code Location**: `bond/pipeline.py` lines 70-98 (connection setup), throughout for queries

### 2. FAISS Index (`faiss_store/`)

**Purpose**: Enables fast dense semantic search over millions of ontology terms.

**Usage in Pipeline**:

- **Dense Retrieval**: Semantic similarity search using query embeddings
- **Two-Stage Search**:
  1. Fast binary search retrieves `k * rescore_multiplier` candidates
  2. Precise int8 rescoring selects top-k results
- **Hybrid Search**: Combined with exact and BM25 results via Reciprocal Rank Fusion (RRF)

**Technical Details**:

- **Binary Index**: Compressed embeddings (1 bit per dimension) for fast approximate search
- **Int8 Rescoring**: Quantized vectors (8 bits) for precise similarity computation
- **Memory-Mapped**: Large arrays (`id_map.npy`, `rescore_vectors.npy`) are memory-mapped to avoid loading into RAM

**Code Location**: `bond/retrieval/faiss_store.py`, `bond/pipeline.py` lines 104-105, 381-411

### 3. Abbreviations Dictionary (`abbreviations.json`)

**Purpose**: Field-specific abbreviation expansion to improve query matching.

**Usage in Pipeline**:

- **Query Normalization**: Expands abbreviations before retrieval (e.g., "T" → "T cell")
- **Field-Aware**: Different expansions for different field types (cell_type, tissue, etc.)
- **Pattern Matching**: Uses regex with word boundaries to avoid over-expansion

**Format**:

```json
{
  "cell_type": {
    "t": "t cell",
    "nk": "natural killer cell",
    "dc": "dendritic cell"
  },
  "tissue": {
    "bm": "bone marrow",
    "ln": "lymph node"
  }
}
```

**Code Location**: `bond/abbrev.py`, `bond/pipeline.py` line 128

## 📊 Asset Statistics

Typical asset sizes for a full ontology database:

- **ontologies.sqlite**: ~500MB - 2GB (depends on number of ontologies)
- **embeddings.faiss**: ~100MB - 500MB (binary compressed)
- **id_map.npy**: ~50MB - 200MB (depends on number of terms)
- **rescore_vectors.npy**: ~200MB - 1GB (int8 quantized, depends on embedding dimension)
- **abbreviations.json**: ~10KB - 50KB

**Total Size**: ~1GB - 4GB for a complete setup

## 🛠️ How to Reproduce Assets with Your Own Data

### Prerequisites

1. **Python 3.11+** with BOND installed
2. **Ontology Files**: OBO or OWL format files for your ontologies
3. **Embedding Model**: Access to an embedding model (local or API)
4. **Disk Space**: 5-10GB free space
5. **Time**: 2-6 hours depending on ontology size and hardware

### Step 1: Prepare Ontology Files

Collect ontology files in OBO or OWL format. Common sources:

- **OBO Foundry**: https://obofoundry.org/
- **BioPortal**: https://bioportal.bioontology.org/
- **Ontology Lookup Service**: https://www.ebi.ac.uk/ols/

**Required Ontologies for Full BOND Functionality**:

- Cell Ontology (CL)
- UBERON (anatomy)
- MONDO (diseases)
- Experimental Factor Ontology (EFO)
- PATO (phenotypic quality)
- HANCESTRO (ancestry/ethnicity)
- NCBI Taxonomy
- Organism-specific development stage ontologies (HsapDv, MmusDv, FBdv, etc.)

**Optional but Recommended**:

- Gene Ontology (GO)
- Disease Ontology (DOID)
- Other domain-specific ontologies

### Step 2: Generate SQLite Database

#### Option A: From OBO/OWL Files (Recommended)

If you have OBO or OWL files:

```bash
# Create assets directory
mkdir -p assets

# Generate SQLite database
bond-generate-sqlite \
  --input_dir /path/to/ontology/files \
  --output_path assets/ontologies.sqlite
```

#### Option B: From JSON Format (Original Method)

If you have ontology data in JSON format (like the original `filtered_ontologies.json`):

```bash
# Use the ontology SQLite generator script
python scripts/ontology_sqlite_generator.py \
  --json_file /path/to/filtered_ontologies.json \
  --output_path assets/ontologies.sqlite
```

**What This Creates**:

- `ontology_terms` table: All terms with labels, CURIE IDs, definitions
- `ontology_terms_fts` table: Full-text search index for BM25
- `ontology_edges` table: Hierarchical relationships
- `term_synonym` table: Structured synonyms
- Additional metadata tables

**Verification**:

```bash
# Check database was created
ls -lh assets/ontologies.sqlite

# Verify tables exist
sqlite3 assets/ontologies.sqlite ".tables"

# Check term count
sqlite3 assets/ontologies.sqlite "SELECT COUNT(*) FROM ontology_terms;"
```

### Step 3: Build FAISS Index

Build the FAISS index for dense semantic search:

```bash
# Set embedding model (choose one)
export BOND_EMBED_MODEL="st:all-MiniLM-L6-v2"  # Sentence Transformers (local)
# OR
export BOND_EMBED_MODEL="ollama:rajdeopankaj/bond-embed-v1-fp16"  # Ollama (local)
# OR
export BOND_EMBED_MODEL="litellm:http://your-embedding-service"  # API endpoint

# Optional: Set batch size for embedding generation
export BOND_EMB_BATCH=16  # Increase for faster processing (requires more memory)

# Build FAISS index
bond-build-faiss \
  --sqlite_path assets/ontologies.sqlite \
  --assets_path assets \
  --embed_model "$BOND_EMBED_MODEL"
```

**What This Creates**:

- `faiss_store/embeddings.faiss`: Binary FAISS index
- `faiss_store/id_map.npy`: Mapping from index positions to CURIE IDs
- `faiss_store/rescore_vectors.npy`: Int8 quantized vectors for rescoring
- `faiss_store/embedding_signature.json`: Metadata about embedding model
- `faiss_store/meta.json`: Index statistics

**Process Details**:

1. Reads all terms from `ontology_terms` table (skips obsolete terms)
2. Formats text as: `"label: {LABEL}; synonyms: {SYNONYMS}; definition: {DEFINITION}"`
3. Generates embeddings in batches using the specified model
4. L2-normalizes vectors
5. Creates binary FAISS index (1 bit per dimension for fast search)
6. Quantizes to int8 for memory-efficient rescoring
7. Saves all components to `faiss_store/`

**Time Estimate**:

- Small ontologies (<100K terms): 30 minutes - 1 hour
- Medium ontologies (100K-500K terms): 1-3 hours
- Large ontologies (>500K terms): 3-6 hours

**Verification**:

```bash
# Check FAISS store was created
ls -lh assets/faiss_store/

# Verify files exist
ls assets/faiss_store/embeddings.faiss
ls assets/faiss_store/id_map.npy
ls assets/faiss_store/rescore_vectors.npy
ls assets/faiss_store/embedding_signature.json
```

### Step 4: Create Abbreviations Dictionary

Create or customize `abbreviations.json` for your use case:

```bash
# Create abbreviations file
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
  },
  "global": {
    "hsc": "hematopoietic stem cell",
    "esc": "embryonic stem cell"
  }
}
EOF
```

**Format Guidelines**:

- **Field-specific sections**: Use field names as keys (e.g., `"cell_type"`, `"tissue"`)
- **Global section**: Use `"global"` for abbreviations that apply across all fields
- **Values**: Can be a string or list (first element is used)
- **Case-insensitive**: Matching is case-insensitive

**Best Practices**:

- Start with common abbreviations in your domain
- Add abbreviations based on query analysis (what terms are users searching for?)
- Test expansions to avoid over-matching (e.g., "T" should match "T cell" but not "T-shirt")

### Step 5: Verify Assets

Test that all assets are working:

```python
from bond import BondMatcher
from bond.config import BondSettings

# Initialize BOND with your assets
settings = BondSettings(
    assets_path="assets",
    sqlite_path="assets/ontologies.sqlite"
)

matcher = BondMatcher(settings=settings)

# Test query
result = matcher.query(
    query="T-cell",
    field_name="cell_type",
    organism="Homo sapiens",
    tissue="blood"
)

print(f"Matched: {result['chosen']['label']}")
print(f"ID: {result['chosen']['id']}")
```

## 🔄 Updating Assets

### Updating SQLite Database

If you add new ontologies or update existing ones:

```bash
# Regenerate SQLite database
bond-generate-sqlite \
  --input_dir /path/to/updated/ontology/files \
  --output_path assets/ontologies.sqlite

# Then rebuild FAISS index (required after SQLite changes)
bond-build-faiss \
  --sqlite_path assets/ontologies.sqlite \
  --assets_path assets \
  --embed_model "$BOND_EMBED_MODEL" \
  --faiss_rebuild  # Force full rebuild
```

### Updating FAISS Index

If you change the embedding model or want to rebuild:

```bash
# Rebuild with new model
export BOND_EMBED_MODEL="new-model-name"

bond-build-faiss \
  --sqlite_path assets/ontologies.sqlite \
  --assets_path assets \
  --embed_model "$BOND_EMBED_MODEL" \
  --faiss_rebuild
```

**Note**: The script automatically detects model changes by comparing `embedding_signature.json` and will rebuild if needed.

### Updating Abbreviations

Simply edit `assets/abbreviations.json` - no rebuild needed. Changes take effect on next BOND initialization.

## 🐛 Troubleshooting

### Issue: "Database not found: assets/ontologies.sqlite"

**Solution**: Ensure the SQLite database exists:

```bash
ls -lh assets/ontologies.sqlite
```

If missing, generate it using Step 2 above.

### Issue: "FAISS store directory not found"

**Solution**: Build the FAISS index:

```bash
bond-build-faiss --sqlite_path assets/ontologies.sqlite --assets_path assets
```

### Issue: "Embedding signature mismatch"

**Solution**: The FAISS index was built with a different embedding model. Rebuild:

```bash
bond-build-faiss \
  --sqlite_path assets/ontologies.sqlite \
  --assets_path assets \
  --embed_model "$BOND_EMBED_MODEL" \
  --faiss_rebuild
```

### Issue: Out of memory during FAISS build

**Solutions**:

1. Reduce batch size: `export BOND_EMB_BATCH=8`
2. Process in smaller chunks (modify script if needed)
3. Use CPU-only FAISS: `pip install faiss-cpu` instead of `faiss-gpu`

### Issue: Slow FAISS build

**Solutions**:

1. Use GPU acceleration (if available): `pip install faiss-gpu`
2. Increase batch size: `export BOND_EMB_BATCH=32` (if memory allows)
3. Use a faster embedding model (smaller models are faster)
4. Use local embedding model instead of API (avoids network latency)

### Issue: Abbreviations not working

**Solutions**:

1. Check file format: Must be valid JSON
2. Verify file path: Should be `assets/abbreviations.json`
3. Check field names: Must match field names used in queries (case-insensitive)
4. Test pattern: Abbreviations use regex with word boundaries - ensure your patterns are correct

## 📚 Additional Resources

- **BOND Installation Guide**: See [../INSTALLATION.md](../INSTALLATION.md)
- **FAISS Documentation**: https://github.com/facebookresearch/faiss
- **SQLite FTS5**: https://www.sqlite.org/fts5.html
- **Ontology Sources**:
  - OBO Foundry: https://obofoundry.org/
  - BioPortal: https://bioportal.bioontology.org/
  - OLS: https://www.ebi.ac.uk/ols/

## 🔐 Security and Privacy

- **SQLite Database**: Contains public ontology data (no sensitive information)
- **FAISS Index**: Contains embeddings of ontology terms (no user data)
- **Abbreviations**: Public domain abbreviations (no sensitive information)

All assets are safe to share and version control (except for very large files, which should use Git LFS).

## 📝 Summary

The `assets/` directory is the **knowledge base** of BOND, containing:

1. **ontologies.sqlite**: Structured ontology data for exact matching and BM25 retrieval
2. **faiss_store/**: Pre-computed embeddings for dense semantic search
3. **abbreviations.json**: Field-specific abbreviation expansions

Together, these assets enable BOND's hybrid retrieval system (Exact + BM25 + Dense) to efficiently search millions of ontology terms and map free-text queries to standardized identifiers.

**To get started**: Follow Steps 1-5 above to create your own assets, or use pre-built assets if available.
