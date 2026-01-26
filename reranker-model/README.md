# BOND Reranker Model

A cross-encoder reranker model fine-tuned for biomedical ontology entity normalization, designed to work with the BOND (Biomedical Ontology Normalization and Disambiguation) system.

## Overview

The BOND reranker is a cross-encoder model that improves ontology normalization accuracy by re-ranking candidate ontology terms retrieved by BOND's initial retrieval stage. It takes a query-candidate pair and outputs a relevance score.

**Key Benefits:**

- **Improves Accuracy**: Boosts Hit@10 accuracy from ~75-80% (retrieval only) to **85-90%** (with reranker)
- **Context-Aware**: Learns context-dependent relevance (e.g., "lymphocyte" in tonsil vs. blood)
- **Handles One-to-Many Mappings**: The same author term can map to different ontology IDs depending on context

## Model Details

- **Model Type:** Cross-Encoder
- **Base Model:** `bioformers/bioformer-16L`
- **Framework:** Sentence Transformers
- **Max Sequence Length:** 512 tokens
- **Parameters:** ~110M (based on BiomedBERT-base)
- **Output:** Single relevance score per query-candidate pair

## Performance Metrics

Evaluated on biomedical entity normalization development set:

| Metric                      | Score  |
| --------------------------- | ------ |
| **Accuracy**          | 97.50% |
| **F1 Score**          | 82.37% |
| **Precision**         | 79.58% |
| **Recall**            | 85.36% |
| **Average Precision** | 88.67% |
| **Eval Loss**         | 0.230  |

**Best Model:** Checkpoint at step 69,500 (epoch 2.28) with best metric score of 0.9734

## Download Model

Download the reranker model from Hugging Face:

**Model URL:** https://huggingface.co/AronowLab/BOND-reranker

## Installation

```bash
# Install required packages
pip install -U sentence-transformers torch

# Download the model files from the Hugging Face link above
# Place all model files in this directory (reranker-model/)
```

## Configuration

### Where to Update the Reranker Model Path

After downloading the model, you need to configure the path in BOND:

**Option 1: Environment Variable (Recommended)**

- Edit the `.env` file in the BOND root directory
- Set the `BOND_RERANKER_PATH` variable:
  ```bash
  BOND_RERANKER_PATH=/path/to/BOND/reranker-model
  ```
- Example:
  ```bash
  BOND_RERANKER_PATH=/Users/yourname/Desktop/conf/BOND/reranker-model
  ```

**Option 2: Direct Configuration**

- The reranker path is defined in `bond/config.py`
- It reads from the environment variable: `BOND_RERANKER_PATH`
- You can also pass it as a parameter when initializing `BondSettings`:

  ```python
  from bond.config import BondSettings
  from bond.pipeline import BondMatcher

  settings = BondSettings(
      reranker_path="/path/to/BOND/reranker-model",
      enable_reranker=True
  )
  matcher = BondMatcher(settings=settings)
  ```

## Usage

### With BOND Pipeline

```python
from bond.config import BondSettings
from bond.pipeline import BondMatcher

# Configure BOND to use this reranker
settings = BondSettings(
    reranker_path="/path/to/BOND/reranker-model",  # Replace with your model path
    enable_reranker=True
)

matcher = BondMatcher(settings=settings)

# Use the matcher
results = matcher.match("T-cell", field_name="cell_type", tissue="blood", organism="Homo sapiens")
```

### Direct Usage

```python
import torch
from sentence_transformers import CrossEncoder

# Load model from local path
model = CrossEncoder(
    "/path/to/BOND/reranker-model",  # Replace with your model path
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Example: Rank candidates for a query
query = "cell_type: C_BEST4; tissue: descending colon; organism: Homo sapiens"
candidates = [
    "label: smooth muscle fiber of descending colon; synonyms: non-striated muscle fiber of descending colon",
    "label: smooth muscle cell of colon; synonyms: non-striated muscle fiber of colon",
    "label: epithelial cell of colon; synonyms: colon epithelial cell"
]

# Get ranked results with probabilities
ranked_results = model.rank(query, candidates, return_documents=True, top_k=3)

print("Top 3 ranked results:")
for result in ranked_results:
    prob = torch.sigmoid(torch.tensor(result['score'])).item()
    print(f"{prob:.8f} - {result['text']}")
```

## Required Files

After downloading from Hugging Face, this directory should contain:

- `config.json` - Model configuration
- `model.safetensors` (or `pytorch_model.bin`) - Model weights
- `tokenizer_config.json` - Tokenizer configuration
- `vocab.txt` - Vocabulary file
- `tokenizer.json` - Fast tokenizer
- `special_tokens_map.json` - Special tokens mapping

## Training Your Own Reranker

If you want to train a custom reranker model, see the comprehensive training guide:

- **Training Notebook**: [BOND_Reranker_Training_Example.ipynb](../notebooks/BOND_Reranker_Training_Example.ipynb) - Complete step-by-step guide
- **Original Training Notebook**: [Open in Colab](https://colab.research.google.com/drive/1USKLEtyWmbvzOM3ZlkUYDg3EUcMRC1MQ#scrollTo=QRdu0nhobRLA)

### Quick Training Overview

1. **Generate Training Data**: Use `scripts/build_reranker_training_data.py` to generate training data from your benchmark
2. **Train Model**: Use the training notebook with your data
3. **Evaluate**: Test on validation set
4. **Integrate**: Add to BOND pipeline using configuration above

### Training Data Format

Training data should be in JSONL format with the following structure:

```json
{
  "query": "cell_type: T-cell; tissue: blood; organism: Homo sapiens",
  "candidate": "label: T cell; synonyms: T lymphocyte | T-lymphocyte | thymocyte",
  "candidate_id": "CL:0000084",
  "correct_id": "CL:0000084",
  "label": 1.0,
  "retrieval_score": 0.85,
  "retrieval_rank": 0,
  "example_type": "positive"
}
```

**Field Descriptions:**

- `query`: Formatted query string with field type, author term, tissue, and organism
- `candidate`: Formatted candidate ontology term with label, synonyms, and definition
- `candidate_id`: Ontology ID (CURIE) of the candidate
- `correct_id`: Ground truth ontology ID
- `label`: Binary label (1.0 = positive, 0.0 = negative)
- `retrieval_score`: Confidence score from initial retrieval
- `retrieval_rank`: Rank position from retrieval
- `example_type`: Type of example (positive, hard_negative, random_negative)

### Training Hyperparameters

The model was trained with:

- **Base Model**: `bioformers/bioformer-16L`
- **Epochs**: 3
- **Batch Size**: 32
- **Learning Rate**: 2e-5
- **Warmup Ratio**: 0.1
- **Weight Decay**: 0.01
- **Max Grad Norm**: 1.0
- **Pos Weight**: 5.0 (for imbalanced data)

## Architecture

The reranker works as part of the BOND pipeline:

```
┌─────────────────────────────────────────────────┐
│ STAGE 1: RETRIEVAL (Existing BOND Pipeline)    │
├─────────────────────────────────────────────────┤
│ • Dense (FAISS) → Top-50                        │
│ • BM25 (SQLite) → Top-20                        │
│ • Exact matching → Top-5                        │
│ • RRF Fusion → Top-50 candidates                │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ STAGE 2: RERANKER (This Model)                 │
├─────────────────────────────────────────────────┤
│ Model: Cross-Encoder (bioformers/bioformer-16L) │
│ Input: (query, candidate) pairs                 │
│ Output: Relevance scores → Top-10               │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ STAGE 3: LLM DISAMBIGUATION (Optional)         │
├─────────────────────────────────────────────────┤
│ Input: Top-10 reranked candidates               │
│ Output: Final chosen term + reasoning           │
└─────────────────────────────────────────────────┘
```

## Training Data

The model was trained on biomedical entity normalization data covering multiple ontologies including:

- MONDO (diseases)
- HPO (phenotypes)
- UBERON (anatomy)
- Cell Ontology (CL)
- Gene Ontology (GO)
- And other biomedical ontologies

Training data consists of query-candidate pairs with relevance labels, where queries are biomedical entity mentions and candidates are ontology terms.

**Training Dataset Size:** 2,401,485 samples

## License

Apache 2.0

## Citation

If you use this reranker model in your research, please cite:

```bibtex
@software{bond_reranker_2026,
  title={BOND Reranker: Cross-Encoder for Biomedical Ontology Normalization},
  author={Rajdeo, Pankaj and Gelal, Rupesh},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/AronowLab/BOND-reranker}
}
```

## Troubleshooting

### Issue: Model not loading

**Solution:** Ensure all required files are present in the model directory. Check that paths are absolute or correctly relative to your working directory.

### Issue: Out of memory during inference

**Solution:** Reduce batch size or use CPU:

```python
model = CrossEncoder(
    "/path/to/model",
    device='cpu'  # Use CPU if GPU memory is limited
)
```

### Issue: Low accuracy with custom data

**Check:**

1. Text formatting: Does query/candidate format match training format?
2. Model compatibility: Is your data from similar biomedical domains?
3. Consider retraining on your specific data using the training notebook

## Additional Resources

- **BOND Repository**: [github.com/Aronow-Lab/BOND](https://github.com/Aronow-Lab/BOND)
- **Benchmark Dataset**: [huggingface.co/datasets/AronowLab/bond-czi-benchmark](https://huggingface.co/datasets/AronowLab/bond-czi-benchmark)
- **Training Example Notebook**: See `notebooks/BOND_Reranker_Training_Example.ipynb` in the BOND repository
