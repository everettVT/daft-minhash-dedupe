<div align="center">

# Daft Minhash Deduplication

Canonical Multimodal Workload Sandbox for Minhash Deduplication on Common Crawl HTML Documents using daft Dataframes

</div>

---

## Overview
`minhash_dedupe.py` implements a scalable deduplication pipeline using MinHash and Locality-Sensitive Hashing (LSH) for processing large text datasets, such as Common Crawl HTML extracts. It leverages daft (a distributed DataFrame library) for efficient computation, including text normalization, MinHash signature generation, LSH banding, and connected components for clustering duplicates. The pipeline is designed for high-throughput deduplication while minimizing false positives/negatives via optimized parameters.

Key goals: Identify and remove near-duplicate text blocks (e.g., from web crawls) based on Jaccard similarity thresholds, outputting unique representatives.

## Key Components
1. **Preprocessing**:
   - Extracts text blocks from HTML using Selectolax (removes scripts/styles).
   - Filters non-empty, valid UTF-8 content.
   - Adds unique block IDs.

2. **Text Normalization**:
   - Optional: Remove punctuation, lowercase, NFD Unicode normalization, whitespace cleanup.
   - Applied via daft's string functions for consistency.

3. **MinHash & LSH**:
   - Computes MinHash signatures (e.g., 64 permutations, 5-grams) using XXHash.
   - Bands signatures into buckets (optimal B/R from threshold) for candidate pair generation.
   - Builds edges between similar nodes.

4. **Connected Components**:
   - Uses alternating Large/Small Star algorithm (or two-phase variant) for union-find-like clustering.
   - Includes global min-label propagation for convergence.
   - Optional igraph validation for correctness.

5. **Output**:
   - Merges results to keep only unique representatives per component.
   - Partitioned Parquet saving with Snappy compression.

## Installation

Clone this repository and then run:

```bash
cd daft-minhash-dedupe && uv venv && uv sync 
```

if you don't have `uv` installed:

```bash
pip install uv
```

## Authentication

In order to access the Common Crawl dataset from S3 you will need to authenticate with a `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` as environment variables.


## Usage
- **Instantiation**: Create a `MinHashDedupePipeline` object with params (e.g., `num_perm=64`, `threshold=0.7`). Note not all thresholds work for any number of permutations. The pipeline will assert an error upon instantiation if this is the case. 
- **Running**: Call `pipeline(df)` on a preprocessed daft DataFrame (e.g., from `preprocess_common_crawl_html`).
- **Main Script**: Handles S3 I/O, env vars, and full pipeline execution for Common Crawl segments.
- **Example** (from main):
  ```python
  pipeline = MinHashDedupePipeline(output_uri="s3://bucket/output", ...)
  df_prepped = preprocess_common_crawl_html("s3://commoncrawl/...")
  results = pipeline(df_prepped)
  partitioned_save("s3://bucket/output", results, chunk_size=200000)
  ```

## Parameters
- **Core**: `num_perm` (signatures), `ngram_size` (shingles), `threshold` (Jaccard similarity), `seed`, `hash_function`.
- **Normalization**: Booleans for punctuation, case, Unicode, whitespace.
- **Algo**: `algorithm` ("alternating" or "two_phase"), `max_loops`, `igraph_validate`.
- **I/O**: S3 configs via `IOConfig`; supports Ray for partitioning.

## Quick Onboarding Tips
- **Dependencies**: daft, Selectolax, SciPy, igraph (optional), Ray (for large-scale).
- **Testing**: Use small `ROW_LIMIT` for local runs; check `friction/` dir for prototypes.
- **Extensions**: Modular—extend normalization or add custom hash functions easily.
- **Performance**: Scales to millions of rows; tune partitions for memory.

