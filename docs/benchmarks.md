# Benchmarks and Evaluation Protocol

This file defines a reproducible evaluation protocol and records functional gate results.

## 1. Environment Capture Template

Record this before benchmark runs:

- CPU model:
- RAM:
- GPU (if any):
- OS + version:
- Python version:
- Embedding model:
- LLM provider + model:
- `EMBEDDING_BATCH_SIZE`:
- `LOW_CONFIDENCE_THRESHOLD` / `HARD_NOT_FOUND_THRESHOLD`:

## 2. Indexing Performance Protocol

Dataset classes:

- Small: <= 5 MB total
- Medium: 5-50 MB total
- Large: > 50 MB total

Metrics:

- files/sec
- chunks/sec
- mean parse time per file
- mean embedding time per batch
- mean upsert time per batch
- total index wall-clock per upload

Measurement source:

- `.rag/log.txt` entries:
  - `file_parsed`
  - `file_batch_done`
  - `file_done`

## 3. Query Performance Protocol

Run each mode (`fast`, `moderate`, `accurate`) over the same prompt set.

Metrics:

- P50/P95 latency per query
- token stream start latency (for `/chat/stream`)
- retrieval hit count distribution
- not-found precision on negative prompts

## 4. Quality Metrics

Recommended scoring dimensions:

- Citation precision: claimed facts supported by cited chunk
- Citation recall: all critical claims in answer have citations
- Retrieval recall@k: known relevant chunk appears in retrieved set
- Hallucination rate: unsupported claims
- Not-found correctness: returns "Not found in indexed docs" when appropriate

## 5. Functional Gate Packs

### 5.1 Legal/Policy stress suite (32 prompts)

Coverage stresses:

- hybrid retrieval
- table generation
- checklist extraction
- diff-style change output
- citation correctness
- negative/not-found behavior

Status (user-reported):

- Result: PASS
- Notes: all listed prompts reported as working.

### 5.2 Textbook quick gate (30 prompts)

Target gate:

- expected >= 27/30

Status (user-reported):

- Result: PASS
- Notes: lookup/compare/checklist/adversarial/not-found prompts reported as working.

## 6. Suggested Automated Harness

Future work:

1. Add scripted prompt runner against `/chat` and `/chat/stream`.
2. Add assertion rules for:
   - required citation count
   - table/checklist format constraints
   - exact not-found string for negative cases
3. Persist run artifacts:
   - prompt
   - raw answer
   - citations
   - pass/fail reason

## 7. Reporting Template

Use this table for publishable benchmark snapshots:

| Mode | Queries | P50 Latency | P95 Latency | Avg Citations | Not-Found Precision |
|---|---:|---:|---:|---:|---:|
| fast | - | - | - | - | - |
| moderate | - | - | - | - | - |
| accurate | - | - | - | - | - |

| Dataset | Files | Total Chunks | Total Index Time | Chunks/sec |
|---|---:|---:|---:|---:|
| Small | - | - | - | - |
| Medium | - | - | - | - |
| Large | - | - | - | - |
