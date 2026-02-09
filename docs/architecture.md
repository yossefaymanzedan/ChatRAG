# Architecture

This document provides executable architecture diagrams (Mermaid) and control-flow notes.

## Ingestion and Indexing Pipeline

```mermaid
flowchart TD
    A[Upload Files\n/index/upload] --> B[Write to .rag/uploads/upload_id]
    B --> C[Index Job Queue\nIndexJobManager.start_index]
    C --> D[Scan Supported Files]
    D --> E[Parse by Extension\nPDF/DOCX/PPTX/TXT/MD/CSV]
    E --> F[Front-Matter + TOC Filtering]
    F --> G[Build Chunks + Anchors]
    G --> H[Optional doc_summary chunk]
    H --> I[Replace chunks in SQLite + FTS5]
    I --> J[Embed in batches\nEMBEDDING_BATCH_SIZE]
    J --> K[Upsert vectors to Chroma\nretry enabled]
    K --> L[Mark document processed]
```

## Query and Answer Pipeline

```mermaid
flowchart TD
    Q[User Query] --> R[Follow-up Expansion]
    R --> S[Retrieve Dense + FTS]
    S --> T[Fusion Score\n0.6*vector + 0.4*fts]
    T --> U{Mode}
    U -->|fast| V[Single-pass retrieval]
    U -->|moderate| W[Iterative retrieval + selective deep dig]
    U -->|accurate| X[LangGraph ReAct + global deep dig]
    V --> Y[Upload-scope filter]
    W --> Y
    X --> Y
    Y --> Z[Strip doc_summary + TOC/front matter chunks]
    Z --> AA[Build citation context C1..Cn]
    AA --> AB[LLM constrained answer synthesis]
    AB --> AC[Citation compression + fallback]
    AC --> AD[Final answer + citations]
```

## Storage Topology

```mermaid
graph LR
    U[(Uploads\n.rag/uploads)] --> P[Parser]
    P --> S1[(SQLite documents/chunks)]
    P --> S2[(SQLite FTS5)]
    P --> E[Embedding Service]
    E --> C[(Chroma Collection)]
    Q2[Retriever] --> C
    Q2 --> S2
    Q2 --> S1
```

## Design Notes

- Upload scoping is strict: retrieval results are filtered by `upload_id` to prevent cross-upload leakage.
- Citation generation excludes `doc_summary` and front-matter-like chunks.
- `moderate` and `accurate` use multi-step retrieval orchestration; `fast` uses a single retrieval pass.
