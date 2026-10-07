# Architecture

## The one-sentence version

EmbeddingGemma 2's text tower is frozen and reused as a shared feature extractor;
everything that actually decides relevance — fusion, token interaction, reranker
blocks, heads — is new trainable code sitting on top of it.

## Main pipeline

```mermaid
flowchart TD
    Q[Query] --> BG
    D[Documents] --> BG

    subgraph BG["EmbeddingGemma 2 — Shared Text Backbone (270M, unchanged)"]
        TE[Token embeddings]
        TL[24 transformer layers<br/>hidden 512]
        PR[Projection 512 → 768]
        TE --> TL --> PR
    end

    BG --> QR[Query representation]
    BG --> DR[Document representation]

    QR --> FU[QueryDocumentFusion]
    DR --> FU

    QR --> TI[Optional token interaction<br/>cross-attention]
    DR --> TI

    FU --> RL[Reranker layers<br/>Norm → FFN → Residual]
    TI --> RL

    RL --> RH[RelevanceHead<br/>raw logit]
    RL --> CH[ConfidenceHead]

    RH --> SC[Scores]
    CH --> CF[Confidence]
    SC --> RK[Ranking + Policy]
    CF --> RK
    RK --> AB[Abstention]

    style BG fill:#1f2937,stroke:#4b5563,color:#f9fafb
    style RK fill:#065f46,stroke:#10b981,color:#ecfdf5
    style AB fill:#7f1d1d,stroke:#ef4444,color:#fee2e2
```

## Retrieval vs reranking

The distinction matters, because a reranker that only wraps cosine similarity is
not a reranker.

```mermaid
flowchart LR
    subgraph BE["Embedding retrieval (bi-encoder)"]
        Q1[Query] --> E1[Encoder]
        E1 --> VQ[768-d vector]
        D1[Doc] --> E2[Encoder]
        E2 --> VD[768-d vector]
        VQ --> COS[Cosine similarity]
        VD --> COS
        COS --> R1[Score]
    end

    subgraph RR["Reranking (cross-encoder)"]
        Q2[Query] --> ENC[Shared backbone]
        D2[Doc] --> ENC
        ENC --> FUSE[Fusion<br/>Q, D, Q⊙D, |Q−D|]
        FUSE --> LAYERS[Reranker layers]
        LAYERS --> HEAD[Relevance head]
        HEAD --> R2[Score]
    end
```

| | Embedding retrieval | Reranking |
|---|---|---|
| Encoding | Query and document **independently** | Query and document **jointly** |
| Representation | One vector per item | Token states + pooled + fused features |
| Interaction | Fixed similarity function | Learned, token-level |
| Cost per document | Amortized, indexable | Must re-run per (query, document) pair |
| Position | Stage 1, over the whole corpus | Stage 2, over ~50–200 candidates |

This project implements **both**: `evaluation/baseline.py` for stage 1,
`embeddinggemma_reranker/` for stage 2.

## Module map

```mermaid
flowchart TD
    CFG[config.py] --> BB[backbone.py]
    CFG --> FUS[fusion.py]
    CFG --> INT[interaction.py]
    CFG --> LYR[reranker_layers.py]
    CFG --> HED[heads.py]
    CFG --> POL[policy.py]
    CFG --> ST[statistics.py]

    BB --> MOD[model.py<br/>QueryEncoder / DocumentEncoder]
    FUS --> MOD
    INT --> MOD
    LYR --> MOD
    HED --> MOD

    MOD --> INF[inference.py<br/>chunking + ranking API]
    POL --> INF
    POOL[pooling.py] --> BB

    DS[dataset.py] --> COL[collator.py]
    DS --> HN[hard_negatives.py]
    COL --> TRN[trainer.py]
    LOSS[losses.py] --> TRN
    HN --> TRN
    TRN --> MOD

    MET[metrics.py] --> EVL[evaluate.py]
    BAS[baseline.py] --> EVL
    CAL[calibration.py] --> EVL
    EVL --> MOD
```

## Data flow through one forward pass

```mermaid
sequenceDiagram
    participant U as User
    participant I as Inference
    participant B as Shared Backbone
    participant F as Fusion
    participant R as Reranker layers
    participant H as Heads
    participant P as Policy

    U->>I: rerank(query, documents)
    I->>I: chunk long documents
    I->>B: encode query (ONCE)
    B-->>I: query token states + pooled + 768-d embedding
    I->>B: encode all document chunks (batched)
    B-->>I: document token states + pooled + embeddings
    I->>F: (query_i, document_j) pairs
    F-->>I: fused ranking vectors
    I->>R: reranker blocks
    R-->>I: ranking representations
    I->>H: relevance head
    H-->>I: raw logits per candidate
    I->>I: aggregate chunks → document score
    I->>H: confidence head (ranking stats)
    H-->>I: confidence logit
    I->>P: thresholds
    P-->>I: accept / abstain
    I-->>U: ranking + scores + confidence
```

Note the **query is encoded once** and tiled across candidates. That is what keeps
a cross-encoder affordable.

## Interaction mechanisms

`fusion.interaction_type` selects the mechanism:

| Value | Features fed to the projection | Cost |
|---|---|---|
| `none` | `Q + D` | lowest |
| `concatenate` | `[Q, D]` | low |
| `elementwise` | `[Q, D, Q⊙D, |Q−D|]` (default) | low |
| `dot` | `[cos(Q,D), Q, D]` | low |
| `token` | plus cross-attention context and an interaction matrix | `O(T_q · T_d)` |

```mermaid
flowchart LR
    Q[Q] --> F
    D[D] --> F
    Q --> M1[Q ⊙ D]
    D --> M1
    Q --> M2-abs[|Q − D|]
    D --> M2-abs
    F[Q] --> PROJ[Fusion projection]
    D --> PROJ
    M1 --> PROJ
    M2-abs --> PROJ
    PROJ --> RR[Ranking representation]

    style RR fill:#065f46,stroke:#10b981,color:#ecfdf5
```

## Reranker layer

```mermaid
flowchart LR
    X[input] --> N[LayerNorm]
    N --> FF[Linear → GELU → Dropout<br/>→ Linear → Dropout]
    FF --> ADD((+))
    X --> ADD
    ADD --> OUT[output]

    style ADD fill:#1f2937,stroke:#6b7280,color:#f9fafb
```

Pre-norm with a residual connection, repeated `num_reranker_layers` times.

## Training modes

```mermaid
flowchart TD
    START[choose train_mode] --> H1
    START --> R1
    START --> L1
    START --> F1

    H1[head_only] --> H2[relevance + confidence heads]
    R1[reranker_only] --> R2[fusion + interaction<br/>+ layers + heads]
    L1[lora] --> L2[reranker_only modules<br/>+ LoRA adapters on backbone]
    F1[full] --> F2[everything — requires<br/>explicit acknowledgement]

    style F2 fill:#7f1d1d,stroke:#ef4444,color:#fee2e2
```

`full` is never selected automatically; it raises `FullFinetuneNotConfirmed`
unless the caller passes `allow_full_finetune=True`.

## Chunk aggregation for long documents

```mermaid
flowchart LR
    DOC[Document] --> SPLIT[Split into token-budgeted chunks]
    SPLIT --> SC[Score each chunk]
    SC --> AGG{aggregation}
    AGG -->|max| M[max chunk score]
    AGG -->|mean| ME[mean chunk score]
    AGG -->|top_k_mean| TK[mean of top-k]
    M --> DS[Document score]
    ME --> DS
    TK --> DS
    DS --> FLAG[chunked=true,<br/>num_chunks=N]

    style FLAG fill:#78350f,stroke:#f59e0b,color:#fffbeb
```

Truncation is always reported. A long document is never silently clipped.

## What is and is not changed in the checkpoint

| Component | Changed? |
|---|---|
| Text tower hidden size (512) | no |
| Text tower layer count (24) | no |
| Attention heads (4) / KV heads (2) | no |
| Vocabulary (262 144) | no |
| Tokenizer | no |
| Vision / audio encoders | not loaded |
| Fusion / interaction / reranker / heads | **new** |

The original checkpoint therefore stays load-compatible. See
`docs/source_model_analysis.md` for the verified values this table refers to.

## Honest status

The architecture above is implemented. What it **is not** yet: a trained model.
No reranker checkpoint has been trained or benchmarked in this repository, so no
claim is made that it beats EmbeddingGemma retrieval. Run
`scripts/evaluate_reranker.py` after training to find out.
