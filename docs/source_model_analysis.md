# Source Model Analysis — EmbeddingGemma 2 270M

This document records what was **actually observed** about the source model before
any reranker code was written. It is deliberately split into three parts:

1. **Verified information** — read from the source artifacts / APIs.
2. **Implementation assumptions** — decisions we made that the source does not define.
3. **New Reranker modules** — everything this project adds on top of the source.

Nothing in section 3 changes the original checkpoint's dimensions, vocabulary,
layer count, or tokenizer, so `google/embeddinggemma-2` remains load-compatible.

---

## 0. What could and could not be done on this machine

| Step | Status | Notes |
|---|---|---|
| `ollama pull embeddinggemma-2:270m` | **NOT DONE** | No Ollama runtime is available on this device (Android, no container runtime). |
| `ollama show embeddinggemma-2:270m` | **NOT DONE** | Requires a local Ollama daemon at `localhost:11434`. |
| `ollama show --modelfile ...` | **NOT DONE** | Same reason. |
| Official HF `config.json` | **DONE** | Fetched from `huggingface.co/google/embeddinggemma-2`. |
| Official HF `config_sentence_transformers.json` | **DONE** | Fetched. |
| Official HF `1_Pooling/config.json` | **DONE** | Fetched. |
| Official HF `README.md` (model card) | **DONE** | Fetched. |
| Official model card (ai.google.dev) | **DONE** | Fetched. |
| Official checkpoint weights | **NOT DOWNLOADED** | ~3 GB safetensors; not required for architecture work. |

### 1.1 What an Ollama artifact actually contains

To be explicit, because this is a common misconception: an Ollama model is **not**
Python training source code. `ollama pull` downloads a GGUF-format blob plus a
template/tokenizer metadata file. `ollama show --modelfile` prints the *Modelfile*
that was used to build the image (FROM, PARAMETER, TEMPLATE, LICENSE lines), not a
training repository. There is no Transformers `config.json`, no module tree, and no
Python source inside an Ollama model.

Therefore this project does **not** claim to have inspected an Ollama blob. It
inspects the upstream Hugging Face checkpoint, which is the artifact Ollama itself
is built from, and cross-references the Ollama library page for the tag-level facts
(sizes, context window, modality support).

---

## 1. Verified information

### 1.1 Ollama library page — `embeddinggemma-2:270m`

Read from `https://ollama.com/library/embeddinggemma-2`:

| Field | Value |
|---|---|
| Tag | `embeddinggemma-2:270m` |
| Download size | **378 MB** |
| Context window | **256K** |
| Input modalities | **Text only** |
| Parameter class | **270m** |

Sibling tags on the same page, for context:

| Tag | Size | Context | Input |
|---|---|---|---|
| `embeddinggemma-2:270m` | 378MB | 256K | Text |
| `embeddinggemma-2:440m` | 714MB | 256K | Text, Image |
| `embeddinggemma-2:570m` | 990MB | 256K | Text |
| `embeddinggemma-2:740m` (latest) | 1.3GB | 256K | Text, Image |

The 270M tag being **text-only** is the key fact for this project: the target
configuration never needs the vision or audio encoders.

### 1.2 Official checkpoint — `google/embeddinggemma-2`

Read from `https://huggingface.co/google/embeddinggemma-2/raw/main/config.json`:

```json
{
  "architectures": ["EmbeddingGemma2Model"],
  "model_type": "embedding_gemma2",
  "dtype": "bfloat16",
  "audio_token_id": 258881,
  "boi_token_id": 255999,
  "boa_token_id": 256000,
  "eoi_token_id": 258882,
  "eoa_token_index": 258883,
  "image_token_id": 258880,
  "video_token_id": 258884,
  "text_config": {
    "model_type": "(omitted upstream)",
    "hidden_size": 512,
    "embedding_dim": 768,
    "head_dim": 256,
    "num_hidden_layers": 24,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "intermediate_size": 2048,
    "hidden_activation": "gelu_pytorch_tanh",
    "vocab_size": 262144,
    "rms_norm_eps": 1e-06,
    "sliding_window": 512,
    "hidden_size_per_layer_input": 512,
    "layer_types": [ "sliding_attention" x5, "full_attention", ... ],
    "per_layer_config": {
      "05": { "head_dim": 512, "num_key_value_heads": 1 },
      "11": { "head_dim": 512, "num_key_value_heads": 1 },
      "17": { "head_dim": 512, "num_key_value_heads": 1 },
      "23": { "head_dim": 512, "num_key_value_heads": 1 }
    },
    "rope_parameters": {
      "full_attention":    { "rope_type": "default", "rope_theta": 1000000.0 },
      "sliding_attention": { "rope_type": "default", "rope_theta": 10000.0 }
    }
  },
  "audio_config": {
    "model_type": "gemma4_audio",
    "hidden_size": 1024,
    "num_hidden_layers": 12,
    "num_attention_heads": 8,
    "output_proj_dims": 1536
  },
  "vision_config": {
    "model_type": "gemma4_vision",
    "hidden_size": 768,
    "num_hidden_layers": 16,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "patch_size": 16,
    "position_embedding_size": 10240
  },
  "vision_soft_tokens_per_image": 280
}
```

**Important structural reading of `text_config`:**

- `hidden_size: 512` is the **per-layer working width** of the text tower.
- `embedding_dim: 768` is the **output** width. The text tower therefore *projects*
  512 → 768 somewhere near the top. That projection is the reason the published
  embedding is 768-dimensional even though the transformer body is 512-wide.
- `num_attention_heads: 8` × `head_dim: 256` = **2048** inner attention width
  (wider than `hidden_size`, which is normal for these decoupled-dim designs).
- `num_key_value_heads: 2` → grouped-query attention with 4 query heads per KV head.
- 4 of the 24 layers (`05`, `11`, `17`, `23`) override to `head_dim: 512` /
  `num_key_value_heads: 1`, i.e. a narrower, cheaper attention stack every 6 layers.

### 1.3 Official model card — parameter breakdown

Read from the Hugging Face model card and `https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2`:

| Component | Parameters | Notes |
|---|---|---|
| Text tower | **270M** | the `270m` tag; text + code |
| Vision tower | **170M** | image / video |
| Audio tower | **300M** | audio |
| **Total** | **740M** | the `latest` tag |

Additional verified facts from the model card:

- Output space: **768-dimensional**, shared across all modalities.
- The repo's `safetensors` index reports **744,371,512** parameters in BF16
  (slightly above the 740M marketing figure — the difference is embedding tables
  and vision soft tokens counted separately).
- **MRL (Matryoshka) support**: representations may be truncated to 512d / 256d / 128d
  and re-normalized. 128d is documented as best for text-only workloads.
- Multilingual: **100+ languages**, `language: multilingual` in `cardData`.
- Code is a first-class input (MTEB code tasks are reported).
- License: **Apache 2.0** (`license: apache-2.0` in `cardData`, and the Ollama page
  links to the Gemma 4 license terms).

### 1.4 Pooling and projection

Read from `1_Pooling/config.json`:

```json
{ "embedding_dimension": 768, "pooling_mode": "mean", "include_prompt": true }
```

Read from `config_sentence_transformers.json`:

```json
{
  "model_type": "SentenceTransformer",
  "similarity_fn_name": "cosine",
  "default_prompt_name": null,
  "prompts": {
    "query":            "task: search result | query: ",
    "document":         "title: none | text: ",
    "Retrieval-query":  "task: search result | query: ",
    "Retrieval-document": "title: none | text: ",
    "Reranking":        "task: search result | query: ",
    "CodeRetrieval":    "task: code retrieval | query: ",
    "QuestionAnswering":"task: question answering | query: ",
    "FactChecking":     "task: fact checking | query: ",
    "Classification":   "task: classification | query: ",
    "Clustering":       "task: clustering | query: ",
    "STS":              "task: sentence similarity | query: "
  }
}
```

Two consequences that directly shape this project:

1. **Pooling is mean over the attention mask, at 768 dims, then L2-normalized.**
2. **The prompt prefixes are asymmetric**: queries get `task: ... | query: ` and
   documents get `title: {title} | text: {content}` (or `title: none | text: `).
   This project preserves exactly that convention in `PromptTemplates`.

### 1.5 Tokenizer

From the checkpoint file list and `config`:

- `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json` are present.
- Special tokens: `bos=<bos>`, `eos=<eos>`, `pad=<pad>`, `unk=<unk>`, `mask=<mask>`.
- `pad_token_id: 0`, `bos_token_id: 2`, `eos_token_id: 1` in `text_config`.
- `vocab_size: 262144` — a very large multilingual vocabulary. This is why
  multilingual text must be passed through untouched, and why there is no
  English-only preprocessing anywhere in this project.

### 1.6 Quantization

- The **checkpoint** is BF16 (`"dtype": "bfloat16"`).
- The **Ollama tag** `embeddinggemma-2:270m` is 378 MB. At BF16 that would be
  ~1.5 GB for 744M parameters, so the 270M text-only tag is consistent with a
  **quantized** build (Q8-family or similar). The exact quantization scheme is a
  property of the Ollama blob, which we could not download here; see §1.0.
- This project does **not** quantize anything. It loads the HF checkpoint in
  whatever dtype the user requests.

### 1.7 Which modules are used for text-only 270M mode

From the model card's *Selective Encoder Loading* section: vision and audio
encoders are independent components and can be disabled via `config_kwargs`.

For the `270m` text target, the modules actually exercised are:

| Module | Used in 270M text mode |
|---|---|
| Text transformer body (24 layers, hidden 512) | **yes** |
| Text 512 → 768 projection | **yes** |
| Mean pooling + L2 normalize | **yes** |
| `vision_config` tower (170M) | **no** |
| `audio_config` tower (300M) | **no** |
| Vision soft tokens (280/image) | **no** |
| Image/video/audio token ids | **no** |

---

## 2. Implementation assumptions

These are *our* choices. The source model does not define them.

1. **Backbone access path.** We assume `transformers` exposes the text tower with
   `output_hidden_states=True`, giving a `(25, B, T, 512)` tensor (embedding output
   + 24 layer outputs). `scripts/inspect_source.py` verifies this at runtime rather
   than assuming it, and the code degrades gracefully to pooled-only output if a
   given `transformers` version does not expose hidden states.

2. **Which hidden layers feed the reranker.** Default is the **last** transformer
   layer. A `hidden_layer_ids` config option allows selecting any subset or a mean
   over a range. The default is a choice, not a finding.

3. **Token truncation.** We assume 512 tokens is the practical working length
   (`sliding_window: 512`, and 270M is an on-device model). Longer documents are
   **chunked and aggregated**, never silently truncated — see §3.6.

4. **Score semantics.** Raw reranker logits are unbounded. We do not squash them
   during training. A sigmoid is applied only when a calibrated probability is
   requested at inference time.

5. **Prompt prefixes.** We reuse the upstream prefixes verbatim rather than
   inventing new ones. Users may disable them via config if they are training
   against a different convention.

6. **Input formatting.** `title: {title} | text: {content}` for documents with a
   title, `title: none | text: {content}` otherwise. Same as upstream.

7. **UTF-8 handling.** No transliteration, no lowercasing, no English stop-word
   removal. Text reaches the tokenizer exactly as provided.

8. **Code handling.** Code is treated as ordinary text; no code-specific tokenizer
   is introduced. The `CodeRetrieval` prefix is available for query side.

---

## 3. New Reranker modules

Everything in this section is new code written by this project. None of it modifies
the checkpoint.

### 3.1 QueryEncoder / DocumentEncoder

Two thin façades over **one shared** `EmbeddingGemma2TextBackbone` instance.
Default is a single backbone, one tokenizer, one copy of the weights in memory.
`share_backbone=False` is available but explicitly opt-in, because duplicating a
270M tower doubles memory for no accuracy benefit.

### 3.2 Hidden-state access

`EmbeddingGemma2TextBackbone` exposes, per forward pass:

- `token_hidden_states` — `(B, T, 512)` selected layer(s)
- `pooled_hidden` — mean-pooled, `(B, 512)`
- `embedding` — the native 768-d output, mean-pooled + normalized
- `attention_mask`

### 3.3 QueryDocumentFusion

Configurable (`interaction_type`):

| Mode | What it computes |
|---|---|
| `none` | concatenate `Q` and `D` only |
| `elementwise` | `Q`, `D`, `Q*D`, `|Q-D|` (the classic four-feature set) |
| `concatenate` | `Q` then `D` |
| `dot` | scalar cosine/dot product as a single feature |

Output is a learned projection to `reranker_dim`.

### 3.4 Token-level interaction

Optional stronger path (`interaction_type: token`):

- `query_tokens × document_tokens` → interaction matrix
- cross-attention where document tokens attend to query tokens
- aggregator reduces the token axis to a vector
- projected into the same `reranker_dim` and concatenated with fusion features

Default is `elementwise` (cheap); `token` is opt-in because token-level attention
is O(T_q · T_d) and costs more than the 270M backbone itself at long lengths.

### 3.5 Reranker layers + heads

`reranker_dim: 512`, `num_reranker_layers: 2`, `dropout: 0.1` by default.
Each layer is pre-norm → FFN (GELU) → residual. On top:

- `RelevanceHead` — one raw logit per (query, document) pair
- `ConfidenceHead` — optional; sees the ranking representation, the top-1 score,
  the top1−top2 margin, and the score distribution

### 3.6 Long-document chunking

Documents are split into token-budgeted chunks, each chunk is scored, and chunk
scores are aggregated by `max`, `mean`, or `top_k_mean`. When chunking happens the
result records `chunked: true` and `num_chunks`, so truncation is never silent.

### 3.7 Abstention / policy

`minimum_relevance`, `minimum_margin`, `minimum_confidence` are **policy**, applied
after the network, never inside it. `insufficient_confidence: true` is returned
when no candidate clears the thresholds.

---

## 4. Summary table

| Property | Value | Source |
|---|---|---|
| Model type | `embedding_gemma2` / `EmbeddingGemma2Model` | verified |
| Text layers | 24 | verified |
| Text hidden size | 512 | verified |
| Text output dim | 768 | verified |
| Attention heads / KV heads | 8 / 2 | verified |
| Head dim | 256 (512 on layers 5/11/17/23) | verified |
| FFN size | 2048 | verified |
| Activation | `gelu_pytorch_tanh` | verified |
| Norm | RMSNorm, eps 1e-6 | verified |
| RoPE | default; theta 1e4 sliding / 1e6 full | verified |
| Sliding window | 512 | verified |
| Vocabulary | 262144 | verified |
| Pooling | mean, include_prompt=true | verified |
| Similarity | cosine | verified |
| Checkpoint dtype | bfloat16 | verified |
| Total params | 744,371,512 (BF16 index) | verified |
| Text / vision / audio split | 270M / 170M / 300M | verified |
| Context window (Ollama tag) | 256K | verified |
| Ollama 270m size / modality | 378 MB, text-only | verified |
| Languages | 100+ | verified |
| Reranker dim | 512 | this project (assumption) |
| Reranker layers | 2 | this project (assumption) |
| Interaction default | `elementwise` | this project (choice) |
