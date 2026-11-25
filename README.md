# myAI — English GPT (from scratch, ~124M parameters)

This repository contains a single **English language-model stack**. It is
trained from scratch on streaming
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and is
designed to run on a free Kaggle **2 × Tesla T4** session.

The project is **English-only**: the legacy Vietnamese edition has been
removed. All source code lives under `english/`, and all generated artifacts
use English-specific directories.

- [Kaggle guide](KAGGLE_EN.md)
- [Kaggle notebook](kaggle_en.ipynb)

---

## 1. What is included

| Component | File | Notes |
| --- | --- | --- |
| Model | `english/model.py` | GPT decoder, RMSNorm, RoPE, SwiGLU, tied embeddings, KV cache |
| Config | `english/config.py` | `tiny`, `base`, `wide`, exact `param_count()` |
| Tokenizer | `english/tokenizer.py` | SentencePiece BPE, 32K vocab, chat formatting/masking |
| Data | `english/data.py` | streaming, hash split, shuffle buffer, packing, exact resume |
| Pretraining | `english/train.py` | DDP, FP16 AMP, validation, resumable checkpoints |
| SFT | `english/sft.py` | assistant-only loss, padding mask, resumable |
| Generation | `english/generate.py` | KV-cached streaming, top-k/top-p, repetition penalty |
| Chat | `english/chat.py` | streaming CLI, RAG/web grounding, prompt budgeting |
| RAG | `english/rag.py` | dependency-free lexical store (`memory_en/`) |
| Web search | `english/web_search.py` | DuckDuckGo HTML + Wikipedia API, no key |
| Agent routing | `english/agent.py` | decides when live web search is needed |
| Tests | `english/tests/` | 118 CPU tests, no network |

---

## 2. Architecture (exact `base` preset)

| Item | Value |
| --- | --- |
| Parameters | **124,278,528** |
| Non-embedding parameters | 99,112,704 |
| Layers | 14 |
| Width `d_model` | 768 |
| Heads | 12 (head dim 64, standard MHA) |
| SwiGLU hidden | 2,048 |
| Context length | 1,024 |
| Vocabulary | 32,768 |
| Normalization | RMSNorm (`eps=1e-5`) |
| Position encoding | RoPE (`base=10000`) |
| Embeddings | tied input/output |
| Bias | none |
| Dropout | 0.0 |
| Precision | FP16 + GradScaler on T4; BF16 on Ampere+; FP32 on CPU |

Parameter formula (tied embeddings counted once):

```
embedding         = 32768 * 768              = 25,165,824
per block         = q(768*768) + k(768*768) + v(768*768) + o(768*768)
                    + SwiGLU(3 * 768 * 2048) + 2 * 768
                  = 7,079,424
14 blocks         =                           99,111,936
final RMSNorm     =                                  768
lm_head           = 0 (tied)
total             = 25,165,824 + 99,111,936 + 768 = 124,278,528
```

`ModelConfig.param_count()` is checked against the built model at startup, and
`english/tests/test_model.py` asserts the exact number.

Other presets: `tiny` = 393,856 parameters (CPU smoke tests), `wide` =
165,864,832 parameters (896-wide variant).

Special token ids: `pad=0, unk=1, bos=2, eos=3, <|user|>=4,
<|assistant|>=5, <|system|>=6`.

---

## 3. Artifact layout

```
english/                 source code + tests
tokenizer_en/            tokenizer.model, tokenizer.vocab, meta.json
data_en/val/             validation token bins (uint16)
checkpoints_en/          pretrain_last/best.pth, sft_last/best.pth
runs_en/                 JSONL training logs
memory_en/               RAG store (documents.jsonl)
data/sft_sample_en.jsonl 43 English SFT examples (shared data folder)
```

Every checkpoint carries `edition: "en"`, `format`, `arch_version`, the model
config, the tokenizer SHA-256, optimizer/scaler state, per-rank RNG state and
per-rank data-stream state. Loading a legacy checkpoint, a mismatched
tokenizer or a different architecture raises `CheckpointError` instead of
silently producing garbage.

---

## 4. Quick start (CPU, no dataset download)

Requires Python 3.10+, `torch`, `datasets`, `sentencepiece`, `numpy`, `pyarrow`
(see `requirements.txt`).

```bash
pip install -r requirements.txt

# 1. Tiny end-to-end smoke test (trains its own 512-token tokenizer):
python -m english.train --smoke-test --max-steps 4
python -m english.sft   --smoke-test --max-steps 2
python -m english.chat  --smoke-test --prompt "Hello" --no-rag --no-web

# 2. Run the CPU test suite (about 15 s):
python -m unittest discover -s english/tests -t . -v
```

Smoke artifacts are written to `tokenizer_en/smoke/` and
`checkpoints_en/smoke/`, so they never collide with a real run.

## 5. Real pretraining

### 5.1 Train the English tokenizer

The tokenizer is trained once on a bounded streamed sample (default 200 MB) so
it never downloads the whole 1.3T-token corpus:

```bash
python -m english.tokenizer --train --sample-mb 200 --vocab-size 32768
python -m english.tokenizer --inspect
```

The corpus sample is deleted after training unless `--keep-corpus` is passed.
To retrain, add `--force`.

### 5.2 Single GPU

```bash
python -m english.train \
  --preset base \
  --plan-tokens 500000000 \
  --batch-size 8 --grad-accum 4 \
  --amp auto \
  --expect-params 124278528
```

### 5.3 Two GPUs (T4, recommended)

```bash
torchrun --nproc_per_node=2 -m english.train \
  --preset base \
  --plan-tokens 500000000 \
  --batch-size 8 --grad-accum 4 \
  --amp fp16 \
  --expect-params 124278528
```

Resume is automatic: if `checkpoints_en/pretrain_last.pth` exists, the run
continues from its step, token count, optimizer, scaler, RNG and data position.
Use `--resume path/to.pth` for a specific file, `--from-scratch` to ignore it,
and `--dry-run` to print the plan without touching the tokenizer or dataset.

`--plan-tokens` is a token budget, not a dataset limit. The default is 5B
tokens (roughly 150k optimizer steps at 32,768 tokens/step); lower it for a
first run. A serious 5B-token run is far longer than one free Kaggle session,
so checkpoint/resume is the normal workflow. See `KAGGLE_EN.md` for estimates.

### 5.4 Training behaviour

- Streaming `datasets` pipeline, sharded per DDP rank; the full dataset is
  never stored on disk.
- Train/validation split by **BLAKE2b hash of the document id**, so the
  validation documents are never trained on and the split is stable across
  sessions.
- A custom shuffle buffer (default 1,000 documents) whose contents and RNG
  state are stored in the checkpoint. HF's own shuffle buffer is not
  restorable, which is why it is not used.
- **Exact data resume**: HF's iterable `state_dict()` only advances at internal
  table boundaries, so the stream stores the state at the *start* of the
  current table plus the number of raw rows already consumed from it. On resume
  it rewinds to that boundary and replays at most one table (128 rows for the
  parquet stream). This avoids both duplicated and skipped documents.
- Sequence packing: documents are concatenated into full `block_size` windows
  with `<bos>`/`<eos>` markers and no padding.
- Validation loss uses held-out random windows from the validation bin, with
  the same tokenizer fingerprint and `block_size` enforced.
- AMP: FP16 + dynamic GradScaler on T4; BF16 only when natively supported;
  non-finite loss/gradients skip the optimizer step instead of corrupting the
  weights.
- Graceful `Ctrl+C`: saves `pretrain_last.pth` and exits.

---

## 6. Supervised fine-tuning (SFT)

SFT uses the included 43-example English sample by default. Prompt, system and
role-header tokens are masked with `-100`; only assistant content plus the
final EOS contribute to the loss. Padding is excluded through the attention
mask.

```bash
python -m english.sft \
  --init checkpoints_en/pretrain_best.pth \
  --data data/sft_sample_en.jsonl \
  --epochs 3 --batch-size 4 \
  --amp auto
```

Useful flags: `--val-ratio 0.1` (split by prompt key, so the same prompt never
appears in both splits), `--max-examples N`, `--dry-run`, `--resume auto`,
`--from-scratch`. The SFT stream position (epoch, batch, RNG) is checkpointed,
so resuming does not replay or skip examples.

The sample file is intentionally small: it proves the pipeline and gives the
model a few instruction-following examples. For a useful assistant, replace or
extend it with a larger English instruction dataset in the same
`{"user": ..., "assistant": ...}` or `{"messages": [...]}` JSONL format.

---

## 7. Chat, RAG and web search

```bash
# Interactive streaming chat (uses SFT checkpoint if present):
python -m english.chat

# One-shot, no network:
python -m english.chat --prompt "Explain what a transformer is." --no-web

# Force a live web lookup and print sources:
python -m english.chat --prompt "What is the latest Python release?" --show-sources

# Store a note in memory_en/ and use it later:
python -m english.chat --prompt "/rag The project targets two T4 GPUs."
```

Commands: `/help`, `/reset`, `/web <query>`, `/rag <text>`, `/sources`,
`/quit`. Retrieval is lexical (TF-style cosine over tokens) with TTL and URL
deduplication, because a 124M model does not need a second embedding model to
ground a few documents. Web search uses DuckDuckGo's HTML endpoint with a
Wikipedia fallback; both are keyless and can fail silently, in which case chat
continues with local context only.

Generation refuses a prompt that would exceed `block_size - max_new_tokens`
rather than sliding the KV cache (sliding would invalidate RoPE positions).
Chat trims old turns, then system instructions, then retrieved context, then
the question, so the chat template stays valid.

---

## 8. Tests

```bash
python -m unittest discover -s english/tests -t . -v
```

Coverage includes: exact parameter counts, forward/backward, KV-cache
equivalence, attention/padding masks, RoPE cache, tokenizer round-trip and
special ids, chat label masking, SFT truncation/splitting, checkpoint
round-trip and rejection paths, generation determinism/stop handling, data
resume exactness (document, synthetic, packed), DDP environment/AMP selection,
RAG TTL/dedupe, routing, offline web parsing, end-to-end training and SFT smoke
runs with resume, and English-only guards (no legacy imports or non-English
diacritics in the English package). No network access is required.

---

## 9. Limitations

- No English model is shipped trained. The repository contains code, tests and
  a tiny smoke checkpoint; real weights must be produced by training.
- FineWeb-Edu is 1.3T tokens. A 5B-token budget on 2×T4 is already a
  multi-session effort; see `KAGGLE_EN.md` before starting.
- Exact resume requires the same dataset source, tokenizer, `block_size` and
  **world size**. Changing the GPU count requires `--allow-data-restart`, which
  keeps the optimizer/step state but re-reads documents from the start.
- CUDA/NCCL code paths cannot be exercised on the CPU-only development
  machine. The DDP configuration, AMP selection and failure paths are unit
  tested, but actual multi-GPU throughput and NCCL stability must be verified
  on Kaggle.
- T4 (Turing) has no native BF16. `--amp auto` correctly selects FP16 +
  GradScaler there; `--amp bf16` raises.
- The SFT sample is tiny and only demonstrates the recipe.
- RAG is lexical, not semantic; web search depends on external endpoints and
  may be rate-limited or unavailable.
- Checkpoints contain Python objects (NumPy RNG tuples), so only load
  checkpoints you trust (`torch.load(..., weights_only=False)` is required).

---

## 10. Repository layout and English-only status

```
english/                  source code and CPU tests
data/sft_sample_en.jsonl  43 English SFT examples
tokenizer_en/             SentencePiece model and metadata
checkpoints_en/           pretrain and SFT checkpoints
data_en/                  validation token bins
runs_en/                  JSONL training logs
memory_en/                RAG document store
README.md                 this file
KAGGLE_EN.md              Kaggle 2×T4 guide
kaggle_en.ipynb           end-to-end Kaggle notebook
requirements.txt          Python dependencies
```

The legacy Vietnamese modules, tokenizer, checkpoints, datasets, test suite
and memory store were removed. No source module outside `english/` is required
at runtime, and the package imports only its own modules and third-party
libraries.
