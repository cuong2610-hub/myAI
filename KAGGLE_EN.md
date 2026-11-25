# Running myAI on Kaggle (2 × Tesla T4)

This is the practical Kaggle guide for the English-only `myAI` project. It
assumes you have a Kaggle account with phone verification (needed for Internet
access) and GPU quota.

All commands run from the repository root and use the `english.*` modules. The
legacy Vietnamese training scripts and artifacts no longer exist.

---

## 1. Notebook settings

1. Create a new notebook, then **File → Import notebook** and upload
   `kaggle_en.ipynb` (or copy its cells).
2. **Settings → Accelerator → GPU T4 ×2**.
3. **Settings → Internet → On** (required for FineWeb-Edu and DuckDuckGo).
4. **Settings → Persistence → Files** (so `/kaggle/working` survives a commit).
5. Make sure the repository source is available. Either:
   - upload this repository as a **Kaggle Dataset** and copy it into
     `/kaggle/working`, or
   - set `REPO_URL` in the first notebook cell and let it `git clone`.

The notebook is safe to **Run All**: real pretraining, tokenizer training and
SFT are gated behind `RUN_REAL_PRETRAIN`, `RUN_TOKENIZER_TRAIN`, `RUN_RESUME`
and `RUN_SFT`, which all default to `False`. Flip only the ones you need.

## 2. Install

```bash
pip install -q -r requirements.txt
```

`datasets` pulls `pyarrow`, which is required for streaming parquet shards of
FineWeb-Edu. The Kaggle image already has a CUDA build of PyTorch; do not
reinstall it unless Kaggle's version is too old.

## 3. Environment

The commands below assume the repository root is the working directory:

```bash
cd /kaggle/working/myai-en   # adjust to where the repository lives
export HF_HOME=/kaggle/working/.hf
export TOKENIZERS_PARALLELISM=false
```

`HF_HOME` keeps the streamed shard cache inside the writable working directory.
The cache can grow to several GB; delete it between runs if disk runs low.

## 4. Smoke test first (CPU or GPU, ~1 minute)

```bash
python -m english.train --smoke-test --max-steps 4
python -m english.sft   --smoke-test --max-steps 2
python -m english.chat  --smoke-test --prompt "Hello" --no-rag --no-web
```

These train a 512-token smoke tokenizer and a 393,856-parameter model under
`tokenizer_en/smoke/` and `checkpoints_en/smoke/`. They do not touch the real
tokenizer or checkpoints. Also run the CPU suite once:

```bash
python -m unittest discover -s english/tests -t . -v
```

## 5. Train the real tokenizer (once)

```bash
python -m english.tokenizer --train --sample-mb 200 --vocab-size 32768
python -m english.tokenizer --inspect
```

This streams a 200 MB English sample, trains a 32K SentencePiece BPE model,
validates the special ids (`pad=0, unk=1, bos=2, eos=3, user=4,
assistant=5, system=6`) and writes `tokenizer_en/tokenizer.model`. It takes a
few minutes and needs Internet only for that sample. The sample text is deleted
afterwards unless `--keep-corpus` is used.

If the tokenizer already exists, do not retrain it for a run you intend to
resume: the checkpoint stores the tokenizer SHA-256 and refuses a mismatch.

## 6. Pretrain

### 6.1 Two GPUs (recommended)

```bash
torchrun --nproc_per_node=2 -m english.train \
  --preset base \
  --plan-tokens 500000000 \
  --batch-size 8 --grad-accum 4 \
  --amp fp16 \
  --expect-params 124278528 \
  --eval-every 250 --save-every 250 --log-every 10
```

- `tokens/step = 2 GPUs × 8 × 4 × 1024 = 65,536`, so 500M tokens is about
  7,630 optimizer steps.
- The base model is exactly 124,278,528 parameters; `--expect-params` aborts
  immediately if that ever changes.
- `--resume auto` is the default behaviour when
  `checkpoints_en/pretrain_last.pth` exists, so **run the same command again**
  in the next session to continue. Add `--from-scratch` only for a deliberate
  new run (it will overwrite `pretrain_last.pth` at the first save).

### 6.2 Resume in a later session

```bash
torchrun --nproc_per_node=2 -m english.train \
  --preset base --plan-tokens 500000000 \
  --batch-size 8 --grad-accum 4 --amp fp16 \
  --expect-params 124278528 \
  --resume auto
```

You should see `[resume] ... data=exact` in the log. `data=exact` means the
shuffle buffer and the raw stream position were restored. If you changed the
number of GPUs, resume requires `--allow-data-restart`; that keeps optimizer and
step state but re-reads the stream from the beginning (not exact).

### 6.3 Choosing a token plan

The default CLI budget is 5B tokens, but free Kaggle quota is the real limit.
Rough aggregate throughput for this model on 2×T4 with FP16 is **8–12k
tokens/s**; wall-clock and GPU-hour estimates:

| `--plan-tokens` | Approx. wall time | Approx. GPU-hours | Sessions (12 h max) |
| --- | --- | --- | --- |
| 100M | 2.5–3.5 h | 5–7 | 1 |
| 300M | 7–10 h | 14–21 | 1 |
| 500M | 12–17 h | 24–35 | 2 |
| 1B | 1–1.5 days | 48–70 | 2–3 |
| 5B | 5–7 days | 240–350 | 10–15 |

Kaggle's free GPU quota is about 30 GPU-hours per week and **both T4s count**,
so a 500M-token run can consume most of a week's quota. 5B tokens is a
multi-week project, not a single notebook run. Start with 100M–300M, verify the
loss curve, and scale only if you have quota. You can stop at any time with
`Ctrl+C` (or let the session end) and resume from `pretrain_last.pth`.

Notes:

- `--amp auto` selects FP16 + GradScaler on T4. Do not force `--amp bf16`;
  Turing has no native BF16 and the code raises an error for it.
- If you hit CUDA OOM, reduce `--batch-size` to 4 and raise `--grad-accum` to 8
  to keep the same effective batch (65,536 tokens/step). Throughput will be
  somewhat lower.
- Validation tokens (1M by default) are built once under `data_en/val/` and
  reused as long as the tokenizer, block size and val fraction are unchanged.
- Logs are appended to `runs_en/pretrain.jsonl`. The console prints loss, LR,
  gradient norm, tokens/s, ETA and GPU memory every `--log-every` steps.

### 6.4 Checkpoint sizes and disk

| File | Approx. size |
| --- | --- |
| `pretrain_last.pth` | ~1.5 GB (model + AdamW + scaler + RNG + data state) |
| `pretrain_best.pth` | ~0.5 GB (model only) |
| `sft_last.pth` | ~1.5 GB |
| tokenizer | a few MB |
| validation bin | ~2 MB per 1M tokens |

Keep an eye on `/kaggle/working` disk. `HF_HOME` shard cache is usually the
largest consumer; delete `/kaggle/working/.hf` if needed (the next run
re-downloads shards).

## 7. Preserve checkpoints between sessions

`/kaggle/working` is saved when you commit the notebook, but the reliable way
to move a 1.5 GB checkpoint between sessions is a **Kaggle Dataset**:

1. After a session, run the notebook's “save artifacts” cell. It copies
   `checkpoints_en/`, `tokenizer_en/`, `runs_en/` and `data_en/val/` into
   `/kaggle/working/artifacts_en/` and writes a ZIP.
2. **Save Version → Save & Run All (Commit)**. The output becomes part of the
   notebook version.
3. Create a **new Dataset** from that notebook output (or download the ZIP and
   upload it as a private Dataset).
4. In the next session, attach the Dataset and copy it back:

```bash
cp -r /kaggle/input/myai-en-artifacts/artifacts_en/* /kaggle/working/myai-en/
```

Alternatively, keep the same notebook version and rely on
`/kaggle/working` persistence, but always verify that
`checkpoints_en/pretrain_last.pth` exists before launching training.

## 8. SFT

```bash
python -m english.sft \
  --init checkpoints_en/pretrain_best.pth \
  --data data/sft_sample_en.jsonl \
  --epochs 3 --batch-size 4 --grad-accum 1 \
  --amp auto --val-ratio 0.1
```

Resume with the same command (`sft_last.pth` is auto-detected) or
`--resume auto`. The config's block size is taken from the pretrained
checkpoint. The bundled 43-example sample is for pipeline validation; replace
it with a larger English instruction JSONL for a useful assistant. Splitting is
by prompt key, so duplicate prompts do not leak across train/validation.

## 9. Chat

```bash
# one-shot, no network
python -m english.chat --prompt "Explain attention in one paragraph." --no-web

# streaming chat with one GPU and live web search available
python -m english.chat --device cuda

# force a web lookup this turn and show sources
python -m english.chat --prompt "/web latest Python release" --show-sources
```

In a notebook, prefer one-shot prompts or a small Python loop using
`english.chat.load_chat(...).reply(...)`; `input()` works in Kaggle's
interactive session but not in committed runs.

## 10. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Distributed training requires CUDA/NCCL` | Enable GPU T4 ×2; `torchrun` must run inside the GPU session. |
| `CUDA out of memory` | `--batch-size 4 --grad-accum 8`; close other notebooks; restart the session. |
| `NCCL` timeout at startup | Re-run the cell; make sure only one `torchrun` is active and both GPUs are visible (`torch.cuda.device_count() == 2`). |
| `Tokenizer not found` | Run the tokenizer cell; check `--tokenizer`/`--tokenizer-dir`. |
| `Tokenizer mismatch` on resume | You retrained the tokenizer. Restore the original one or start a new run; token ids are not interchangeable. |
| Resume says `data=restart` | World size changed. Use the same GPU count, or accept re-reading with `--allow-data-restart`. |
| `Validation stream produced only N tokens` | Rare hash-split imbalance; raise `--val-fraction` slightly or lower `--val-tokens`. |
| Web search returns nothing | Internet is off, DuckDuckGo is rate-limiting, or the query found no pages. Chat falls back to local RAG. |
| Loss becomes `nan` | The scaler skips bad steps automatically; if it persists, lower `--lr`, check the data, and resume from the last finite checkpoint. |
| Session ends mid-run | Nothing is lost if `--save-every` fired; otherwise the last saved step is used. Resume with the same command. |

## 11. Exact command summary

```bash
pip install -r requirements.txt

python -m english.tokenizer --train --sample-mb 200 --vocab-size 32768

python -m english.train --smoke-test --max-steps 4
python -m english.sft   --smoke-test --max-steps 2
python -m english.chat  --smoke-test --prompt "Hello" --no-rag --no-web

torchrun --nproc_per_node=2 -m english.train \
  --preset base --plan-tokens 500000000 \
  --batch-size 8 --grad-accum 4 --amp fp16 \
  --expect-params 124278528 \
  --eval-every 250 --save-every 250 --log-every 10

python -m english.sft \
  --init checkpoints_en/pretrain_best.pth \
  --data data/sft_sample_en.jsonl \
  --epochs 3 --batch-size 4 --amp auto

python -m english.chat --prompt "Explain what a transformer is." --no-web
python -m unittest discover -s english/tests -t . -v
```

All artifacts stay under `tokenizer_en/`, `checkpoints_en/`, `data_en/`,
`runs_en/` and `memory_en/`. The repository is English-only.
