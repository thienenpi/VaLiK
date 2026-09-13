# VaLiK reproduction — ScienceQA / Table 3

Minimal-cost reproduction of *Aligning Vision to Language* (ICCV 2025,
arXiv:2503.12972v3), targeting the four ScienceQA rows of Table 3.
Nothing under `src/` is modified. New code: `repro/` (Python), `jobs/*.slurm` (sbatch),
and `env.sh` / `setup.sh` / `submit_all.sh` at the repo root (run by hand, never submitted).

## Run it

```bash
# once, on the login node
bash setup.sh

# smoke run FIRST — first 1000 train + 1000 test pids, whole chain, ~1.5h
bash submit_all.sh 1000
python repro/report.py outputs/results        # check the parse rate, not the accuracy

# then the real thing, ~12-14h
bash submit_all.sh
python repro/report.py
```

Every job takes one GPU (the cluster caps jobs at two), uses `--requeue`, and
resumes where it stopped. Everything lives under `/media/lhbac29` — see `env.sh`.

`submit_all.sh N` sets `VALIK_LIMIT=N`, which reaches **all four** stages: caption,
prune, KG build and eval each slice the same first-N prefix of `pid_splits.json`, so
the smoke subset stays coherent. Smoke artefacts are tagged `-n<N>`
(`outputs/kg_text_image-n1000`, `outputs/results/nokg-n1000.json`) and can never
collide with — or skip — a full run.

A smoke run proves the plumbing: vLLM comes up, LightRAG extracts entities, answers
parse. It does **not** prove accuracy — a KG built from 1000 problems is far too
small for the paper's numbers. Read the parse rate, not the score.

## Pipeline

| Stage | Job | Paper | Wall-clock |
|---|---|---|---|
| Caption | `10_caption.slurm` (array 0–3) | §3.1 CoE | ~2.5h |
| Prune | `20_prune.slurm` (array 0–3) | §3.2 SV, τ=0.20 | ~20min |
| Build MMKG | `30_build_kg.slurm` (array 0–1) | §3.3 / Alg. 1 | 3–6h |
| Evaluate | `40_eval.slurm` (array 0–2) | Table 3 | ~1h |

## Deviations from the paper

| | Paper §4.1 | Here | Why |
|---|---|---|---|
| VLM cascade | BLIP-2 → LLaVA → Qwen2-VL | Qwen2-VL-7B alone | Appendix G: *"a single, strong VLM can achieve performance comparable to a cascade"* |
| Graph LLM | DeepSeek-R1-70B | Qwen2.5-32B-Instruct-AWQ | README L84 recommends Qwen2.5; §4.2 reports R1 anomalous; R1's `<think>` breaks LightRAG's delimiter parsing |
| Serving | Ollama | vLLM | no continuous batching in Ollama — the cause of Appendix F's 196k tok/h |
| Embedding | `nomic-embed-text` via Ollama | `nomic-ai/nomic-embed-text-v1.5` in-process | same 768-dim, no second server |
| `llm_model_max_async` | 160 | 32 | 160 was tuned for Ollama; vLLM thrashes above ~64 |
| Configs built | 4 rows | 3 (`nokg`, `image_only`, `text_image`) | enough to isolate the visual contribution |
| Retrieval, τ, hybrid mode, full train KB, full test split | — | **unchanged** | |

Everything else follows the paper.

## Upstream bugs fixed (in `repro/`, not in `src/`)

1. **`similarity_verification.py:16`** calls the CLIP processor with `padding=True`
   but no `truncation=True`. CLIP's text encoder caps at 77 tokens — the file's own
   comment on L81 says so — and one long Qwen2-VL sentence takes down the batch.
   `repro/prune.py` passes `truncation=True`.
2. **`similarity_verification.py`** writes an empty file when every sentence falls
   below τ, which silently deletes that image from an image-only KG.
   `repro/prune.py` keeps the original text and counts the event.
3. **`lightrag_ollama_demo.py:63`** inserts one giant string. LightRAG checkpoints per
   *document* (`lightrag.py:373-390`), so a preempted job loses everything.
   `repro/build_kg.py` inserts a list — one doc per problem, one per image.
4. **`lightrag/llm/openai.py:207`** sets `response_format="json"` (a bare string) for
   keyword extraction; the OpenAI schema and vLLM both want
   `{"type": "json_object"}` and reject the string with a 400. Both `repro/build_kg.py`
   and `repro/eval_scienceqa.py` wrap the completion func to fix this.
5. **`Qwen2VL_ScienceQA.py:65`** generates with `max_new_tokens=32768`; the paper's own
   throughput figure implies ~240. `repro/caption.py` uses 384.
6. **`lightrag_ollama_demo.py:21`** names `deepseek-r1:72b`, which is not an Ollama tag
   (the distill sizes are 1.5/7/8/14/32/70b). Not hit here since we use vLLM.

## What is *not* reproduced

The upstream repo ships **no evaluation code** — `grep -rniE
"accuracy|evaluate|f1_score|sklearn"` outside the vendored LightRAG returns nothing.
The prompt format, answer parsing and category grouping in `repro/eval_scienceqa.py`
are this reproduction's reading of §4.1, not the authors'. Absolute numbers therefore
carry an unknown offset.

**Judge the run on ordering, not decimals** (`repro/report.py` checks these):

- 74.72 `nokg` < 78.88 `image_only` < 83.16 `text_image`
- `text_image` (7B) > 78.37, native Qwen2.5-72B
- MMKG on disk ≈ 489 MB (§4.2)

Expect 1–3 points below the paper. A **parse rate under ~95%** means the prompt is
wrong, not the KG — fix that before reading any accuracy.

## Baseline fairness

The `nokg` row reads ScienceQA's own `captions.json`, not our Qwen2-VL descriptions.
Table 3 gives the un-augmented baseline **65.79** on the IMG column, far above chance,
so it is clearly fed some description; handing it our captions would raise the
baseline and shrink the gain VaLiK is credited with. Override with
`--caption-source valik` if you want that ablation.

## Free extras

`20_prune.slurm` keeps both `image.txt` and `image.pruned.txt`, so the Table 4
ablations cost no re-captioning:

```bash
sbatch jobs/30_build_kg.slurm .txt   # −SV: builds kg_image_only-nosv, kg_text_image-nosv
```

Table 4 has SV *reducing* ScienceQA image-only accuracy (NAT −0.92, LAN −1.28).
Reproducing that dip is evidence of a faithful implementation — don't tune τ away
from 0.20 to hide it.
