# Turkish ASR: fine-tuned Whisper-small vs zero-shot Whisper-large-v3

Where does a LoRA fine-tuned Whisper-small beat zero-shot Whisper-large-v3 on
Turkish speech — and where does it lose? This repo contains the full pipeline
to answer that with sliced error analysis instead of a single headline WER:
data preparation, LoRA fine-tuning, evaluation (raw *and* Turkish-normalized
WER/CER, RTF, memory), and per-slice breakdowns.

See [PLAN.md](PLAN.md) for the step-by-step plan and
[RESULTS.md](RESULTS.md) for results — including an explicit account of which
numbers could and could not be produced in the environment this repo was
built in.

## Problem

Whisper-large-v3 transcribes Turkish reasonably well zero-shot but is heavy
(~1.5B parameters). Whisper-small is 8× smaller; on-target fine-tuning with
LoRA might close the accuracy gap in-domain. The interesting question is not
the aggregate WER but *where* each wins: short vs long utterances, in-domain
(Common Voice) vs out-of-domain (FLEURS), speaker groups, and utterances
exercising Turkish-specific orthography (ğ ı ş ç ö ü).

Compared systems, all evaluated on identical test sets:

1. `openai/whisper-small`, zero-shot
2. `openai/whisper-small` + LoRA fine-tune on Common Voice 17 Turkish
3. `openai/whisper-large-v3`, zero-shot
4. `openai/whisper-tiny`, fine-tuned — the cheap-baseline control
   (distil-whisper variants are English-only, so tiny is the honest control)

## Data

- **mozilla-foundation/common_voice_17_0**, config `tr` — training,
  validation and in-domain test. Gated on the Hugging Face Hub: accept the
  terms there, then `export HF_TOKEN=...` before preparing it.
- **google/fleurs**, config `tr_tr` — **held out**: used only as an
  out-of-domain test set, never trained on. Two interchangeable sources:
  the Hub, or the original FLEURS release tarball on Google Cloud Storage
  (`source: gcs` in the config, no authentication needed).

All audio is resampled to 16 kHz. Manifests and per-model log-mel features
are cached under `cache/` keyed by dataset/split/sample-cap/seed and feature
parameters, so reruns are cheap.

## Layout

```
configs/            YAML configs (everything seeded, no hardcoded paths)
src/turkish_asr/
  data.py           dataset download, resampling, manifest + feature caches
  train.py          LoRA (PEFT) or full fine-tuning via Seq2SeqTrainer
  evaluate.py       WER/CER (raw + normalized), RTF, peak memory, predictions
  analyze.py        sliced analysis, error mining, data stats, markdown emit
  benchmark.py      random-weight architecture RTF benchmark (no accuracy)
  normalize.py      Turkish-aware normalizer (dotted/dotless i done right)
  metrics.py        jiwer-based corpus metrics, duration buckets
  debug_model.py    fully local tiny Whisper for hermetic smoke tests
tests/              pytest suite incl. 5-sample end-to-end smoke test
runs/               one JSON log per run (config, env, metrics, or the error)
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # pick the right torch build for your box
pip install -e .

# 0. Smoke test first — hermetic, no downloads. Do not train if this fails.
pytest

# 1. Data (FLEURS needs no auth; Common Voice needs HF_TOKEN)
turkish-asr-data --config configs/default.yaml --dataset fleurs --split test
turkish-asr-data --config configs/default.yaml --dataset common_voice --split test
turkish-asr-analyze --config configs/default.yaml --data-stats --dataset fleurs --split test

# 2. Zero-shot baselines (identical test sets for every system)
turkish-asr-evaluate --config configs/default.yaml --model openai/whisper-small    --dataset common_voice --split test
turkish-asr-evaluate --config configs/default.yaml --model openai/whisper-small    --dataset fleurs --split test
turkish-asr-evaluate --config configs/default.yaml --model openai/whisper-large-v3 --dataset common_voice --split test
turkish-asr-evaluate --config configs/default.yaml --model openai/whisper-large-v3 --dataset fleurs --split test

# 3. Fine-tunes (LoRA small + full tiny control)
turkish-asr-train --config configs/train_small_lora.yaml
turkish-asr-train --config configs/train_tiny_full.yaml

# 4. Evaluate the fine-tunes
turkish-asr-evaluate --config configs/default.yaml --model openai/whisper-small \
    --adapter models/whisper-small-lora-cv17-tr/adapter --dataset common_voice --split test
turkish-asr-evaluate --config configs/default.yaml --model openai/whisper-small \
    --adapter models/whisper-small-lora-cv17-tr/adapter --dataset fleurs --split test
turkish-asr-evaluate --config configs/default.yaml --model models/whisper-tiny-full-cv17-tr/model \
    --dataset common_voice --split test

# 5. Sliced analysis + report tables
turkish-asr-analyze --config configs/default.yaml --runs runs/<eval-run-id> [runs/<eval-run-id> ...] \
    --emit-markdown reports/analysis.md --examples 10
```

Every run writes `runs/<run_id>/run.json` (resolved config, environment, git
commit, metrics — or the error when it fails) plus `predictions.jsonl` for
evaluations. Any key is overridable without editing files, e.g.
`--set train.max_steps=2000 --set seed=7`.

CPU-only debugging: add `--max-samples 100 --batch-size 1 --threads 4` to
evaluations, and `--set train.max_train_samples=500` to training.

## Hardware

Targets a single ≥16 GB GPU (T4-class) for training; LoRA on whisper-small
needs ~6 GB. Everything degrades gracefully to CPU with sample caps.

The environment this repo was initially built and validated in had **no GPU**
(4-core CPU, 15 GB RAM) and an egress allowlist **blocking huggingface.co**
(PyPI, GitHub and storage.googleapis.com were open). That is why FLEURS
supports the GCS source, why the smoke test builds its own local tiny
Whisper instead of downloading one, and why RESULTS.md reports real numbers
only for what could actually run there (data statistics, architecture-level
CPU RTF/memory benchmarks, pipeline validation) while listing every blocked
run with its logged error. Rerunning the commands above on a GPU machine
with Hub access fills in the rest.

## Limitations

- Common Voice sentences are short read speech with self-reported, sparse
  speaker metadata; slices with thin coverage are reported with their n and
  should not be over-interpreted.
- FLEURS tr metadata is unreliable for speaker slicing: in the tr_tr
  release every dev/test row carries the same gender label (see RESULTS.md).
- The Turkish normalizer strips punctuation and case; numbers are left
  as-is, so digit-vs-word mismatches ("50" vs "elli") count as errors in
  both raw and normalized scoring.
- `benchmark.py` RTF numbers are architecture-level (random weights, pinned
  decode length): valid for speed/memory comparisons, meaningless for
  accuracy, and labeled as such wherever cited.
- WER/CER are computed with jiwer at corpus level (error-weighted, not
  averaged per utterance).
