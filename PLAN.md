# Project plan: Turkish ASR fine-tuning and evaluation

Goal: fine-tune Whisper-small on Turkish Common Voice with LoRA and find out
where it beats zero-shot Whisper-large-v3 and where it does not, with sliced
error analysis rather than a single headline number.

## Step order

| # | Step | Output | Expected runtime (T4 16GB GPU) | Expected runtime (4-core CPU) |
|---|------|--------|-------------------------------|-------------------------------|
| 1 | Repo skeleton: configs, requirements, package layout | `src/`, `configs/` | — | minutes |
| 2 | Turkish text normalizer + unit tests | `src/turkish_asr/normalize.py` | — | minutes |
| 3 | Data pipeline: download, 16 kHz resample, feature cache | `data.py`, cached arrow features | 30–60 min (download-bound) | same (download-bound) |
| 4 | Preprocessing tests (shape/dtype) | `tests/test_preprocessing.py` | <1 min | <1 min |
| 5 | 5-sample smoke test of full train→eval loop | `tests/test_smoke.py` | ~2 min | ~5 min |
| 6 | Zero-shot eval: whisper-small on CV test + FLEURS test | `runs/*.json` | ~40 min (2 test sets, ~10k utts capped) | days — cap to a few hundred utts |
| 7 | Zero-shot eval: whisper-large-v3 on both test sets | `runs/*.json` | ~2.5 h | not feasible beyond ~50 utts |
| 8 | LoRA fine-tune whisper-small on CV tr train (~35k utts, 1–2 epochs) | adapter + `runs/*.json` | 6–10 h | not feasible |
| 9 | Fine-tune whisper-tiny (cheap-baseline control) | adapter + `runs/*.json` | 1.5–3 h | ~24 h on a small subset |
| 10 | Eval both fine-tuned models on both test sets | `runs/*.json` | ~1.5 h | days |
| 11 | RTF benchmark (CPU and GPU, ≥50 utts) + size/memory audit | `runs/*.json` | ~30 min | ~1–2 h |
| 12 | Sliced analysis: duration buckets, speaker metadata, domain shift, Turkish-character slices | `analyze.py` output, RESULTS.md tables | minutes | minutes |
| 13 | Error-pair mining (10 examples with commentary) | RESULTS.md section | minutes | minutes |
| 14 | Write RESULTS.md + README.md from run logs | docs | — | minutes |

Rule enforced throughout: the smoke test (step 5) must pass before any real
training run is launched, and every run writes its config + metrics to a JSON
under `runs/` so every number in RESULTS.md is traceable to a run log.

## Execution environments

The project targets a single ≥16 GB GPU (LoRA keeps whisper-small trainable
well under that; whisper-large-v3 is inference-only here). Everything is also
runnable on CPU with reduced sample caps for debugging.

The environment used for the initial build of this repo has no GPU (4 CPU
cores, 15 GB RAM) and sits behind an egress allowlist that permits PyPI,
GitHub and `storage.googleapis.com` but blocks `huggingface.co`. Consequences
for the initial commit, recorded here so the numbers in RESULTS.md are
interpreted correctly:

- FLEURS Turkish is fetched from its original Google Cloud Storage bucket
  (`xtreme_translations/FLEURS102/tr_tr.tar.gz`), so real out-of-domain audio
  and reference statistics are available.
- Common Voice 17 is doubly unavailable here: the host is blocked and the
  dataset is gated (requires accepting terms with an HF account even on an
  open network). The data code supports it and is exercised by unit tests on
  synthetic examples, but no CV-derived number can be produced in this
  environment.
- No pretrained Whisper checkpoint can be downloaded here, so no WER/CER for
  the four comparison systems can be produced in this environment. Steps
  6–10 are implemented, smoke-tested end-to-end with a locally constructed
  tiny random-weight Whisper model, and must be executed on a machine with
  Hugging Face access to fill in RESULTS.md. RESULTS.md states explicitly
  which numbers exist and which are blocked; it contains no estimated
  figures.
- RTF (step 11) is still measured for real on CPU, as an architecture-level
  benchmark at a fixed decode length with random weights — trace timing
  depends on architecture shape, not weight values, once output length is
  pinned. GPU RTF is reported as not measured (no GPU present).

## Design decisions

- LoRA (PEFT) on attention projections of whisper-small; rank/alpha in YAML.
  Full fine-tuning is out of scope by design (VRAM and time budget).
- `openai/whisper-tiny` fine-tune is the cheap-baseline control
  (distil-whisper has no Turkish-capable multilingual small variant;
  distil-large-v3 is English-only, so tiny is the honest control).
- WER/CER via `jiwer`, both raw and Turkish-normalized. The normalizer
  handles the dotted/dotless i pair explicitly (İ→i, I→ı) before casefold,
  strips punctuation, collapses whitespace. Both numbers are always reported.
- Duration buckets: <3 s, 3–6 s, 6–10 s, >10 s.
- Speaker slices from CV metadata (age, gender, accent) with per-slice
  coverage counts reported, since CV metadata is sparse and self-reported.
- Turkish-character slice: utterances whose reference contains any of
  ğ ı ş ç ö ü İ Ğ Ş Ç Ö Ü vs. those that do not, plus per-character error
  attribution on the character level.
- All randomness seeded from a single `seed` key in the YAML config; all
  paths come from config or CLI flags.
