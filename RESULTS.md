# Results

Integrity rule for this file: **every number below comes from a script that
actually ran in this repository**, and each is traceable to a JSON log under
`runs/` (run ids are cited inline). Where a run could not be executed, the
table says so and points at the failed run's log instead of an estimate.
No placeholder or extrapolated figures appear anywhere in this file.

## Execution environment for these results

Everything below was produced on a **4-core Intel Xeon @ 2.10 GHz, 15 GB
RAM, no GPU**, behind an egress allowlist that permits PyPI, GitHub and
`storage.googleapis.com` but **blocks `huggingface.co`** (the proxy returns a
policy denial on CONNECT). Consequences:

- FLEURS Turkish was obtained from the original Google Cloud Storage release
  tarball — real audio, real statistics below.
- Common Voice 17 `tr` could not be obtained: the Hub is blocked, and the
  dataset is additionally gated (requires accepting terms with an HF
  account). Attempt logged in `20260802T165134Z_datastats_common_voice-test`
  (`ConnectionError: Couldn't reach 'mozilla-foundation/common_voice_17_0'`).
- No pretrained Whisper checkpoint could be downloaded, so **no WER/CER for
  the four comparison systems exists yet**. Each attempt is logged (see the
  headline table). The full pipeline is implemented and smoke-tested
  end-to-end; running the commands in README.md on a machine with Hub access
  fills in every blocked cell of this file.
- GPU RTF is reported as not measured (no GPU present).

## Headline comparison

All four systems, both test sets. Cells contain measured values or the
logged reason none exists.

| System | CV-17 tr test WER/CER (raw and norm) | FLEURS tr test WER/CER (raw and norm) |
|---|---|---|
| whisper-small zero-shot | not run — Common Voice unavailable here (gated + host blocked) | not run — checkpoint download blocked; attempt: `20260802T165146Z_eval_whisper-small_fleurs-test_attempt` (ProxyError to huggingface.co) |
| whisper-small + LoRA on CV-17 tr | not run — training blocked; attempt: `20260802T165320Z_train_whisper-small-lora-cv17-tr` (ProxyError to huggingface.co) | not run — same blocked training |
| whisper-large-v3 zero-shot | not run — Common Voice unavailable here | not run — checkpoint download blocked; attempt: `20260802T165215Z_eval_whisper-large-v3_fleurs-test_attempt` (ProxyError) |
| whisper-tiny fine-tuned (control) | not run — Common Voice unavailable here | not run — checkpoint download blocked; attempt: `20260802T165243Z_eval_whisper-tiny_fleurs-test_attempt` (ProxyError) |

There is deliberately no accuracy number in this table. Writing one would
have required inventing it.

## What was measured

### Out-of-domain test set: FLEURS tr_tr statistics

Source: original FLEURS102 GCS tarball, resampled/verified at 16 kHz.
Runs: `20260802T165007Z_datastats_fleurs-test`,
`20260802T165116Z_datastats_fleurs-validation`,
`20260802T165121Z_datastats_fleurs-train`.

| split | utterances | audio hours | mean dur (s) | <3s | 3–6s | 6–10s | >10s | refs with ğ/ı/ş/ç/ö/ü | mean ref words |
|---|---|---|---|---|---|---|---|---|---|
| test | 743 | 2.606 | 12.62 | 0 | 20 | 163 | 560 | 99.2% | 17.3 |
| validation | 338 | 1.121 | 11.93 | 0 | 15 | 86 | 237 | 100.0% | 16.0 |
| train (unused; held out) | 2526 | 8.300 | 11.83 | 0 | 86 | 833 | 1607 | 100.0% | 16.4 |

Two observations that matter for the sliced analysis design:

- **FLEURS is long-utterance domain**: 75% of test utterances exceed 10 s
  and none is under 3 s, while Common Voice sentences are mostly under 6 s.
  The duration-bucket slice therefore doubles as a domain-shift probe, and
  the <3 s bucket can only be populated by Common Voice test data.
- **Turkish-specific characters are ubiquitous**: 99.2% of FLEURS test
  references contain at least one of ğ ı ş ç ö ü. The with/without slice on
  FLEURS will be dominated by the "with" side (737 vs 6 utterances); the
  per-character error attribution in `analyze.py` is the informative view
  there, and the with/without split is only well-powered on Common Voice.

### Speaker metadata coverage (reported honestly)

- **FLEURS tr gender labels are degenerate in eval splits**: every one of
  the 743 test rows and 338 dev rows carries the label `MALE`, while the
  train split has 1753 `FEMALE` / 773 `MALE`. Verified directly against the
  release TSVs. Gender slicing on FLEURS test is therefore meaningless and
  is not reported; treat the field as unreliable in this release.
- **Common Voice metadata** (age, gender, accents) could not be inspected
  here (dataset unavailable). `analyze.py` reports per-field coverage next
  to every slice, and slices below n=20 are suppressed by default, so sparse
  self-reported metadata cannot silently masquerade as a finding.

### Model size, memory and CPU real-time factor (architecture-level)

Runs: `20260802T165355Z_benchmark_tiny_cpu`,
`20260802T165418Z_benchmark_small_cpu`,
`20260802T171318Z_benchmark_large-v3_cpu`.

These are **random-weight architecture benchmarks**: inference wall-time
depends on architecture shape, input length and decoder step count, not on
weight values, so decoder steps are pinned at 3.2 per audio second (~real
transcript rate) and real FLEURS test audio drives the encoder. Valid for
speed/size/memory comparison; says nothing about accuracy; WER from random
weights is deliberately not computed. Measured on 4 CPU threads, fp32,
batch size 1, seeded utterance sample, warmup excluded.

| architecture | params | fp32 size (MiB) | fp16 size (MiB) | n utts | audio (s) | RTF overall | RTF median | RTF p90 | peak RSS (MiB) |
|---|---|---|---|---|---|---|---|---|---|
| tiny | 37.8 M | 144 | 72 | 60 | 749.9 | 0.024 | 0.024 | 0.032 | 1507 |
| small | 241.7 M | 922 | 461 | 60 | 749.9 | 0.163 | 0.162 | 0.219 | 2479 |
| large-v3 | 1543.5 M | 5888 | 2944 | 50 | 631.1 | 0.894 | 0.906 | 1.210 | 8263 |

GPU RTF: **not measured** — no GPU in this environment.

Parameter counts line up with the published checkpoints (tiny 37,760,640;
small 241,734,912; large-v3 1,543,490,560 with the output projection tied
to the token embedding, as in the real model).

What this already establishes for the deployment question: on a modest
4-core CPU, whisper-small transcribes ~6.1× faster than real time while
large-v3 barely keeps up with real time (RTF 0.894 overall, p90 above 1.0 —
i.e. the slowest tenth of utterances fall behind live audio), and the fp16
weight footprint differs by 6.4× (461 MiB vs 2944 MiB) with an 8.1 GiB peak
RSS for fp32 large-v3 inference. If the fine-tuned small model matches
large-v3 in-domain, the efficiency case for it is not subtle.

### Pipeline validation on real FLEURS audio

To prove the full loop on real data despite the checkpoint block, a locally
constructed tiny random-weight Whisper (local BPE tokenizer, ~0.5 M
parameters; see `debug_model.py`) went through the production code path
end to end:

- `20260802T171242Z_train_pipeline-validation`: LoRA training (r=2, 3
  optimizer steps) on 5 real FLEURS **train**-split utterances — feature
  cache built from the real TSVs and 16 kHz wavs, adapter saved.
- `20260802T171256Z_eval_pipeline-validation`: evaluation of the merged
  adapter on 40 real FLEURS **test** utterances: batched generation,
  predictions.jsonl, raw+normalized WER/CER, peak-RSS tracking.
- `reports/pipeline_validation.md`: the sliced analysis (duration buckets,
  metadata coverage, Turkish-character slices, per-character attribution,
  worst pairs) generated from that run by `analyze.py`.

Measured result: WER_raw = WER_norm = CER = 1.0 (100% error) at 1.3 GiB
peak RSS. **That number is expected and carries no accuracy information**
— a randomly initialized 2-layer model with 3 training steps cannot
transcribe; it is quoted only as evidence that the pipeline executes,
computes metrics, and slices correctly on real Turkish audio. The identical
commands with `openai/whisper-small` produce the real numbers on an
unrestricted machine.

## Sliced analysis

The slicing machinery (duration buckets, metadata with coverage,
in-domain vs out-of-domain, Turkish-character presence, per-character error
attribution) is implemented in `analyze.py`, unit- and smoke-tested, and
demonstrated on the pipeline-validation run above. **Accuracy-bearing slice
tables require the blocked evaluation runs and are therefore absent.** The
data-side slice structure that could be measured (FLEURS duration and
character composition, metadata degeneracy) is reported above.

## Ten example error pairs

**Not produced.** Error pairs require hypotheses from a real model; no real
model could be loaded in this environment. The mining is implemented
(`analyze.py --examples 10` ranks utterances by normalized WER with a
minimum-length filter and flags Turkish-character involvement) and runs
against any completed evaluation. Inventing illustrative pairs would violate
the integrity rule at the top of this file.

## Where the fine-tuned small model loses to zero-shot large

This section exists to be written from measurements, and the measurements
are blocked in this environment — so here is, honestly, what is and is not
known:

**Known from runs in this repo:** nothing about the accuracy comparison.
No cell of the headline table was measured; no claim of the form
"fine-tuned small wins in-domain" or "large wins out-of-domain" is made
here, in the README, or anywhere else in this repository.

**How this section gets its content** (the exact procedure, so a rerun on
an unrestricted machine produces it mechanically): run the eight
evaluation commands in README.md, then
`turkish-asr-analyze --runs <the six eval run dirs> --emit-markdown ...`.
The comparison surfaces where the LoRA model's normalized WER exceeds
zero-shot large-v3's, sliced by: FLEURS vs Common Voice (domain
transfer), duration bucket (long-form behavior), metadata slices with
adequate coverage, and Turkish-character attribution. Whatever those
tables say — including a clean sweep for large-v3 — gets written here
with the run ids.

**Expectations that motivated the slice design — labeled as expectations,
not findings:** fine-tuning on Common Voice (short, read, crowd-recorded
speech) is expected to help most on in-domain short utterances and Turkish
orthography, and to help least — possibly hurt, via catastrophic forgetting
of general robustness — on FLEURS's longer out-of-domain utterances, where
zero-shot large-v3's scale advantage typically shows. The duration and
domain slices exist precisely to catch that failure mode; the FLEURS
statistics above (75% of utterances >10 s, zero <3 s) show the out-of-domain
set genuinely stresses it. Until the blocked runs execute, these remain
hypotheses.

## Run ledger

Every run in `runs/`, including failures:

| run id | kind | status | note |
|---|---|---|---|
| 20260802T164949Z_datastats_fleurs-test | datastats | failed | code bug (half-extracted tarball accepted); fixed in `data.py`, superseded by the next run |
| 20260802T165007Z_datastats_fleurs-test | datastats | completed | FLEURS test statistics |
| 20260802T165116Z_datastats_fleurs-validation | datastats | completed | FLEURS dev statistics |
| 20260802T165121Z_datastats_fleurs-train | datastats | completed | FLEURS train statistics (held out, never trained on) |
| 20260802T165134Z_datastats_common_voice-test | datastats | failed | huggingface.co blocked by egress policy |
| 20260802T165146Z_eval_whisper-small_fleurs-test_attempt | eval | failed | checkpoint download blocked (ProxyError) |
| 20260802T165215Z_eval_whisper-large-v3_fleurs-test_attempt | eval | failed | checkpoint download blocked (ProxyError) |
| 20260802T165243Z_eval_whisper-tiny_fleurs-test_attempt | eval | failed | checkpoint download blocked (ProxyError) |
| 20260802T165320Z_train_whisper-small-lora-cv17-tr | train | failed | checkpoint download blocked (ProxyError) |
| 20260802T165355Z_benchmark_tiny_cpu | benchmark | completed | CPU RTF/memory, architecture-level |
| 20260802T165418Z_benchmark_small_cpu | benchmark | completed | CPU RTF/memory, architecture-level |
| 20260802T170145Z_benchmark_large-v3_cpu | benchmark | completed | superseded: meta-device init left the output projection untied (+66.4 M params, +0.25 GiB); timing agrees with the rerun |
| 20260802T171242Z_train_pipeline-validation | train | completed | debug-model LoRA train on 5 real FLEURS train utterances |
| 20260802T171256Z_eval_pipeline-validation | eval | completed | debug-model eval on 40 real FLEURS test utterances (WER 1.0 expected; no accuracy meaning) |
| 20260802T171318Z_benchmark_large-v3_cpu | benchmark | completed | CPU RTF/memory, architecture-level, tied weights — the row cited above |

Two additional large-v3 benchmark launches were terminated before producing
metrics (they overlapped and contended for the 4 CPU cores, which would have
corrupted the timings) and their empty run directories were removed; the
surviving runs above were each measured on an otherwise idle machine.

The pytest suite (46 tests: Turkish normalizer including İ/I/ı/i edge cases,
preprocessing shapes/dtypes, config handling, and the 5-sample end-to-end
train→eval→analyze smoke test) passed before any run above was launched, as
required by the plan.

## Regenerating this file's blocked cells

On a machine with a GPU and Hugging Face access:

```bash
pytest                                   # gate: must pass first
# then the eight evaluate/train commands from README.md, then:
turkish-asr-analyze --config configs/default.yaml \
  --runs runs/<eval-run-1> ... runs/<eval-run-6> \
  --emit-markdown reports/analysis.md --examples 10
```

and replace the "not run" cells with the values from the new `runs/*/run.json`
files, citing their run ids as done throughout this file.
