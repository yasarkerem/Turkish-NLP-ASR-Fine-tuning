"""Data pipeline: download, resample to 16 kHz, cache manifests and features.

Two dataset families are supported:

- ``common_voice``: mozilla-foundation/common_voice_17_0 config ``tr`` from
  the Hugging Face Hub. The dataset is gated: accept the terms on the Hub and
  export ``HF_TOKEN`` before preparing it.
- ``fleurs``: google/fleurs config ``tr_tr``. Either from the Hub
  (``source: hf``) or from the original Google Cloud Storage tarball
  (``source: gcs``), which is byte-identical audio and requires no
  authentication. FLEURS is held out as an out-of-domain test set and is
  never used for training.

Everything lands in one manifest schema so downstream code does not care
where audio came from::

    utt_id, audio (16 kHz), reference_raw, duration_s,
    dataset, split, age, gender, accents, speaker_id

Manifests and per-model input features are cached under ``paths.cache_root``
keyed by dataset/split/sample-cap/seed (and feature parameters), so reruns
are cheap.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tarfile
from pathlib import Path

from turkish_asr.config import get_path, load_config, seed_everything

MANIFEST_COLUMNS = [
    "utt_id",
    "reference_raw",
    "duration_s",
    "dataset",
    "split",
    "age",
    "gender",
    "accents",
    "speaker_id",
]

_FLEURS_SPLIT_FILES = {"train": "train.tsv", "validation": "dev.tsv", "test": "test.tsv"}


def _download(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url} -> {dest}", file=sys.stderr)
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)


def ensure_fleurs_gcs(cfg: dict) -> Path:
    """Download and extract the FLEURS tr_tr tarball if not present."""
    ds_cfg = cfg["datasets"]["fleurs"]
    raw_root = get_path(cfg, "data_root", "data") / "raw" / "fleurs"
    lang_dir = raw_root / "tr_tr"
    complete = all(
        (lang_dir / f).exists() and (lang_dir / "audio" / f.split(".")[0]).is_dir()
        for f in _FLEURS_SPLIT_FILES.values()
    )
    if complete:
        return lang_dir
    tarball = raw_root / "tr_tr.tar.gz"
    if not tarball.exists():
        _download(ds_cfg["gcs_url"], tarball)
    print(f"extracting {tarball}", file=sys.stderr)
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(raw_root)
    return lang_dir


def _read_fleurs_tsv(lang_dir: Path, split: str) -> list[dict]:
    """Parse a headerless FLEURS TSV.

    Columns: id, filename, raw_transcription, transcription,
    char-tokenized transcription, num_samples, gender.
    """
    tsv = lang_dir / _FLEURS_SPLIT_FILES[split]
    audio_dir = lang_dir / "audio" / _FLEURS_SPLIT_FILES[split].split(".")[0]
    rows = []
    with open(tsv, newline="", encoding="utf-8") as f:
        for rec in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(rec) < 7:
                continue
            utt_id, fname, raw_tr, _norm_tr, _chars, num_samples, gender = rec[:7]
            path = audio_dir / fname
            if not path.exists():
                continue
            rows.append(
                {
                    "utt_id": f"fleurs-{split}-{utt_id}",
                    "audio": str(path),
                    "reference_raw": raw_tr.strip(),
                    "duration_s": int(num_samples) / 16000.0,
                    "dataset": "fleurs",
                    "split": split,
                    "age": None,
                    "gender": gender.strip().lower() or None,
                    "accents": None,
                    "speaker_id": None,
                }
            )
    return rows


def load_fleurs(cfg: dict, split: str):
    from datasets import Audio, Dataset

    ds_cfg = cfg["datasets"]["fleurs"]
    sr = cfg["audio"]["sampling_rate"]
    if ds_cfg.get("source", "gcs") == "gcs":
        lang_dir = ensure_fleurs_gcs(cfg)
        rows = _read_fleurs_tsv(lang_dir, split)
        if not rows:
            raise RuntimeError(f"no FLEURS rows found for split {split!r} in {lang_dir}")
        ds = Dataset.from_list(rows)
    else:
        from datasets import load_dataset

        raw = load_dataset(
            ds_cfg["hf_id"],
            ds_cfg["hf_config"],
            split=split,
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )

        def to_manifest(ex, idx):
            return {
                "utt_id": f"fleurs-{split}-{ex['id']}",
                "audio": ex["audio"],
                "reference_raw": ex["raw_transcription"].strip(),
                "duration_s": ex["num_samples"] / 16000.0,
                "dataset": "fleurs",
                "split": split,
                "age": None,
                "gender": {0: "male", 1: "female"}.get(ex.get("gender"), None),
                "accents": None,
                "speaker_id": None,
            }

        ds = raw.map(to_manifest, with_indices=True, remove_columns=raw.column_names)
    return ds.cast_column("audio", Audio(sampling_rate=sr))


def load_common_voice(cfg: dict, split: str):
    from datasets import Audio, load_dataset

    ds_cfg = cfg["datasets"]["common_voice"]
    sr = cfg["audio"]["sampling_rate"]
    token = os.environ.get("HF_TOKEN")
    raw = load_dataset(
        ds_cfg["hf_id"],
        ds_cfg["hf_config"],
        split=split,
        trust_remote_code=True,
        token=token,
    )
    raw = raw.cast_column("audio", Audio(sampling_rate=sr))

    def to_manifest(ex, idx):
        audio = ex["audio"]
        return {
            "utt_id": f"cv-{split}-{idx:07d}",
            "audio": audio,
            "reference_raw": (ex.get("sentence") or "").strip(),
            "duration_s": len(audio["array"]) / audio["sampling_rate"],
            "dataset": "common_voice",
            "split": split,
            "age": (ex.get("age") or "").strip() or None,
            "gender": (ex.get("gender") or "").strip() or None,
            "accents": (ex.get("accents") or "").strip() or None,
            "speaker_id": ex.get("client_id"),
        }

    ds = raw.map(to_manifest, with_indices=True, remove_columns=raw.column_names)
    ds = ds.filter(lambda ex: len(ex["reference_raw"]) > 0)
    return ds.cast_column("audio", Audio(sampling_rate=sr))


_LOADERS = {"fleurs": load_fleurs, "common_voice": load_common_voice}


def manifest_cache_dir(cfg: dict, dataset: str, split: str, max_samples: int | None) -> Path:
    cap = f"n{max_samples}" if max_samples else "all"
    seed = cfg.get("seed", 42)
    return get_path(cfg, "cache_root", "cache") / "manifests" / f"{dataset}_{split}_{cap}_seed{seed}"


def prepare_manifest(
    cfg: dict, dataset: str, split: str, max_samples: int | None = None, force: bool = False
):
    """Build (or load from cache) the unified manifest for a dataset split."""
    from datasets import load_from_disk

    cache = manifest_cache_dir(cfg, dataset, split, max_samples)
    if cache.exists() and not force:
        return load_from_disk(str(cache))
    ds = _LOADERS[dataset](cfg, split)
    if max_samples and max_samples < len(ds):
        ds = ds.shuffle(seed=cfg.get("seed", 42)).select(range(max_samples))
    cache.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(cache))
    return load_from_disk(str(cache))


def build_features(manifest, processor, max_label_length: int = 448):
    """Map a manifest to model inputs: log-mel ``input_features`` + ``labels``.

    The tokenizer's prefix tokens (language/task) must already be set on the
    processor. Returns a dataset with exactly two columns; dtype checks are in
    the test suite.
    """

    def featurize(ex):
        audio = ex["audio"]
        features = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        labels = processor.tokenizer(ex["reference_raw"]).input_ids
        return {"input_features": features, "labels": labels}

    features = manifest.map(featurize, remove_columns=manifest.column_names)
    return features.filter(lambda ex: len(ex["labels"]) <= max_label_length)


def feature_cache_dir(
    cfg: dict, dataset: str, split: str, max_samples: int | None, model_slug: str, n_mels: int
) -> Path:
    cap = f"n{max_samples}" if max_samples else "all"
    seed = cfg.get("seed", 42)
    return (
        get_path(cfg, "cache_root", "cache")
        / "features"
        / model_slug
        / f"{dataset}_{split}_{cap}_seed{seed}_mel{n_mels}"
    )


def prepare_features(
    cfg: dict,
    dataset: str,
    split: str,
    processor,
    model_slug: str,
    max_samples: int | None = None,
    force: bool = False,
):
    """Cached version of :func:`build_features`, keyed by model feature params."""
    from datasets import load_from_disk

    n_mels = processor.feature_extractor.feature_size
    cache = feature_cache_dir(cfg, dataset, split, max_samples, model_slug, n_mels)
    if cache.exists() and not force:
        return load_from_disk(str(cache))
    manifest = prepare_manifest(cfg, dataset, split, max_samples)
    features = build_features(manifest, processor)
    cache.parent.mkdir(parents=True, exist_ok=True)
    features.save_to_disk(str(cache))
    return load_from_disk(str(cache))


def manifest_stats(manifest) -> dict:
    """Descriptive statistics used by ``analyze --data-stats``."""
    import numpy as np

    from turkish_asr.metrics import DURATION_BUCKET_ORDER, duration_bucket
    from turkish_asr.normalize import contains_turkish_specific

    durations = np.array(manifest["duration_s"], dtype=np.float64)
    refs = manifest["reference_raw"]
    buckets = {b: 0 for b in DURATION_BUCKET_ORDER}
    for d in durations:
        buckets[duration_bucket(float(d))] += 1
    genders: dict[str, int] = {}
    for g in manifest["gender"]:
        genders[g or "unknown"] = genders.get(g or "unknown", 0) + 1
    n_turkish = sum(1 for r in refs if contains_turkish_specific(r))
    return {
        "n_utterances": len(manifest),
        "total_audio_hours": round(float(durations.sum()) / 3600, 3),
        "duration_s_mean": round(float(durations.mean()), 2),
        "duration_s_median": round(float(np.median(durations)), 2),
        "duration_s_p95": round(float(np.percentile(durations, 95)), 2),
        "duration_buckets": buckets,
        "gender_counts": genders,
        "n_refs_with_turkish_specific_chars": n_turkish,
        "pct_refs_with_turkish_specific_chars": round(100.0 * n_turkish / max(len(refs), 1), 1),
        "ref_words_mean": round(float(np.mean([len(r.split()) for r in refs])), 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare dataset manifests and feature caches")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="K=V")
    parser.add_argument("--dataset", required=True, choices=sorted(_LOADERS))
    parser.add_argument("--split", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    parser.add_argument(
        "--features-for",
        metavar="MODEL",
        default=None,
        help="also build input features for this model id/path (needs its processor)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.get("seed", 42))

    manifest = prepare_manifest(cfg, args.dataset, args.split, args.max_samples, args.force)
    stats = manifest_stats(manifest)
    print(json.dumps({"manifest": str(manifest_cache_dir(cfg, args.dataset, args.split, args.max_samples)), "stats": stats}, indent=2, ensure_ascii=False))

    if args.features_for:
        from transformers import WhisperProcessor

        model_cfg = cfg.get("model", {})
        processor = WhisperProcessor.from_pretrained(
            args.features_for,
            language=model_cfg.get("language", "turkish"),
            task=model_cfg.get("task", "transcribe"),
        )
        slug = args.features_for.rstrip("/").split("/")[-1]
        feats = prepare_features(
            cfg, args.dataset, args.split, processor, slug, args.max_samples, args.force
        )
        print(json.dumps({"features_rows": len(feats), "model_slug": slug}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
