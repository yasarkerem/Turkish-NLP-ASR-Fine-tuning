"""Sliced analysis of evaluation runs.

Consumes ``runs/<id>/predictions.jsonl`` files produced by ``evaluate.py``
and breaks WER/CER down by:

- utterance duration bucket (<3s, 3-6s, 6-10s, >10s);
- speaker metadata where present (age, gender, accents), with per-field
  coverage reported so sparse metadata cannot masquerade as a solid slice;
- presence of Turkish-specific characters (ğ ı ş ç ö ü) in the reference,
  plus per-character error attribution from aligned normalized strings;
- dataset/domain (runs are per-dataset; cross-domain deltas come from
  comparing runs side by side).

Also has ``--data-stats`` for dataset descriptive statistics (logged under
runs/ like everything else) and ``--examples`` to mine the worst error pairs.
"""

from __future__ import annotations

import argparse
import difflib
import json
from collections import defaultdict
from pathlib import Path

from turkish_asr.config import get_path, load_config, seed_everything
from turkish_asr.metrics import (
    DURATION_BUCKET_ORDER,
    compute_pair_metrics,
    duration_bucket,
    utterance_wer,
)
from turkish_asr.normalize import TURKISH_SPECIFIC_CHARS, TurkishNormalizer, contains_turkish_specific
from turkish_asr.runlog import RunLog

LOWER_TURKISH_SPECIFIC = "çğıöşü"


def load_run(run_dir: str | Path) -> tuple[dict, list[dict]]:
    run_dir = Path(run_dir)
    with open(run_dir / "run.json") as f:
        run = json.load(f)
    records = []
    preds = run_dir / "predictions.jsonl"
    if preds.exists():
        with open(preds, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    return run, records


def _metrics_for(records: list[dict], normalizer: TurkishNormalizer) -> dict:
    m = compute_pair_metrics(
        [r["reference_raw"] for r in records],
        [r["hypothesis_raw"] for r in records],
        normalizer,
    )
    return m.as_dict()


def slice_by(records: list[dict], key_fn, normalizer, min_n: int = 1) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[key_fn(r)].append(r)
    out = {}
    for key, rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if key is None or len(rows) < min_n:
            continue
        out[key] = {"n": len(rows), **_metrics_for(rows, normalizer)}
    return out


def duration_slices(records, normalizer) -> dict:
    raw = slice_by(records, lambda r: duration_bucket(r["duration_s"]), normalizer)
    return {b: raw[b] for b in DURATION_BUCKET_ORDER if b in raw}


def metadata_slices(records, normalizer, min_n: int = 20) -> dict:
    out = {}
    for field in ("gender", "age", "accents"):
        with_value = [r for r in records if r.get(field)]
        coverage = len(with_value) / max(len(records), 1)
        out[field] = {
            "coverage": round(coverage, 3),
            "n_with_value": len(with_value),
            "slices": slice_by(with_value, lambda r: r[field], normalizer, min_n=min_n),
        }
    return out


def turkish_char_slices(records, normalizer) -> dict:
    has = [r for r in records if contains_turkish_specific(r["reference_raw"])]
    without = [r for r in records if not contains_turkish_specific(r["reference_raw"])]
    out = {}
    if has:
        out["with_turkish_chars"] = {"n": len(has), **_metrics_for(has, normalizer)}
    if without:
        out["without_turkish_chars"] = {"n": len(without), **_metrics_for(without, normalizer)}
    return out


def per_char_error_attribution(records, normalizer) -> dict:
    """Character-level error rates from aligned normalized ref/hyp pairs.

    For each reference character we mark it correct when it sits inside an
    'equal' opcode of the alignment, and an error otherwise (substituted or
    deleted). Reported per Turkish-specific letter, with the aggregate over
    all other letters as the baseline.
    """
    stats = {ch: {"ref_count": 0, "errors": 0} for ch in LOWER_TURKISH_SPECIFIC}
    other = {"ref_count": 0, "errors": 0}
    for r in records:
        ref = normalizer(r["reference_raw"])
        hyp = normalizer(r["hypothesis_raw"])
        matcher = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
        for tag, a0, a1, _b0, _b1 in matcher.get_opcodes():
            for ch in ref[a0:a1]:
                if not ch.isalpha():
                    continue
                target = stats[ch] if ch in stats else other
                target["ref_count"] += 1
                if tag != "equal":
                    target["errors"] += 1
    out = {}
    for ch, s in stats.items():
        out[ch] = {
            **s,
            "char_error_rate": round(s["errors"] / s["ref_count"], 4) if s["ref_count"] else None,
        }
    out["all_other_letters"] = {
        **other,
        "char_error_rate": round(other["errors"] / other["ref_count"], 4)
        if other["ref_count"]
        else None,
    }
    return out


def worst_error_pairs(records, normalizer, n: int = 10, min_words: int = 4) -> list[dict]:
    scored = []
    for r in records:
        ref_n = normalizer(r["reference_raw"])
        hyp_n = normalizer(r["hypothesis_raw"])
        if len(ref_n.split()) < min_words:
            continue
        scored.append(
            {
                "utt_id": r["utt_id"],
                "duration_s": round(r["duration_s"], 1),
                "wer_norm": round(utterance_wer(ref_n, hyp_n), 3),
                "reference": r["reference_raw"],
                "hypothesis": r["hypothesis_raw"],
                "turkish_chars_in_ref": contains_turkish_specific(r["reference_raw"]),
            }
        )
    scored.sort(key=lambda d: -d["wer_norm"])
    return scored[:n]


def analyze_run(run_dir: str | Path, normalizer=None, examples: int = 10, min_slice_n: int = 20) -> dict:
    normalizer = normalizer or TurkishNormalizer()
    run, records = load_run(run_dir)
    if not records:
        return {
            "run_id": run.get("run_id"),
            "model": run.get("model"),
            "status": run.get("status"),
            "error": (run.get("error") or {}).get("message"),
            "no_predictions": True,
        }
    return {
        "run_id": run.get("run_id"),
        "model": run.get("model"),
        "adapter": run.get("adapter"),
        "status": run.get("status"),
        "dataset": records[0]["dataset"],
        "split": records[0]["split"],
        "aggregate": _metrics_for(records, normalizer),
        "duration_slices": duration_slices(records, normalizer),
        "metadata_slices": metadata_slices(records, normalizer, min_n=min_slice_n),
        "turkish_char_slices": turkish_char_slices(records, normalizer),
        "per_char_errors": per_char_error_attribution(records, normalizer),
        "worst_pairs": worst_error_pairs(records, normalizer, n=examples),
    }


def _fmt_pct(x) -> str:
    return "—" if x is None or x != x else f"{100 * x:.1f}"


def emit_markdown(analyses: list[dict], out_path: Path) -> None:
    lines: list[str] = ["<!-- generated by turkish-asr-analyze; do not edit numbers by hand -->", ""]
    ok = [a for a in analyses if not a.get("no_predictions")]
    if len(ok) > 1:
        lines += ["## Model comparison (aggregate)", ""]
        lines += [
            "| run | dataset | WER raw % | WER norm % | CER raw % | CER norm % | n |",
            "|---|---|---|---|---|---|---|",
        ]
        for a in ok:
            g = a["aggregate"]
            lines.append(
                f"| {a['run_id']} | {a['dataset']}/{a['split']} | {_fmt_pct(g['wer_raw'])} "
                f"| {_fmt_pct(g['wer_norm'])} | {_fmt_pct(g['cer_raw'])} | {_fmt_pct(g['cer_norm'])} "
                f"| {g['n_utterances']} |"
            )
        lines.append("")
    for a in analyses:
        lines += [f"## Run `{a.get('run_id')}`", ""]
        if a.get("no_predictions"):
            lines += [
                f"Status: **{a.get('status')}** — no predictions produced.",
                f"Error: `{a.get('error')}`" if a.get("error") else "",
                "",
            ]
            continue
        g = a["aggregate"]
        lines += [
            f"Model: `{a.get('model')}`" + (f" + adapter `{a.get('adapter')}`" if a.get("adapter") else ""),
            f"Dataset: {a['dataset']}/{a['split']}, {g['n_utterances']} utterances.",
            "",
            "| metric | raw | normalized |",
            "|---|---|---|",
            f"| WER % | {_fmt_pct(g['wer_raw'])} | {_fmt_pct(g['wer_norm'])} |",
            f"| CER % | {_fmt_pct(g['cer_raw'])} | {_fmt_pct(g['cer_norm'])} |",
            "",
            "### WER by duration bucket",
            "",
            "| bucket | n | WER raw % | WER norm % |",
            "|---|---|---|---|",
        ]
        for b, m in a["duration_slices"].items():
            lines.append(f"| {b} | {m['n']} | {_fmt_pct(m['wer_raw'])} | {_fmt_pct(m['wer_norm'])} |")
        lines += ["", "### Speaker metadata slices", ""]
        for field, info in a["metadata_slices"].items():
            lines.append(
                f"- **{field}**: coverage {info['coverage'] * 100:.0f}% "
                f"({info['n_with_value']} utterances with a value)"
            )
            if info["slices"]:
                lines += ["", "| value | n | WER norm % |", "|---|---|---|"]
                for val, m in info["slices"].items():
                    lines.append(f"| {val} | {m['n']} | {_fmt_pct(m['wer_norm'])} |")
                lines.append("")
        lines += ["", "### Turkish-specific characters", ""]
        lines += ["| slice | n | WER raw % | WER norm % |", "|---|---|---|---|"]
        for name, m in a["turkish_char_slices"].items():
            lines.append(
                f"| {name} | {m['n']} | {_fmt_pct(m['wer_raw'])} | {_fmt_pct(m['wer_norm'])} |"
            )
        lines += ["", "Per-character error rates (aligned, normalized text):", ""]
        lines += ["| char | ref count | errors | char error rate % |", "|---|---|---|---|"]
        for ch, s in a["per_char_errors"].items():
            cer = s["char_error_rate"]
            lines.append(
                f"| {ch} | {s['ref_count']} | {s['errors']} | "
                f"{'—' if cer is None else f'{100 * cer:.1f}'} |"
            )
        lines += ["", "### Worst error pairs", ""]
        for i, p in enumerate(a["worst_pairs"], 1):
            lines += [
                f"{i}. `{p['utt_id']}` ({p['duration_s']}s, utterance WER {p['wer_norm']:.2f})",
                f"   - ref: {p['reference']}",
                f"   - hyp: {p['hypothesis']}",
            ]
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sliced analysis of evaluation runs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="K=V")
    parser.add_argument("--runs", nargs="*", default=[], help="run directories to analyze")
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--min-slice-n", type=int, default=20)
    parser.add_argument("--emit-markdown", default=None, metavar="PATH")
    parser.add_argument("--out-json", default=None, metavar="PATH")
    parser.add_argument("--data-stats", action="store_true", help="dataset statistics mode")
    parser.add_argument("--dataset", default=None, choices=["fleurs", "common_voice"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.get("seed", 42))
    runs_dir = get_path(cfg, "runs_dir", "runs")

    if args.data_stats:
        if not args.dataset:
            parser.error("--data-stats requires --dataset")
        from turkish_asr import data as data_mod

        log = RunLog(runs_dir, "datastats", f"{args.dataset}-{args.split}", cfg)
        try:
            manifest = data_mod.prepare_manifest(cfg, args.dataset, args.split, args.max_samples)
            stats = data_mod.manifest_stats(manifest)
        except BaseException as exc:
            log.fail(exc)
            raise
        log.log_metrics(**stats)
        log.finish()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        print(f"run log: {log.json_path}")
        return 0

    if not args.runs:
        parser.error("provide --runs run_dir [run_dir ...] or --data-stats")
    analyses = [
        analyze_run(r, examples=args.examples, min_slice_n=args.min_slice_n) for r in args.runs
    ]
    payload = json.dumps(analyses, indent=2, ensure_ascii=False)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(payload, encoding="utf-8")
    else:
        print(payload)
    if args.emit_markdown:
        emit_markdown(analyses, Path(args.emit_markdown))
        print(f"markdown written to {args.emit_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
