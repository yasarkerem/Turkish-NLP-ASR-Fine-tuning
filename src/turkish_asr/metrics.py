"""WER/CER computation (raw and Turkish-normalized) built on jiwer."""

from __future__ import annotations

from dataclasses import dataclass, field

import jiwer

from turkish_asr.normalize import TurkishNormalizer


@dataclass
class PairMetrics:
    wer_raw: float
    cer_raw: float
    wer_norm: float
    cer_norm: float
    n_utterances: int
    n_ref_words_raw: int
    n_ref_words_norm: int
    n_skipped_empty_norm: int = 0
    n_skipped_empty_raw: int = 0
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {
            "wer_raw": self.wer_raw,
            "cer_raw": self.cer_raw,
            "wer_norm": self.wer_norm,
            "cer_norm": self.cer_norm,
            "n_utterances": self.n_utterances,
            "n_ref_words_raw": self.n_ref_words_raw,
            "n_ref_words_norm": self.n_ref_words_norm,
            "n_skipped_empty_norm": self.n_skipped_empty_norm,
            "n_skipped_empty_raw": self.n_skipped_empty_raw,
        }
        d.update(self.extras)
        return d


def _corpus_wer_cer(refs: list[str], hyps: list[str]) -> tuple[float, float, int]:
    """Corpus-level WER and CER; hypotheses may be empty, references not."""
    kept = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not kept:
        return float("nan"), float("nan"), 0
    refs_k = [r for r, _ in kept]
    hyps_k = [h for _, h in kept]
    wer = jiwer.wer(refs_k, hyps_k)
    cer = jiwer.cer(refs_k, hyps_k)
    return float(wer), float(cer), len(refs) - len(kept)


def compute_pair_metrics(
    refs_raw: list[str],
    hyps_raw: list[str],
    normalizer: TurkishNormalizer | None = None,
) -> PairMetrics:
    """Compute raw and normalized corpus WER/CER for aligned ref/hyp lists.

    "Raw" applies no text processing beyond whitespace tokenization; case and
    punctuation differences count as errors. "Norm" applies the Turkish
    normalizer to both sides. Utterances whose reference is empty (or becomes
    empty after normalization) are excluded from the affected metric and
    counted in ``n_skipped_*``.
    """
    if len(refs_raw) != len(hyps_raw):
        raise ValueError(f"got {len(refs_raw)} references but {len(hyps_raw)} hypotheses")
    normalizer = normalizer or TurkishNormalizer()

    wer_raw, cer_raw, skipped_raw = _corpus_wer_cer(refs_raw, hyps_raw)

    refs_norm = [normalizer(r) for r in refs_raw]
    hyps_norm = [normalizer(h) for h in hyps_raw]
    wer_norm, cer_norm, skipped_norm = _corpus_wer_cer(refs_norm, hyps_norm)

    return PairMetrics(
        wer_raw=wer_raw,
        cer_raw=cer_raw,
        wer_norm=wer_norm,
        cer_norm=cer_norm,
        n_utterances=len(refs_raw),
        n_ref_words_raw=sum(len(r.split()) for r in refs_raw),
        n_ref_words_norm=sum(len(r.split()) for r in refs_norm),
        n_skipped_empty_raw=skipped_raw,
        n_skipped_empty_norm=skipped_norm,
    )


def utterance_wer(ref: str, hyp: str) -> float:
    """Single-utterance WER; returns NaN for an empty reference."""
    if not ref.strip():
        return float("nan")
    return float(jiwer.wer(ref, hyp))


def duration_bucket(seconds: float) -> str:
    if seconds < 3:
        return "<3s"
    if seconds < 6:
        return "3-6s"
    if seconds < 10:
        return "6-10s"
    return ">10s"


DURATION_BUCKET_ORDER = ["<3s", "3-6s", "6-10s", ">10s"]
