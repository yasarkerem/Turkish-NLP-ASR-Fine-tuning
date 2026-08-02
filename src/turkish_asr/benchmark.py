"""Architecture-level inference throughput benchmark.

Measures real-time factor and peak memory for the Whisper architectures being
compared, using randomly initialized weights and a pinned number of decoder
steps per utterance.

Why random weights are legitimate here and only here: inference wall-time is
a function of architecture shape, input length and the number of decoder
steps — not of the weight values. Pinning decoder steps to a rate per audio
second removes the one weight-dependent variable (output length). The
resulting RTF is an architecture property; it says nothing about accuracy and
is labeled accordingly in reports. WER/CER from random weights would be
meaningless and is deliberately not computed.

The decoder loop is hand-rolled (encoder forward once, then N cached decoder
steps with argmax feedback) so the measurement does not depend on
``generate()``'s stopping behavior with untrained logits.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from turkish_asr.config import get_path, load_config, seed_everything
from turkish_asr.runlog import RunLog

# Architecture hyperparameters of the OpenAI Whisper family, from the model
# cards / config.json of the corresponding checkpoints.
WHISPER_ARCHS = {
    "tiny": dict(d_model=384, layers=4, heads=6, ffn=1536, n_mels=80, vocab=51865),
    "base": dict(d_model=512, layers=6, heads=8, ffn=2048, n_mels=80, vocab=51865),
    "small": dict(d_model=768, layers=12, heads=12, ffn=3072, n_mels=80, vocab=51865),
    "medium": dict(d_model=1024, layers=24, heads=16, ffn=4096, n_mels=80, vocab=51865),
    "large-v3": dict(d_model=1280, layers=32, heads=20, ffn=5120, n_mels=128, vocab=51866),
}


def build_random_model(arch: str, seed: int):
    import torch
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    spec = WHISPER_ARCHS[arch]
    torch.manual_seed(seed)
    config = WhisperConfig(
        vocab_size=spec["vocab"],
        num_mel_bins=spec["n_mels"],
        d_model=spec["d_model"],
        encoder_layers=spec["layers"],
        decoder_layers=spec["layers"],
        encoder_attention_heads=spec["heads"],
        decoder_attention_heads=spec["heads"],
        encoder_ffn_dim=spec["ffn"],
        decoder_ffn_dim=spec["ffn"],
        max_source_positions=1500,
        max_target_positions=448,
    )
    # Allocate on the meta device and fill with a single vectorized uniform
    # draw instead of running the per-module init graph: billions of values
    # are never needed for timing, only that they are sane floats (no NaNs,
    # no denormals — both would distort CPU timings). This keeps peak memory
    # at one weight copy, which matters for large-v3 on small machines.
    with torch.device("meta"):
        model = WhisperForConditionalGeneration(config)
    model = model.to_empty(device="cpu")
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.02, 0.02)
        for b in model.buffers():
            if b.dtype.is_floating_point:
                b.uniform_(-0.02, 0.02)
    # to_empty materializes tied tensors as separate copies; retie so the
    # parameter count and memory match the real checkpoints (Whisper ties
    # proj_out to the decoder token embedding).
    model.tie_weights()
    return model


def timed_transcode(model, features, n_steps: int) -> dict:
    """One utterance: encoder forward + n_steps cached greedy decoder steps."""
    import torch

    t0 = time.perf_counter()
    encoder_outputs = model.model.encoder(features)
    t_enc = time.perf_counter() - t0

    decoder_input = torch.tensor([[model.config.decoder_start_token_id]])
    past_key_values = None
    t1 = time.perf_counter()
    for _ in range(n_steps):
        out = model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = out.past_key_values
        decoder_input = out.logits[:, -1:].argmax(-1)
    t_dec = time.perf_counter() - t1
    return {"encoder_s": t_enc, "decoder_s": t_dec, "total_s": t_enc + t_dec}


def run_benchmark(cfg: dict, args, log: RunLog) -> dict:
    import torch
    from transformers import WhisperFeatureExtractor

    from turkish_asr import data as data_mod
    from turkish_asr.evaluate import RssSampler

    seed = cfg.get("seed", 42)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    manifest = data_mod.prepare_manifest(cfg, args.dataset, args.split, None)
    import random as _random

    rng = _random.Random(seed)
    idxs = list(range(len(manifest)))
    rng.shuffle(idxs)
    idxs = idxs[: args.n_utterances]

    spec = WHISPER_ARCHS[args.arch]
    extractor = WhisperFeatureExtractor(feature_size=spec["n_mels"])

    with RssSampler() as rss:
        model = build_random_model(args.arch, seed)
        model.eval()
        post_load_rss = rss.peak_bytes

        params_total = sum(p.numel() for p in model.parameters())
        params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

        # Warmup so first-call allocator noise stays out of the timings.
        row = manifest[idxs[0]]
        feats = extractor(
            row["audio"]["array"], sampling_rate=row["audio"]["sampling_rate"], return_tensors="pt"
        ).input_features
        timed_transcode(model, feats, 8)

        per_utt = []
        for k, i in enumerate(idxs):
            row = manifest[i]
            audio = row["audio"]
            duration = row["duration_s"]
            n_steps = max(args.min_steps, int(round(args.tokens_per_audio_second * duration)))
            t0 = time.perf_counter()
            feats = extractor(
                audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
            ).input_features
            t_feat = time.perf_counter() - t0
            timing = timed_transcode(model, feats, n_steps)
            per_utt.append(
                {
                    "duration_s": duration,
                    "n_steps": n_steps,
                    "feature_s": t_feat,
                    **timing,
                    "rtf": (t_feat + timing["total_s"]) / duration,
                }
            )
            if (k + 1) % 10 == 0 or k + 1 == len(idxs):
                print(f"  {args.arch}: {k + 1}/{len(idxs)} utterances", flush=True)
                log.log_metrics(progress_utterances=k + 1)

    proc = sum(u["feature_s"] + u["total_s"] for u in per_utt)
    audio_s = sum(u["duration_s"] for u in per_utt)
    rtfs = sorted(u["rtf"] for u in per_utt)
    metrics = {
        "arch": args.arch,
        "params_total": params_total,
        "params_bytes_fp32": params_bytes,
        "params_mib_fp32": round(params_bytes / 2**20, 1),
        "params_mib_fp16_equiv": round(params_bytes / 2 / 2**20, 1),
        "n_utterances": len(per_utt),
        "audio_seconds": round(audio_s, 1),
        "proc_seconds": round(proc, 1),
        "rtf_overall": round(proc / audio_s, 4),
        "rtf_median": round(rtfs[len(rtfs) // 2], 4),
        "rtf_p90": round(rtfs[int(len(rtfs) * 0.9) - 1], 4),
        "encoder_seconds": round(sum(u["encoder_s"] for u in per_utt), 2),
        "decoder_seconds": round(sum(u["decoder_s"] for u in per_utt), 2),
        "tokens_per_audio_second": args.tokens_per_audio_second,
        "torch_threads": __import__("torch").get_num_threads(),
        "post_load_rss_mib": round(post_load_rss / 2**20, 1),
        "peak_rss_mib": round(rss.peak_bytes / 2**20, 1),
    }
    log.log_metrics(**metrics)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Random-weight architecture throughput benchmark (no accuracy)"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="K=V")
    parser.add_argument("--arch", required=True, choices=sorted(WHISPER_ARCHS))
    parser.add_argument("--dataset", default="fleurs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-utterances", type=int, default=60)
    parser.add_argument(
        "--tokens-per-audio-second",
        type=float,
        default=3.2,
        help="pinned decoder steps per audio second (proxy for transcript length)",
    )
    parser.add_argument("--min-steps", type=int, default=8)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.get("seed", 42))
    runs_dir = get_path(cfg, "runs_dir", "runs")
    log = RunLog(runs_dir, "benchmark", f"{args.arch}_cpu", cfg)
    try:
        metrics = run_benchmark(cfg, args, log)
    except BaseException as exc:
        log.fail(exc)
        raise
    log.finish()
    import json

    print(json.dumps(metrics, indent=2))
    print(f"run log: {log.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
