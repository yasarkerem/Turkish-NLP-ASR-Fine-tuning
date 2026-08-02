"""Evaluate an ASR model on a prepared test set.

Reports, per run:

- corpus WER/CER, both raw and Turkish-normalized (always both);
- real-time factor (processing seconds per audio second) measured over a
  seeded subset at batch size 1, plus batched throughput for the main pass;
- parameter count/bytes, on-disk size when the model is a local path, and
  peak inference memory (GPU allocator peak on CUDA, RSS peak sampled by a
  background thread on CPU);
- a ``predictions.jsonl`` with one record per utterance for sliced analysis.

A failed run (e.g. the checkpoint cannot be downloaded) still writes its
run.json with status "failed" and the error, so unproduced numbers remain
visible instead of silently missing.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

from turkish_asr.config import get_path, load_config, seed_everything
from turkish_asr.metrics import compute_pair_metrics
from turkish_asr.normalize import TurkishNormalizer
from turkish_asr.runlog import RunLog


class RssSampler:
    """Background thread sampling process RSS to catch the peak."""

    def __init__(self, interval_s: float = 0.05):
        import psutil

        self._proc = psutil.Process()
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_bytes = 0

    def __enter__(self):
        self.peak_bytes = self._proc.memory_info().rss
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss
            if rss > self.peak_bytes:
                self.peak_bytes = rss
            self._stop.wait(self._interval)

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return False


def _disk_size_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    exts = {".safetensors", ".bin", ".pt", ".msgpack", ".h5"}
    total = sum(f.stat().st_size for f in path.rglob("*") if f.suffix in exts)
    return total or None


def load_model_and_processor(model_id: str, adapter: str | None, device, dtype):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor_src = model_id
    if adapter:
        # The training script saves the processor next to the adapter dir.
        candidate = Path(adapter).parent
        if (candidate / "preprocessor_config.json").exists():
            processor_src = str(candidate)
    processor = WhisperProcessor.from_pretrained(processor_src)

    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
    adapter_params_bytes = None
    if adapter:
        from peft import PeftModel

        peft_model = PeftModel.from_pretrained(model, adapter)
        adapter_params_bytes = sum(
            p.numel() * p.element_size()
            for n, p in peft_model.named_parameters()
            if "lora_" in n
        )
        model = peft_model.merge_and_unload()
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    return model, processor, adapter_params_bytes


def transcribe_batch(model, processor, batch_audio, language, task, num_beams, max_new_tokens, device, dtype):
    features = processor.feature_extractor(
        [a["array"] for a in batch_audio],
        sampling_rate=batch_audio[0]["sampling_rate"],
        return_tensors="pt",
    ).input_features.to(device=device, dtype=dtype)
    generated = model.generate(
        features,
        language=language,
        task=task,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
    )
    return processor.tokenizer.batch_decode(generated, skip_special_tokens=True)


def run_evaluation(cfg: dict, args, log: RunLog) -> None:
    import torch

    from turkish_asr import data as data_mod

    seed = cfg.get("seed", 42)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (args.dtype == "float16" and device == "cuda") else torch.float32
    if args.threads:
        torch.set_num_threads(args.threads)

    manifest = data_mod.prepare_manifest(cfg, args.dataset, args.split, args.max_samples)
    log.log_metrics(n_utterances=len(manifest), device=device, dtype=str(dtype))

    model, processor, adapter_bytes = load_model_and_processor(
        args.model, args.adapter, device, dtype
    )
    params_total = sum(p.numel() for p in model.parameters())
    params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    size = {
        "params_total": params_total,
        "params_bytes": params_bytes,
        "params_mib": round(params_bytes / 2**20, 1),
    }
    if adapter_bytes is not None:
        size["adapter_params_bytes"] = adapter_bytes
        size["adapter_disk_bytes"] = _disk_size_bytes(Path(args.adapter))
    local = Path(args.model)
    if local.exists():
        size["model_disk_bytes"] = _disk_size_bytes(local)
    log.log_metrics(model_size=size)

    language = cfg.get("model", {}).get("language_code", "tr")
    task = cfg.get("model", {}).get("task", "transcribe")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    records: list[dict] = []
    durations = manifest["duration_s"]
    order = sorted(range(len(manifest)), key=lambda i: -durations[i])
    batch_time = 0.0
    batch_audio_s = 0.0
    with RssSampler() as rss:
        for start in range(0, len(order), args.batch_size):
            idxs = order[start : start + args.batch_size]
            rows = [manifest[i] for i in idxs]
            t0 = time.perf_counter()
            hyps = transcribe_batch(
                model,
                processor,
                [r["audio"] for r in rows],
                language,
                task,
                args.num_beams,
                args.max_new_tokens,
                device,
                dtype,
            )
            batch_time += time.perf_counter() - t0
            batch_audio_s += sum(r["duration_s"] for r in rows)
            for row, hyp in zip(rows, hyps):
                records.append(
                    {
                        "utt_id": row["utt_id"],
                        "reference_raw": row["reference_raw"],
                        "hypothesis_raw": hyp.strip(),
                        "duration_s": row["duration_s"],
                        "dataset": row["dataset"],
                        "split": row["split"],
                        "age": row["age"],
                        "gender": row["gender"],
                        "accents": row["accents"],
                        "speaker_id": row.get("speaker_id"),
                    }
                )
            done = start + len(idxs)
            if done % (args.batch_size * 10) < args.batch_size or done == len(order):
                print(f"  transcribed {done}/{len(order)}", flush=True)

        # RTF pass: batch size 1 over a seeded subset, so numbers are
        # per-utterance latency, not batching throughput.
        rtf_records = []
        if args.rtf_utterances > 0:
            import random as _random

            rng = _random.Random(seed)
            rtf_idx = list(range(len(manifest)))
            rng.shuffle(rtf_idx)
            rtf_idx = rtf_idx[: args.rtf_utterances]
            for i in rtf_idx:
                row = manifest[i]
                t0 = time.perf_counter()
                transcribe_batch(
                    model, processor, [row["audio"]], language, task,
                    args.num_beams, args.max_new_tokens, device, dtype,
                )
                dt = time.perf_counter() - t0
                rtf_records.append({"duration_s": row["duration_s"], "proc_s": dt})

    memory = {"peak_rss_mib": round(rss.peak_bytes / 2**20, 1)}
    if device == "cuda":
        memory["peak_cuda_alloc_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    log.log_metrics(memory=memory)

    preds_path = log.dir / "predictions.jsonl"
    with open(preds_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.log_artifact("predictions", preds_path)

    m = compute_pair_metrics(
        [r["reference_raw"] for r in records],
        [r["hypothesis_raw"] for r in records],
        TurkishNormalizer(),
    )
    throughput = {
        "batched_proc_s": round(batch_time, 2),
        "batched_audio_s": round(batch_audio_s, 2),
        "batched_rtf": round(batch_time / batch_audio_s, 4) if batch_audio_s else None,
        "batch_size": args.batch_size,
    }
    rtf = None
    if rtf_records:
        proc = sum(r["proc_s"] for r in rtf_records)
        audio = sum(r["duration_s"] for r in rtf_records)
        per_utt = sorted(r["proc_s"] / r["duration_s"] for r in rtf_records)
        rtf = {
            "n_utterances": len(rtf_records),
            "rtf_overall": round(proc / audio, 4),
            "rtf_median": round(per_utt[len(per_utt) // 2], 4),
            "rtf_p90": round(per_utt[int(len(per_utt) * 0.9) - 1], 4),
            "batch_size": 1,
        }
    log.log_metrics(wer_cer=m.as_dict(), throughput=throughput, rtf=rtf)

    print(json.dumps({"wer_cer": m.as_dict(), "rtf": rtf, "memory": memory}, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate an ASR model on a test set")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="K=V")
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--adapter", default=None, help="PEFT adapter dir (merged before eval)")
    parser.add_argument("--dataset", required=True, choices=["fleurs", "common_voice"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=225)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument("--rtf-utterances", type=int, default=60)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.get("seed", 42))
    runs_dir = get_path(cfg, "runs_dir", "runs")
    name = args.run_name or f"{Path(args.model).name}_{args.dataset}-{args.split}"
    log = RunLog(runs_dir, "eval", name, cfg)
    log.payload["model"] = args.model
    log.payload["adapter"] = args.adapter
    try:
        run_evaluation(cfg, args, log)
    except BaseException as exc:
        log.fail(exc)
        print(f"run FAILED, log: {log.json_path}")
        raise
    log.finish()
    print(f"run log: {log.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
