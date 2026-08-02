"""Fine-tune Whisper on Turkish speech, with LoRA (PEFT) or full fine-tuning.

Everything is driven by a YAML config (see ``configs/``); any key can be
overridden on the command line with ``--set a.b.c=value``. Each run writes
``runs/<run_id>/run.json`` with the resolved config, environment and final
metrics before and after training.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from turkish_asr.config import get_path, load_config, seed_everything
from turkish_asr.metrics import compute_pair_metrics
from turkish_asr.normalize import TurkishNormalizer
from turkish_asr.runlog import RunLog


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pad log-mel inputs and label ids independently.

    Label padding becomes -100 so it is ignored by the loss; a leading
    decoder-start token is stripped because the model re-adds it when
    shifting labels right.
    """

    processor: object

    def __call__(self, features: list[dict]) -> dict:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        sot = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if (labels[:, 0] == sot).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def _load_processor(model_cfg: dict):
    from transformers import WhisperProcessor

    return WhisperProcessor.from_pretrained(
        model_cfg["id"],
        language=model_cfg.get("language", "turkish"),
        task=model_cfg.get("task", "transcribe"),
    )


def _apply_lora(model, lora_cfg: dict):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def _make_compute_metrics(processor, normalizer: TurkishNormalizer):
    def compute_metrics(pred):
        import numpy as np

        pad_id = processor.tokenizer.pad_token_id
        label_ids = np.where(pred.label_ids == -100, pad_id, pred.label_ids)
        hyps = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        refs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        m = compute_pair_metrics(refs, hyps, normalizer)
        return {"wer_norm": m.wer_norm, "cer_norm": m.cer_norm, "wer_raw": m.wer_raw}

    return compute_metrics


def run_training(cfg: dict, log: RunLog) -> dict:
    import torch
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
    )

    from turkish_asr import data as data_mod

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    seed = cfg.get("seed", 42)

    processor = _load_processor(model_cfg)
    slug = model_cfg["id"].rstrip("/").split("/")[-1]

    train_ds = data_mod.prepare_features(
        cfg,
        train_cfg["dataset"],
        train_cfg["split"],
        processor,
        slug,
        train_cfg.get("max_train_samples"),
    )
    eval_ds = None
    if train_cfg.get("eval_split"):
        eval_ds = data_mod.prepare_features(
            cfg,
            train_cfg["dataset"],
            train_cfg["eval_split"],
            processor,
            slug,
            train_cfg.get("max_eval_samples"),
        )
    log.log_metrics(n_train=len(train_ds), n_eval=len(eval_ds) if eval_ds else 0)

    model = WhisperForConditionalGeneration.from_pretrained(model_cfg["id"])
    model.config.forced_decoder_ids = None
    model.config.use_cache = False
    model.generation_config.language = model_cfg.get("language_code", "tr")
    model.generation_config.task = model_cfg.get("task", "transcribe")
    model.generation_config.forced_decoder_ids = None

    method = train_cfg.get("method", "lora")
    if method == "lora":
        model = _apply_lora(model, train_cfg.get("lora", {}))
    elif method != "full":
        raise ValueError(f"unknown train.method {method!r}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.log_metrics(trainable_params=trainable, total_params=total)

    models_dir = get_path(cfg, "models_dir", "models")
    out_name = cfg.get("output_name", f"{slug}-{method}")
    out_dir = models_dir / out_name
    fp16 = train_cfg.get("fp16", "auto")
    fp16 = torch.cuda.is_available() if fp16 == "auto" else bool(fp16)

    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=train_cfg.get("batch_size", 16),
        per_device_eval_batch_size=train_cfg.get("eval_batch_size", 8),
        gradient_accumulation_steps=train_cfg.get("grad_accum", 1),
        learning_rate=float(train_cfg.get("lr", 1e-3)),
        warmup_steps=train_cfg.get("warmup_steps", 50),
        num_train_epochs=train_cfg.get("num_epochs", 1),
        max_steps=train_cfg.get("max_steps") or -1,
        fp16=fp16,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="no",
        predict_with_generate=train_cfg.get("predict_with_generate", True),
        generation_max_length=train_cfg.get("generation_max_length", 225),
        logging_steps=train_cfg.get("logging_steps", 25),
        report_to=[],
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=train_cfg.get("dataloader_workers", 0),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
        compute_metrics=_make_compute_metrics(processor, TurkishNormalizer()) if eval_ds is not None else None,
        processing_class=processor.feature_extractor,
    )

    result = trainer.train()
    metrics = dict(result.metrics)

    if eval_ds is not None:
        metrics.update(trainer.evaluate())

    out_dir.mkdir(parents=True, exist_ok=True)
    if method == "lora":
        adapter_dir = out_dir / "adapter"
        model.save_pretrained(str(adapter_dir))
        log.log_artifact("adapter", adapter_dir)
    else:
        model.save_pretrained(str(out_dir / "model"))
        log.log_artifact("model", out_dir / "model")
    processor.save_pretrained(str(out_dir))
    log.log_artifact("processor", out_dir)

    log.log_metrics(**{k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for Turkish ASR")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="K=V")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.get("seed", 42))
    runs_dir = get_path(cfg, "runs_dir", "runs")
    name = cfg.get("output_name", "train")
    log = RunLog(runs_dir, "train", name, cfg)
    try:
        run_training(cfg, log)
    except BaseException as exc:
        log.fail(exc)
        raise
    log.finish()
    print(f"run log: {log.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
