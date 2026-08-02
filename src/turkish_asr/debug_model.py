"""Build a tiny, fully local Whisper model for smoke tests.

Creates from scratch — no network access — a working
(feature extractor, tokenizer, model) triple that goes through the exact same
train/evaluate code paths as the real checkpoints:

- a byte-level BPE tokenizer trained on a small embedded Turkish corpus and
  wrapped in ``WhisperTokenizer``, with Whisper's special-token layout
  (``<|startoftranscript|>`` followed immediately by one token per language in
  ``LANGUAGES`` order, which is what ``WhisperTokenizer.prefix_tokens``
  assumes when it computes language ids positionally);
- a randomly initialized ``WhisperForConditionalGeneration`` a few dozen
  hidden units wide;
- a generation config with ``lang_to_id``/``task_to_id`` so
  ``generate(language="tr", task="transcribe")`` works.

The model cannot transcribe anything; it exists so the pipeline mechanics
(feature shapes, label building, LoRA wrapping, generation, WER computation)
can be exercised hermetically.
"""

from __future__ import annotations

from pathlib import Path

DEBUG_CORPUS = [
    "Merhaba, bugün hava çok güzel.",
    "İstanbul Boğazı'nda vapurla karşıya geçtik.",
    "Ağaçların yaprakları sonbaharda sararır.",
    "Şu köşedeki fırından taze ekmek aldım.",
    "Çocuklar bahçede sessizce oyun oynuyor.",
    "Iğdır ve Ağrı doğu illerimizdendir.",
    "Öğretmen öğrencilere yeni ödev verdi.",
    "Üzüm bağları bu yıl bol ürün verdi.",
    "Gökyüzünde yıldızlar parlıyordu.",
    "Türkçe, sondan eklemeli bir dildir.",
    "Kedi pencereden dışarıyı izliyor.",
    "Yarın sabah erkenden yola çıkacağız.",
    "Denizin üzerinde martılar uçuşuyor.",
    "Bu şarkının sözlerini ezbere biliyorum.",
    "Ilık bir rüzgar esiyordu akşamüstü.",
    "İnsanlar meydanda toplanmaya başladı.",
]


def build_debug_model(
    out_dir: str | Path,
    vocab_size: int = 400,
    d_model: int = 64,
    num_layers: int = 2,
    num_heads: int = 2,
    ffn_dim: int = 128,
    seed: int = 0,
    extra_corpus: list[str] | None = None,
) -> Path:
    import torch
    from tokenizers import ByteLevelBPETokenizer
    from transformers import (
        GenerationConfig,
        WhisperConfig,
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperTokenizer,
    )
    from transformers.models.whisper.tokenization_whisper import LANGUAGES

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    corpus = DEBUG_CORPUS + (extra_corpus or [])
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(
        corpus,
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=["<|endoftext|>"],
    )
    bpe.save_model(str(out_dir))

    tokenizer = WhisperTokenizer(
        vocab_file=str(out_dir / "vocab.json"),
        merges_file=str(out_dir / "merges.txt"),
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    # Whisper resolves language token ids as sot_id + 1 + index-in-LANGUAGES,
    # so the language tokens must be contiguous right after the sot token.
    specials = ["<|startoftranscript|>"]
    specials += [f"<|{code}|>" for code in LANGUAGES]
    specials += [
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
    ]
    tokenizer.add_special_tokens({"additional_special_tokens": specials})
    tokenizer.set_prefix_tokens(language="turkish", task="transcribe")

    tid = tokenizer.convert_tokens_to_ids
    eot = tid("<|endoftext|>")
    sot = tid("<|startoftranscript|>")

    config = WhisperConfig(
        vocab_size=len(tokenizer),
        num_mel_bins=80,
        d_model=d_model,
        encoder_layers=num_layers,
        decoder_layers=num_layers,
        encoder_attention_heads=num_heads,
        decoder_attention_heads=num_heads,
        encoder_ffn_dim=ffn_dim,
        decoder_ffn_dim=ffn_dim,
        max_source_positions=1500,
        max_target_positions=448,
        pad_token_id=eot,
        bos_token_id=eot,
        eos_token_id=eot,
        decoder_start_token_id=sot,
        suppress_tokens=[],
        begin_suppress_tokens=[],
        forced_decoder_ids=None,
    )
    model = WhisperForConditionalGeneration(config)

    model.generation_config = GenerationConfig(
        max_length=448,
        pad_token_id=eot,
        bos_token_id=eot,
        eos_token_id=eot,
        decoder_start_token_id=sot,
        suppress_tokens=[],
        begin_suppress_tokens=[],
        is_multilingual=True,
        lang_to_id={f"<|{code}|>": tid(f"<|{code}|>") for code in LANGUAGES},
        task_to_id={"translate": tid("<|translate|>"), "transcribe": tid("<|transcribe|>")},
        no_timestamps_token_id=tid("<|notimestamps|>"),
        prev_sot_token_id=tid("<|startofprev|>"),
    )

    feature_extractor = WhisperFeatureExtractor()
    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    return out_dir


def make_synthetic_utterances(
    n: int, seed: int = 0, sampling_rate: int = 16000
) -> list[dict]:
    """Deterministic sine-plus-noise clips with Turkish reference texts.

    Useful only for exercising pipeline mechanics; the audio does not contain
    speech.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    utterances = []
    for i in range(n):
        text = DEBUG_CORPUS[i % len(DEBUG_CORPUS)]
        duration = float(rng.uniform(1.2, 3.5))
        t = np.arange(int(duration * sampling_rate)) / sampling_rate
        freq = float(rng.uniform(80, 400))
        wave = 0.15 * np.sin(2 * np.pi * freq * t) + 0.02 * rng.standard_normal(t.shape)
        utterances.append(
            {
                "utt_id": f"synth-{i:04d}",
                "audio": wave.astype(np.float32),
                "sampling_rate": sampling_rate,
                "reference_raw": text,
                "duration_s": duration,
            }
        )
    return utterances
