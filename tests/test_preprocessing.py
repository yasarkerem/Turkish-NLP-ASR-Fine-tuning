import numpy as np

from turkish_asr.data import build_features, manifest_stats
from turkish_asr.train import DataCollatorSpeechSeq2SeqWithPadding

EXPECTED_FRAMES = 3000  # 30 s at 16 kHz with hop 160
EXPECTED_MELS = 80


class TestFeatureExtraction:
    def test_log_mel_shape_and_dtype(self, debug_processor, rng):
        audio = rng.standard_normal(16000 * 2).astype(np.float32)
        feats = debug_processor.feature_extractor(audio, sampling_rate=16000)
        arr = np.asarray(feats.input_features[0])
        assert arr.shape == (EXPECTED_MELS, EXPECTED_FRAMES)
        assert arr.dtype == np.float32
        assert np.isfinite(arr).all()

    def test_short_audio_padded_to_30s(self, debug_processor, rng):
        audio = rng.standard_normal(1600).astype(np.float32)  # 0.1 s
        arr = np.asarray(debug_processor.feature_extractor(audio, sampling_rate=16000).input_features[0])
        assert arr.shape == (EXPECTED_MELS, EXPECTED_FRAMES)

    def test_labels_carry_prefix_and_eos(self, debug_processor):
        tok = debug_processor.tokenizer
        ids = tok("merhaba dünya").input_ids
        assert ids[0] == tok.convert_tokens_to_ids("<|startoftranscript|>")
        assert tok.convert_tokens_to_ids("<|tr|>") in ids[:4]
        assert tok.convert_tokens_to_ids("<|transcribe|>") in ids[:4]
        assert ids[-1] == tok.eos_token_id

    def test_tokenizer_roundtrip_preserves_turkish_chars(self, debug_processor):
        text = "ığdır'da şöför çğöşü"
        tok = debug_processor.tokenizer
        decoded = tok.decode(tok(text).input_ids, skip_special_tokens=True)
        assert decoded.strip() == text


class TestBuildFeatures:
    def test_dataset_shapes_and_dtypes(self, synthetic_manifest, debug_processor):
        feats = build_features(synthetic_manifest, debug_processor)
        assert len(feats) == len(synthetic_manifest)
        assert set(feats.column_names) == {"input_features", "labels"}
        arr = feats.with_format("numpy")["input_features"][0]
        assert arr.shape == (EXPECTED_MELS, EXPECTED_FRAMES)
        assert arr.dtype == np.float32
        labels = feats[0]["labels"]
        assert all(isinstance(t, int) for t in labels)
        vocab = len(debug_processor.tokenizer)
        assert all(0 <= t < vocab for t in labels)

    def test_collator_batch_shapes(self, synthetic_manifest, debug_processor):
        import torch

        feats = build_features(synthetic_manifest, debug_processor)
        collator = DataCollatorSpeechSeq2SeqWithPadding(debug_processor)
        batch = collator([feats[i] for i in range(4)])
        assert batch["input_features"].shape == (4, EXPECTED_MELS, EXPECTED_FRAMES)
        assert batch["input_features"].dtype == torch.float32
        assert batch["labels"].dtype == torch.long
        # Leading <|startoftranscript|> is stripped (re-added by the model's
        # shift-right), and padding is masked with -100.
        sot = debug_processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        assert (batch["labels"][:, 0] != sot).all()
        lengths = [len(feats[i]["labels"]) for i in range(4)]
        if len(set(lengths)) > 1:
            assert (batch["labels"] == -100).any()

    def test_resampling_to_16k(self, tmp_path):
        import soundfile as sf
        from datasets import Audio, Dataset

        rng = np.random.default_rng(1)
        path = tmp_path / "a.wav"
        sf.write(path, rng.standard_normal(44100).astype(np.float32), 44100)
        ds = Dataset.from_list([{"audio": str(path)}]).cast_column(
            "audio", Audio(sampling_rate=16000)
        )
        audio = ds[0]["audio"]
        assert audio["sampling_rate"] == 16000
        assert abs(len(audio["array"]) - 16000) <= 2


class TestManifestStats:
    def test_stats_fields(self, synthetic_manifest):
        stats = manifest_stats(synthetic_manifest)
        assert stats["n_utterances"] == 5
        assert sum(stats["duration_buckets"].values()) == 5
        assert 0 <= stats["pct_refs_with_turkish_specific_chars"] <= 100
        assert stats["total_audio_hours"] > 0
