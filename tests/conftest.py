import numpy as np
import pytest

from turkish_asr.debug_model import build_debug_model, make_synthetic_utterances


@pytest.fixture(scope="session")
def debug_model_dir(tmp_path_factory):
    """A tiny local Whisper (random weights, local BPE tokenizer), built once."""
    out = tmp_path_factory.mktemp("debug_model")
    return build_debug_model(out, seed=0)


@pytest.fixture(scope="session")
def debug_processor(debug_model_dir):
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(debug_model_dir)
    processor.tokenizer.set_prefix_tokens(language="turkish", task="transcribe")
    return processor


def write_synthetic_manifest(root, n=5, seed=0, split="test"):
    """Write n synthetic wav files and return a FLEURS-shaped manifest Dataset."""
    import soundfile as sf
    from datasets import Audio, Dataset

    wav_dir = root / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for utt in make_synthetic_utterances(n, seed=seed):
        path = wav_dir / f"{utt['utt_id']}.wav"
        sf.write(path, utt["audio"], utt["sampling_rate"])
        rows.append(
            {
                "utt_id": utt["utt_id"],
                "audio": str(path),
                "reference_raw": utt["reference_raw"],
                "duration_s": utt["duration_s"],
                "dataset": "fleurs",
                "split": split,
                "age": None,
                "gender": "female" if int(utt["utt_id"].split("-")[1]) % 2 else "male",
                "accents": None,
                "speaker_id": None,
            }
        )
    ds = Dataset.from_list(rows)
    return ds.cast_column("audio", Audio(sampling_rate=16000))


@pytest.fixture()
def synthetic_manifest(tmp_path):
    return write_synthetic_manifest(tmp_path, n=5, seed=0)


@pytest.fixture()
def rng():
    return np.random.default_rng(0)
