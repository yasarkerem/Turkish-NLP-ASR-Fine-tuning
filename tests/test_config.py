import pytest

from turkish_asr.config import load_config


def test_inherit_and_override(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 1\ntrain:\n  lr: 0.001\n  batch_size: 4\n")
    (tmp_path / "child.yaml").write_text("inherit: base.yaml\ntrain:\n  lr: 0.01\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg["seed"] == 1
    assert cfg["train"]["lr"] == 0.01
    assert cfg["train"]["batch_size"] == 4


def test_cli_overrides_parse_yaml_values(tmp_path):
    (tmp_path / "c.yaml").write_text("seed: 1\n")
    cfg = load_config(tmp_path / "c.yaml", ["train.max_steps=null", "train.lr=1e-4", "seed=7"])
    assert cfg["seed"] == 7
    assert cfg["train"]["max_steps"] is None
    assert cfg["train"]["lr"] == pytest.approx(1e-4)


def test_bad_override_rejected(tmp_path):
    (tmp_path / "c.yaml").write_text("seed: 1\n")
    with pytest.raises(ValueError):
        load_config(tmp_path / "c.yaml", ["no_equals_sign"])


def test_repo_configs_load():
    for name in ["default", "smoke", "train_small_lora", "train_tiny_full"]:
        cfg = load_config(f"configs/{name}.yaml")
        assert cfg["seed"] == 42
        assert cfg["audio"]["sampling_rate"] == 16000
