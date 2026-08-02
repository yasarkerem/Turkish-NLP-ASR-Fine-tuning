"""Five-sample smoke test of the full train -> evaluate -> analyze loop.

Uses the locally built debug Whisper and synthetic audio, but goes through
the exact CLI entrypoints and cached-manifest machinery used by real runs.
This must pass before any real training run is launched.
"""

import json

import pytest

from turkish_asr import analyze as analyze_mod
from turkish_asr import evaluate as evaluate_mod
from turkish_asr import train as train_mod
from turkish_asr.config import load_config
from turkish_asr.data import manifest_cache_dir

from conftest import write_synthetic_manifest

N_SMOKE = 5


@pytest.fixture(scope="module")
def workspace(tmp_path_factory, debug_model_dir):
    """Isolated data/cache/runs/models tree with a pre-seeded manifest cache."""
    root = tmp_path_factory.mktemp("smoke_ws")
    overrides = [
        f"paths.data_root={root / 'data'}",
        f"paths.cache_root={root / 'cache'}",
        f"paths.runs_dir={root / 'runs'}",
        f"paths.models_dir={root / 'models'}",
        f"model.id={debug_model_dir}",
    ]
    cfg = load_config("configs/smoke.yaml", overrides)
    manifest = write_synthetic_manifest(root, n=N_SMOKE, seed=0)
    cache = manifest_cache_dir(cfg, "fleurs", "test", N_SMOKE)
    cache.parent.mkdir(parents=True, exist_ok=True)
    manifest.save_to_disk(str(cache))
    # The uncapped manifest cache is used by evaluate when --max-samples is
    # not passed; seed it too so nothing tries to download FLEURS.
    cache_all = manifest_cache_dir(cfg, "fleurs", "test", None)
    manifest.save_to_disk(str(cache_all))
    return {"root": root, "overrides": overrides, "cfg": cfg}


def test_smoke_train(workspace, debug_model_dir):
    rc = train_mod.main(
        ["--config", "configs/smoke.yaml"]
        + [f"--set={o}" for o in workspace["overrides"]]
    )
    assert rc == 0
    adapter = workspace["root"] / "models" / "smoke" / "adapter"
    assert (adapter / "adapter_model.safetensors").exists()
    run_dirs = list((workspace["root"] / "runs").glob("*_train_smoke"))
    assert len(run_dirs) == 1
    run = json.loads((run_dirs[0] / "run.json").read_text())
    assert run["status"] == "completed"
    assert run["metrics"]["n_train"] == N_SMOKE
    assert run["metrics"]["trainable_params"] < run["metrics"]["total_params"]
    assert "train_loss" in run["metrics"]


def test_smoke_evaluate_and_analyze(workspace, debug_model_dir):
    adapter = workspace["root"] / "models" / "smoke" / "adapter"
    assert adapter.exists(), "training smoke test must run first"
    rc = evaluate_mod.main(
        [
            "--config", "configs/smoke.yaml",
            *[f"--set={o}" for o in workspace["overrides"]],
            "--model", str(debug_model_dir),
            "--adapter", str(adapter),
            "--dataset", "fleurs",
            "--split", "test",
            "--max-samples", str(N_SMOKE),
            "--batch-size", "2",
            "--max-new-tokens", "8",
            "--rtf-utterances", "3",
            "--run-name", "smoke-eval",
        ]
    )
    assert rc == 0
    run_dirs = list((workspace["root"] / "runs").glob("*_eval_smoke-eval"))
    assert len(run_dirs) == 1
    run = json.loads((run_dirs[0] / "run.json").read_text())
    assert run["status"] == "completed"
    wc = run["metrics"]["wer_cer"]
    for key in ("wer_raw", "wer_norm", "cer_raw", "cer_norm"):
        assert isinstance(wc[key], float) and wc[key] >= 0.0
    assert wc["n_utterances"] == N_SMOKE
    assert run["metrics"]["rtf"]["n_utterances"] == 3
    assert run["metrics"]["rtf"]["rtf_overall"] > 0
    assert run["metrics"]["memory"]["peak_rss_mib"] > 0

    preds = (run_dirs[0] / "predictions.jsonl").read_text().strip().splitlines()
    assert len(preds) == N_SMOKE

    out_md = workspace["root"] / "analysis.md"
    rc = analyze_mod.main(
        [
            "--config", "configs/smoke.yaml",
            *[f"--set={o}" for o in workspace["overrides"]],
            "--runs", str(run_dirs[0]),
            "--examples", "3",
            "--min-slice-n", "1",
            "--emit-markdown", str(out_md),
            "--out-json", str(workspace["root"] / "analysis.json"),
        ]
    )
    assert rc == 0
    text = out_md.read_text()
    assert "WER by duration bucket" in text
    assert "Turkish-specific characters" in text
    analysis = json.loads((workspace["root"] / "analysis.json").read_text())[0]
    assert analysis["aggregate"]["n_utterances"] == N_SMOKE
    assert analysis["metadata_slices"]["gender"]["coverage"] == 1.0
