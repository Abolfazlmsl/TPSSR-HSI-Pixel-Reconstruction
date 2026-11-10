from types import SimpleNamespace

import pytest

import experiment_config as experiments


def make_folders(root, train, test):
    for name in train:
        (root / "train" / name).mkdir(parents=True, exist_ok=True)
    for name in test:
        (root / "test" / name).mkdir(parents=True, exist_ok=True)


def test_discovery_and_validation(tmp_path):
    raw = tmp_path / "input_raw"
    make_folders(raw, ["SWIR1", "SWIR2", "SWIR3", "SWIR4"], ["SWIR15"])
    train, tests = experiments.discover_folder_split(raw)
    assert train == ["SWIR1", "SWIR2", "SWIR3", "SWIR4"]
    assert tests == ["SWIR15"]
    spec = experiments._finalize_spec(train, tests, None, None)
    assert experiments.spectral_validation_cores(spec) == ("SWIR3", "SWIR4")
    assert experiments.selected_lineaware_region(spec, "learned_router3_lineaware_rankcv")


def test_duplicate_acquisition_is_rejected(tmp_path):
    raw = tmp_path / "input_raw"
    make_folders(raw, ["SWIR1", "SWIR2", "SWIR3", "SWIR4"], ["SWIR4"])
    with pytest.raises(ValueError, match="both train and test"):
        experiments.discover_folder_split(raw)


def test_mixed_sensor_is_rejected(tmp_path):
    raw = tmp_path / "input_raw"
    make_folders(raw, ["SWIR1", "SWIR2", "SWIR3", "SWIR4"], ["VNIR5"])
    with pytest.raises(ValueError, match="same sensor"):
        experiments.discover_folder_split(raw)


def test_presets_unchanged():
    args = SimpleNamespace(from_folders=False, preset="swir_8x2", active_test=None,
                           train_datasets=None, test_datasets=None, dataset=None, experiment_name=None)
    spec = experiments.resolve_experiment(args)
    assert spec.tests == ("SWIR5", "SWIR10")
    assert spec.folder_mode is False
    assert experiments.spectral_validation_cores(spec) == ("SWIR4", "SWIR9")


def test_core_role_directories_and_shadow_validation(tmp_path):
    import json
    import os
    import subprocess
    import sys

    raw = tmp_path / "input_raw"
    make_folders(raw, ["SWIR1", "SWIR2", "SWIR3", "SWIR4"], ["SWIR5"])
    env = os.environ.copy()
    env.update(TPSSR_PROJECT_DIR=str(tmp_path), HSI_TRAIN_DATASETS="SWIR1,SWIR2,SWIR3",
               HSI_TEST_DATASETS="SWIR4", HSI_TEST_DATASET="SWIR4",
               HSI_ACTIVE_TEST_DATASET="SWIR4", HSI_INPUT_LAYOUT="folders")
    core = experiments.Path(__file__).resolve().parents[1] / "core"
    code = (
        "import config, json; "
        "print(json.dumps([str(config.input_dir_for('SWIR1')), "
        "str(config.input_dir_for('SWIR4')), config.HELD_OUT_TEST_FRACTION]))"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=str(core), env=env,
                            check=True, capture_output=True, text=True)
    train_source, shadow_test, fraction = json.loads(result.stdout)
    assert train_source == str(raw / "train" / "SWIR1")
    assert shadow_test == str(raw / "train" / "SWIR4")
    assert fraction == 1.0
