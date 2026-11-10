from types import SimpleNamespace

import pytest

import tpssr
from evaluate import results_folder


def test_final_test_command(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tpssr.subprocess, "run", fake_run)
    assert tpssr.main(["test", "--preset", "swir_8x2"]) == 0
    command, options = calls[0]
    assert command[1].endswith("test.py")
    assert command[command.index("--router") + 1] == tpssr.FINAL_ROUTER
    assert command[command.index("--minority-policy") + 1] == "hybrid_gap"
    assert command[command.index("--cache-mode") + 1] == "lazy"
    assert options["env"]["TPSSR_FUSION_DONOR_MATCH"] == "robust"


def test_evaluate_points_to_final_results(monkeypatch):
    calls = []
    monkeypatch.setattr(tpssr.subprocess, "run", lambda cmd, **kw: (
        calls.append(cmd) or SimpleNamespace(returncode=0)))
    assert tpssr.main(["evaluate", "--preset", "swir_8x2", "--visuals", "0"]) == 0
    cmd = calls[0]
    assert cmd[cmd.index("--results-folder") + 1] == tpssr.RESULTS_FOLDER
    assert cmd[cmd.index("--visuals") + 1] == "0"


def test_evaluate_folder_override():
    spec = SimpleNamespace(experiment_name="test")
    assert results_folder(spec, "v23", override="custom_result") == "custom_result"
    with pytest.raises(ValueError):
        results_folder(spec, "v23", override="../untrusted")


def test_external_source_lock():
    import json
    source_lock = tpssr.ROOT / "external_source_lock.json"
    data = json.loads(source_lock.read_text(encoding="utf-8"))
    assert set(data["methods"]) == {"DDS2M", "PFGIN"}
    assert "functions/denoising.py" in data["methods"]["DDS2M"]["files"]
    assert "src/models.py" in data["methods"]["PFGIN"]["files"]


def test_setup_external_cli(monkeypatch):
    calls = []
    monkeypatch.setattr(tpssr.subprocess, "run", lambda cmd, **kw: (
        calls.append(cmd) or SimpleNamespace(returncode=0)))
    assert tpssr.main(["setup-external", "--method", "pfgin"]) == 0
    assert calls[0][1].endswith("setup_external.py")
    assert calls[0][-2:] == ["--method", "pfgin"]


def test_external_cli_uses_same_experiment(monkeypatch):
    calls = []
    monkeypatch.setattr(tpssr.subprocess, "run", lambda cmd, **kw: (
        calls.append((cmd, kw)) or SimpleNamespace(returncode=0)))
    assert tpssr.main(["external", "--from-folders", "--method", "all", "--stage", "test"]) == 0
    assert calls[0][0][1].endswith("external_methods.py")
    assert "--from-folders" in calls[0][0]
