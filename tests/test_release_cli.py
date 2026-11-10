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
