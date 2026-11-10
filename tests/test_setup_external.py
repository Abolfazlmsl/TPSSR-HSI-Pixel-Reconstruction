import hashlib
import json
from pathlib import Path
import subprocess

import pytest

import setup_external


def _git(path, *args):
    subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)


def _repo(path, files):
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "Tester")
    _git(path, "config", "user.email", "test@example.com")
    for name, data in files.items():
        dest = path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    _git(path, "add", ".")
    _git(path, "commit", "-qm", "original")
    initial = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    first = path / next(iter(files))
    first.write_bytes(b"changed since original")
    _git(path, "add", ".")
    _git(path, "commit", "-qm", "newer")
    return initial


@pytest.mark.parametrize("name", ["PFGIN", "DDS2M"])
def test_installs_original_matching_commit(tmp_path, monkeypatch, name):
    old = b'os.path.join(image_folder, f"x_{iii}.mat")\n'
    new = b'os.path.join(image_folder, "latest.mat")\n'
    if name == "DDS2M":
        files = {"functions/denoising.py": old, "runners/VS2M.py": b"print('v1')\n"}
        expected = {"functions/denoising.py": new, "runners/VS2M.py": files["runners/VS2M.py"]}
    else:
        files = {"src/models.py": b"print('v1')\n", "kpn/network.py": b"pass\n"}
        expected = files

    upstream = tmp_path / (name + "_upstream")
    old_commit = _repo(upstream, files)
    folder = tmp_path / "external"
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"methods": {name: {
        "url": str(upstream),
        "files": {p: hashlib.sha256(blob).hexdigest() for p, blob in expected.items()}
    }}}), encoding="utf-8")
    monkeypatch.setattr(setup_external, "EXTERNAL", folder)
    monkeypatch.setattr(setup_external, "LOCK", lock)
    assert setup_external.main(["--method", name.lower()]) == 0
    assert setup_external.git("rev-parse", "HEAD", cwd=folder/name) == old_commit
    assert setup_external.mismatches(folder/name, name, {p: hashlib.sha256(b).hexdigest() for p,b in expected.items()}) == []
    assert setup_external.main(["--method", name.lower(), "--verify-only"]) == 0
    assert setup_external.main(["--method", name.lower()]) == 0


def test_refuses_incompatible_source(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    _repo(upstream, {"src/models.py": b"not expected"})
    monkeypatch.setattr(setup_external, "EXTERNAL", tmp_path / "external")
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"methods": {"PFGIN": {
        "url": str(upstream),
        "files": {"src/models.py": hashlib.sha256(b"expected").hexdigest()}
    }}}))
    monkeypatch.setattr(setup_external, "LOCK", lock)
    with pytest.raises(SystemExit) as exc:
        setup_external.main(["--method", "pfgin"])
    assert exc.value.code == 1
    assert not (setup_external.EXTERNAL/"PFGIN").exists()
