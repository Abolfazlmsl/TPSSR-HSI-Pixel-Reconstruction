#!/usr/bin/env python3
"""Install the external benchmarks from version-matched upstream sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
EXTERNAL = ROOT / "external"
LOCK = ROOT / "external_source_lock.json"

DDS_ORIGINAL = b'os.path.join(image_folder, f"x_{iii}.mat")'
DDS_COMPAT = b'os.path.join(image_folder, "latest.mat")'


def git(*args, cwd=None):
    result = subprocess.run(
        ["git", *map(str, args)], cwd=cwd, capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(map(str, args))} failed")
    return result.stdout.strip()


def normalized(data):
    return data.replace(b"\r\n", b"\n")


def content_for_check(method, name, data):
    data = normalized(data)
    if method == "DDS2M" and name == "functions/denoising.py":
        data = data.replace(DDS_ORIGINAL, DDS_COMPAT)
    return data


def mismatches(folder, method, checks):
    wrong = []
    for name, digest in checks.items():
        source = folder / name
        if not source.is_file():
            wrong.append(name)
            continue
        content = content_for_check(method, name, source.read_bytes())
        if hashlib.sha256(content).hexdigest() != digest:
            wrong.append(name)
    return wrong


def patch_dds2m(folder):
    path = folder / "functions" / "denoising.py"
    data = path.read_bytes()
    if DDS_COMPAT in data:
        return
    if data.count(DDS_ORIGINAL) != 1:
        raise RuntimeError("The expected DDS2M output path was not found")
    path.write_bytes(data.replace(DDS_ORIGINAL, DDS_COMPAT))
    print("DDS2M: enabled latest.mat output storage")


def install(method, source):
    folder = EXTERNAL / method
    checks = source["files"]
    if folder.exists():
        problems = mismatches(folder, method, checks)
        if problems:
            raise RuntimeError(
                f"{folder} already exists but differs from the locked version "
                f"({', '.join(problems[:5])}). Move it elsewhere before installing."
            )
        print(f"{method}: matching source already installed")
        return

    with tempfile.TemporaryDirectory(prefix=f".{method}.install.", dir=EXTERNAL) as tmp:
        checkout = Path(tmp) / "repo"
        print(f"{method}: cloning {source['url']}", flush=True)
        git("clone", "--quiet", source["url"], checkout)
        commits = git("rev-list", "--all", cwd=checkout).splitlines()
        selected = None
        for sha in commits:
            git("-c", "advice.detachedHead=false", "checkout", "--quiet", "--detach", sha, cwd=checkout)
            if not mismatches(checkout, method, checks):
                selected = sha
                break
        if selected is None:
            differences = mismatches(checkout, method, checks)
            raise RuntimeError(
                f"No commit in {source['url']} matches the original {method} source. "
                f"Last mismatch: {', '.join(differences[:5])}. "
                "No incompatible version was installed."
            )

        if method == "DDS2M":
            patch_dds2m(checkout)
            if mismatches(checkout, method, checks):
                raise RuntimeError("DDS2M source verification failed after patching")
        shutil.move(str(checkout), str(folder))
        print(f"{method}: verified and installed at commit {selected}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download version-matched DDS2M/PFGIN sources")
    parser.add_argument("--method", choices=("all", "dds2m", "pfgin"), default="all")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)

    if shutil.which("git") is None and not args.verify_only:
        parser.error("Git is required. Install Git and run this command again.")

    with LOCK.open(encoding="utf-8") as stream:
        sources = json.load(stream)["methods"]
    EXTERNAL.mkdir(exist_ok=True)
    names = ["PFGIN", "DDS2M"] if args.method == "all" else [args.method.upper()]

    try:
        for name in names:
            if args.verify_only:
                folder = EXTERNAL / name
                problems = mismatches(folder, name, sources[name]["files"])
                if problems:
                    raise RuntimeError(f"{name}: not installed or source differs ({', '.join(problems[:5])})")
                print(f"{name}: source verified")
            else:
                install(name, sources[name])
    except RuntimeError as exc:
        parser.exit(1, f"Setup failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
