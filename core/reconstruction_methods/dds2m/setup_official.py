from pathlib import Path
import argparse
import subprocess
import sys

HERE = Path(__file__).resolve()
PROJECT_ROOT = next(parent for parent in HERE.parents if (parent / "config.py").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def run(command, cwd=None):
    print(" ".join(str(v) for v in command))
    subprocess.run(command, cwd=cwd, check=True)


def clone_official_repo():
    repo = config.DDS2M_REPO_DIR
    if (repo / ".git").exists():
        print(f"Official DDS2M repository already exists: {repo}")
        return repo

    repo.parent.mkdir(parents=True, exist_ok=True)
    try:
        run(["git", "clone", config.DDS2M_GIT_URL, str(repo)])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Could not clone the official DDS2M repository. Check the Internet "
            "connection, then run setup_official.py again. You can also clone "
            f"{config.DDS2M_GIT_URL} manually into {repo}."
        ) from exc
    return repo


def patch_output_storage(repo):
    """Make upstream DDS2M overwrite latest.mat instead of saving every step."""
    path = repo / "functions" / "denoising.py"
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")
    original = 'os.path.join(image_folder, f"x_{iii}.mat")'
    replacement = 'os.path.join(image_folder, "latest.mat")'

    if replacement in text:
        print("DDS2M output-storage patch is already applied.")
        return
    if original not in text:
        raise RuntimeError(
            "The official DDS2M denoising.py no longer matches the expected "
            "public implementation. No automatic patch was applied."
        )

    path.write_text(text.replace(original, replacement), encoding="utf-8")
    print("Applied disk-saving patch to DDS2M/functions/denoising.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--install-requirements",
        action="store_true",
        help="Also install the official DDS2M requirements into this Python environment.",
    )
    args = parser.parse_args()

    repo = clone_official_repo()
    patch_output_storage(repo)

    if args.install_requirements:
        if sys.version_info >= (3, 11):
            print(
                "WARNING: the upstream repository pins very old NumPy/SciPy/PyTorch "
                "versions. Those pins may not install on this Python version. "
                "A separate Python 3.9/3.10 environment is safer."
            )
        run([sys.executable, "-m", "pip", "install", "-r", str(repo / "requirements.txt")])

    print("\nDDS2M setup completed.")
    print(f"Official repository: {repo}")
    print("\nNext command:")
    print("python reconstruction_methods/dds2m/test.py")


if __name__ == "__main__":
    main()
