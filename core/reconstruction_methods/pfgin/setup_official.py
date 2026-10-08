from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve()
PROJECT_ROOT = next(parent for parent in HERE.parents if (parent / "config.py").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def main():
    repo = config.PFGIN_REPO_DIR
    if (repo / "src" / "models.py").exists() and (repo / "kpn").exists():
        print(f"Official PFGIN already exists: {repo}")
        return

    repo.parent.mkdir(parents=True, exist_ok=True)
    if repo.exists():
        raise RuntimeError(
            f"{repo} exists but does not look like the official PFGIN repository. "
            "Rename/remove it, then run this script again."
        )

    print("Cloning official PFGIN repository...")
    subprocess.run(
        ["git", "clone", "--depth", "1", config.PFGIN_GIT_URL, str(repo)],
        check=True,
    )
    print("PFGIN repository installed.")
    print("Source: https://github.com/yinhuwu/PFGIN")


if __name__ == "__main__":
    main()
