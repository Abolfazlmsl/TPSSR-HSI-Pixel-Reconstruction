from pathlib import Path
import sys

HERE = Path(__file__).resolve()
PROJECT_ROOT = next(
    parent for parent in HERE.parents
    if (parent / "config.py").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
from sklearn.decomposition import PCA

import config
from src.evaluation_utils import (
    load_global_training_data,
)
from src.ml_common import band_stats


def main():
    (
        train_cubes,
        _,
        wavelengths,
    ) = load_global_training_data()

    model_dir = (
        config.MODELS_DIR
        / "pca"
    )
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    band_mean, _ = band_stats(
        train_cubes
    )

    rng = np.random.default_rng(
        config.SEED + 700
    )

    spectra = []
    attempts = 0

    while (
        len(spectra)
        < config.PCA_TRAIN_SPECTRA
        and attempts
        < config.PCA_TRAIN_SPECTRA
        * 30
    ):
        attempts += 1

        cube_index = int(
            rng.integers(
                0,
                len(
                    train_cubes
                ),
            )
        )
        cube = train_cubes[
            cube_index
        ]

        y = int(
            rng.integers(
                0,
                cube.shape[0],
            )
        )
        x = int(
            rng.integers(
                0,
                cube.shape[1],
            )
        )

        spectrum = np.asarray(
            cube[
                y,
                x,
                :,
            ],
            dtype=np.float32,
        )

        if np.all(
            np.isfinite(
                spectrum
            )
        ):
            spectra.append(
                spectrum
            )

    if (
        len(spectra)
        < config.PCA_COMPONENTS + 1
    ):
        raise RuntimeError(
            "Not enough complete spectra for global PCA training."
        )

    X = np.asarray(
        spectra,
        dtype=np.float32,
    )

    print(
        "Global PCA training datasets:",
        ", ".join(
            config.TRAIN_DATASET_FOLDERS
        ),
    )
    print(
        "Held-out test dataset:",
        config.TEST_DATASET_FOLDER,
    )
    print(
        "PCA training spectra:",
        len(
            X
        ),
    )

    model = PCA(
        n_components=min(
            config.PCA_COMPONENTS,
            X.shape[1],
        ),
        svd_solver="randomized",
        random_state=config.SEED,
    )
    model.fit(
        X
    )

    path = (
        model_dir
        / "pca_model.joblib"
    )

    joblib.dump(
        {
            "model": model,
            "band_mean": band_mean,
            "wavelengths_nm": np.asarray(
                wavelengths,
                dtype=np.float32,
            ),
            "training_datasets": list(
                config.TRAIN_DATASET_FOLDERS
            ),
            "test_dataset": (
                config.TEST_DATASET_FOLDER
            ),
        },
        path,
        compress=3,
    )

    print(
        "Saved:"
    )
    print(
        path
    )


if __name__ == "__main__":
    main()
