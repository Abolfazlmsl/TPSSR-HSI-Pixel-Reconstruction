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
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

import config
from src.evaluation_utils import (
    load_global_training_data,
)
from src.ml_common import (
    band_stats,
    extract_patch,
    rf_feature,
)
from src.training_corruptions import (
    generate_training_mask,
)


def main():
    (
        train_cubes,
        _,
        wavelengths,
    ) = load_global_training_data()

    model_dir = (
        config.MODELS_DIR
        / "random_forest"
    )
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    band_mean, _ = band_stats(
        train_cubes
    )

    ry = (
        config.RF_PATCH_Y
        // 2
    )
    rx = (
        config.RF_PATCH_X
        // 2
    )
    rb = (
        config.RF_PATCH_B
        // 2
    )

    feature_dim = (
        2
        * config.RF_PATCH_Y
        * config.RF_PATCH_X
        * config.RF_PATCH_B
        + 1
    )

    X = np.empty(
        (
            config.RF_TRAIN_SAMPLES,
            feature_dim,
        ),
        dtype=np.float32,
    )
    y_target = np.empty(
        config.RF_TRAIN_SAMPLES,
        dtype=np.float32,
    )

    rng = np.random.default_rng(
        config.SEED + 800
    )

    filled = 0

    progress = tqdm(
        total=config.RF_TRAIN_SAMPLES,
        desc="Global RF training samples",
    )

    while (
        filled
        < config.RF_TRAIN_SAMPLES
    ):
        cube_index = int(
            rng.integers(
                0,
                len(
                    train_cubes
                ),
            )
        )
        train = train_cubes[
            cube_index
        ]

        y_size, x_size, b_size = (
            train.shape
        )

        if (
            y_size <= 2 * ry
            or x_size <= 2 * rx
            or b_size <= 2 * rb
        ):
            continue

        y = int(
            rng.integers(
                ry,
                y_size - ry,
            )
        )
        x = int(
            rng.integers(
                rx,
                x_size - rx,
            )
        )
        b = int(
            rng.integers(
                rb,
                b_size - rb,
            )
        )

        values, _ = extract_patch(
            train,
            y,
            x,
            b,
            config.RF_PATCH_Y,
            config.RF_PATCH_X,
            config.RF_PATCH_B,
            mask=None,
        )

        target = float(
            values[
                ry,
                rx,
                rb,
            ]
        )

        if not np.isfinite(
            target
        ):
            continue

        bad, _, _ = (
            generate_training_mask(
                values.shape,
                rng,
                severity=int(
                    rng.integers(
                        1,
                        6,
                    )
                ),
            )
        )

        bad |= (
            ~np.isfinite(
                values
            )
        )

        X[
            filled
        ] = rf_feature(
            values,
            bad,
            band_mean,
            b,
        )
        y_target[
            filled
        ] = target

        filled += 1
        progress.update(
            1
        )

    progress.close()

    print(
        "Global RF training datasets:",
        ", ".join(
            config.TRAIN_DATASET_FOLDERS
        ),
    )
    print(
        "Held-out test dataset:",
        config.TEST_DATASET_FOLDER,
    )

    model = RandomForestRegressor(
        n_estimators=0,
        warm_start=True,
        max_depth=config.RF_MAX_DEPTH,
        max_features=0.35,
        n_jobs=config.RF_N_JOBS,
        random_state=config.SEED,
    )

    tree_counts = list(
        range(
            config.RF_TREE_BATCH,
            config.RF_N_TREES + 1,
            config.RF_TREE_BATCH,
        )
    )

    if (
        tree_counts[
            -1
        ]
        != config.RF_N_TREES
    ):
        tree_counts.append(
            config.RF_N_TREES
        )

    for n_trees in tqdm(
        tree_counts,
        desc="RF trees",
    ):
        model.set_params(
            n_estimators=n_trees
        )
        model.fit(
            X,
            y_target,
        )

    path = (
        model_dir
        / "random_forest_model.joblib"
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
