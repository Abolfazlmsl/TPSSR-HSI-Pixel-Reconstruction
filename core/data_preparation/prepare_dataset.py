from pathlib import Path
import sys

HERE = Path(__file__).resolve()
PROJECT_ROOT = next(
    parent for parent in HERE.parents
    if (parent / "config.py").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import config
from src.evaluation_utils import (
    load_dataset,
    split_blocks,
    save_mask_file,
    validate_wavelength_grid,
)
from src.saturation_simulation import (
    compute_difficulty_maps,
    generate_saturation_mask,
)
from src.envi_io import (
    auto_find_headers,
    calibrate_reflectance,
)


def save_spatial_preview(
    mask,
    path,
):
    projection = np.count_nonzero(
        mask,
        axis=2,
    )

    fig = plt.figure(
        figsize=(8, 5)
    )
    ax = fig.add_subplot(
        111
    )
    image = ax.imshow(
        projection,
        aspect="auto",
    )
    ax.set_xlabel(
        "x"
    )
    ax.set_ylabel(
        "test y"
    )
    ax.set_title(
        "Masked bands per spatial pixel"
    )
    fig.colorbar(
        image,
        ax=ax,
        label="Masked bands",
    )
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=160,
    )
    plt.close(
        fig
    )


def save_spectral_preview(
    mask,
    path,
):
    spectral_load = np.count_nonzero(
        mask,
        axis=(0, 1),
    )

    fig = plt.figure(
        figsize=(8, 4)
    )
    ax = fig.add_subplot(
        111
    )
    ax.plot(
        np.arange(
            len(
                spectral_load
            )
        ),
        spectral_load,
    )
    ax.set_xlabel(
        "Band index"
    )
    ax.set_ylabel(
        "Masked spatial pixels"
    )
    ax.set_title(
        "Spectral mask load"
    )
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=160,
    )
    plt.close(
        fig
    )



def enforce_strict_whole_spectrum_mask(
    raw_mask,
    valid_target,
):
    """
    Preserve the exact meaning of the whole_spectrum_pixels scenario.

    A selected spatial pixel is kept only when ground truth is valid in ALL
    spectral bands. Every kept pixel is then masked across the complete
    spectrum.

    Why this is necessary:
    the old generic operation

        mask &= valid_target

    could turn a 272/272 whole-spectrum target into 271/272 whenever the
    original cube already contained one invalid band at that spatial pixel.
    That made a small fraction of "whole_spectrum_pixels" targets partially
    spectral and incorrectly enabled a spectral anchor.

    Returns
    -------
    strict_mask : bool ndarray, same shape as raw_mask
    dropped_spatial_pixels : int
        Selected spatial pixels removed because at least one original band
        lacks valid ground truth.
    """
    raw_mask = np.asarray(
        raw_mask,
        dtype=bool,
    )

    valid_target = np.asarray(
        valid_target,
        dtype=bool,
    )

    if raw_mask.shape != valid_target.shape:
        raise ValueError(
            "raw_mask and valid_target must have identical shapes."
        )

    selected_spatial = np.any(
        raw_mask,
        axis=2,
    )

    fully_valid_spectrum = np.all(
        valid_target,
        axis=2,
    )

    keep_spatial = (
        selected_spatial
        & fully_valid_spectrum
    )

    dropped_spatial_pixels = int(
        np.count_nonzero(
            selected_spatial
            & ~fully_valid_spectrum
        )
    )

    strict_mask = np.zeros_like(
        raw_mask,
        dtype=bool,
    )

    strict_mask[
        keep_spatial,
        :
    ] = True

    # Hard sanity check: an affected pixel must have either zero masked bands
    # or every band masked. Partial spectra are forbidden in this scenario.
    masked_bands_per_pixel = np.count_nonzero(
        strict_mask,
        axis=2,
    )

    partial = (
        (masked_bands_per_pixel > 0)
        & (
            masked_bands_per_pixel
            < strict_mask.shape[2]
        )
    )

    if np.any(
        partial
    ):
        raise RuntimeError(
            "whole_spectrum_pixels strict-mask enforcement failed: "
            f"{int(np.count_nonzero(partial))} partial spectra remain."
        )

    return (
        strict_mask,
        dropped_spatial_pixels,
    )


def dataset_role(
    dataset_name,
):
    if (
        dataset_name
        in config.TEST_DATASET_FOLDERS
    ):
        return "held_out_test"

    if (
        dataset_name
        in config.TRAIN_DATASET_FOLDERS
    ):
        return "train_validation"

    raise ValueError(
        f"Dataset {dataset_name} is not configured."
    )


def write_split_info(
    output_dir,
    dataset_name,
    total_y,
):
    role = dataset_role(
        dataset_name
    )

    if role == "train_validation":
        train_end = int(
            round(
                total_y
                * config.TRAIN_IMAGE_TRAIN_FRACTION
            )
        )

        train_end = min(
            max(
                train_end,
                1,
            ),
            total_y - 1,
        )

        val_end = total_y
        test_start = total_y
        test_end = total_y

    else:
        test_size = max(
            1,
            int(
                round(
                    total_y
                    * config.HELD_OUT_TEST_FRACTION
                )
            ),
        )

        test_start = max(
            0,
            total_y - test_size,
        )

        # split_blocks() returns the third block from val_end onward.
        train_end = 0
        val_end = test_start
        test_end = total_y

    np.savez(
        output_dir
        / "split_info.npz",
        train_start=0,
        train_end=train_end,
        val_start=train_end,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
        role=np.asarray(
            role
        ),
        dataset_name=np.asarray(
            dataset_name
        ),
    )

    return {
        "role": role,
        "train_y": int(
            train_end
        ),
        "val_y": int(
            max(
                0,
                val_end - train_end,
            )
        ),
        "test_y": int(
            max(
                0,
                test_end - test_start,
            )
        ),
    }


def ensure_calibrated_dataset(
    dataset_name,
):
    input_dir = (
        config.INPUT_ROOT
        / dataset_name
    )
    output_dir = (
        config.PROCESSED_DATA_ROOT
        / dataset_name
    )

    calibration_files = [
        output_dir
        / "calibrated_reflectance.npy",
        output_dir
        / "invalid_mask.npy",
        output_dir
        / "wavelengths_nm.npy",
    ]

    if all(
        path.exists()
        for path in calibration_files
    ):
        print(
            f"{dataset_name}: using existing calibration"
        )
    else:
        print(
            f"{dataset_name}: calibrating reflectance"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            main_hdr,
            white_hdr,
            dark_hdr,
        ) = auto_find_headers(
            input_dir
        )

        (
            main_cube,
            reflectance,
            invalid,
        ) = calibrate_reflectance(
            main_hdr,
            white_hdr,
            dark_hdr,
        )

        np.save(
            output_dir
            / "calibrated_reflectance.npy",
            reflectance,
        )
        np.save(
            output_dir
            / "invalid_mask.npy",
            invalid,
        )
        np.save(
            output_dir
            / "wavelengths_nm.npy",
            main_cube.wavelengths,
        )

    cube = np.load(
        output_dir
        / "calibrated_reflectance.npy",
        mmap_mode="r",
    )
    wavelengths = np.load(
        output_dir
        / "wavelengths_nm.npy",
    )

    split_summary = write_split_info(
        output_dir,
        dataset_name,
        cube.shape[0],
    )

    return (
        tuple(
            int(value)
            for value in cube.shape
        ),
        np.asarray(
            wavelengths,
            dtype=np.float32,
        ),
        split_summary,
    )


def validate_all_datasets(
    dataset_rows,
):
    first = dataset_rows[0]
    reference_name = first[
        "dataset"
    ]
    reference_bands = first[
        "shape"
    ][2]
    reference_wavelengths = first[
        "wavelengths"
    ]

    for current in dataset_rows[
        1:
    ]:
        current_bands = current[
            "shape"
        ][2]

        if (
            current_bands
            != reference_bands
        ):
            raise RuntimeError(
                "All SWIR acquisitions must have the same number of bands. "
                f"{reference_name} has {reference_bands}; "
                f"{current['dataset']} has {current_bands}."
            )

        validate_wavelength_grid(
            reference_wavelengths,
            current[
                "wavelengths"
            ],
            reference_name,
            current[
                "dataset"
            ],
        )


def prepare_calibrated_datasets():
    missing_folders = [
        name
        for name in config.PREPARE_DATASETS
        if not (
            config.INPUT_ROOT
            / name
        ).exists()
        and not (
            config.PROCESSED_DATA_ROOT
            / name
            / "calibrated_reflectance.npy"
        ).exists()
    ]

    if missing_folders:
        raise FileNotFoundError(
            "The following dataset folders are missing from input_raw:\n"
            + "\n".join(
                missing_folders
            )
        )

    rows = []

    for dataset_name in config.PREPARE_DATASETS:
        (
            shape,
            wavelengths,
            split_summary,
        ) = ensure_calibrated_dataset(
            dataset_name
        )

        rows.append({
            "dataset": dataset_name,
            "shape": shape,
            "wavelengths": wavelengths,
            **split_summary,
        })

    validate_all_datasets(
        rows
    )

    config.BENCHMARK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        config.BENCHMARK_DIR
        / "dataset_manifest.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "role",
                "y",
                "x",
                "bands",
                "train_y",
                "val_y",
                "test_y",
            ],
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({
                "dataset": row[
                    "dataset"
                ],
                "role": row[
                    "role"
                ],
                "y": row[
                    "shape"
                ][0],
                "x": row[
                    "shape"
                ][1],
                "bands": row[
                    "shape"
                ][2],
                "train_y": row[
                    "train_y"
                ],
                "val_y": row[
                    "val_y"
                ],
                "test_y": row[
                    "test_y"
                ],
            })

    return rows


def main():
    dataset_rows = (
        prepare_calibrated_datasets()
    )

    mask_dir = (
        config.BENCHMARK_DIR
        / "masks"
    )
    preview_dir = (
        config.BENCHMARK_DIR
        / "mask_previews"
    )

    mask_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    preview_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in mask_dir.glob(
        "*.npz"
    ):
        path.unlink()

    (
        cube,
        invalid,
        split,
        wavelengths,
    ) = load_dataset(
        config.ACTIVE_TEST_DATASET
    )

    _, _, test = split_blocks(
        cube,
        split,
    )
    _, _, test_invalid = split_blocks(
        invalid,
        split,
    )

    if len(test) == 0:
        raise RuntimeError(
            "Held-out test block is empty."
        )

    print("")
    print(
        "Global training datasets:",
        ", ".join(
            config.TRAIN_DATASET_FOLDERS
        ),
    )
    print(
        "Active held-out test dataset:",
        config.ACTIVE_TEST_DATASET,
    )
    print(
        "Active held-out benchmark cube:",
        test.shape,
    )
    print("")

    print(
        "Computing spatial and spectral difficulty maps..."
    )

    margin = max(
        24,
        config.SPARSE_ROUTER_PATCH_SIZE // 2 + 2,
    )

    difficulty = compute_difficulty_maps(
        test,
        spatial_quantile=config.HARD_SPATIAL_QUANTILE,
        spectral_quantile=config.HARD_SPECTRAL_QUANTILE,
        margin=margin,
    )

    np.save(
        config.BENCHMARK_DIR
        / "spatial_difficulty.npy",
        difficulty[
            "spatial_score"
        ],
    )
    np.save(
        config.BENCHMARK_DIR
        / "challenging_spatial_pixels.npy",
        difficulty[
            "challenging_pixels"
        ],
    )
    np.save(
        config.BENCHMARK_DIR
        / "spectral_curvature.npy",
        difficulty[
            "spectral_curvature"
        ],
    )
    np.save(
        config.BENCHMARK_DIR
        / "challenging_bands.npy",
        difficulty[
            "challenging_bands"
        ],
    )

    with (
        config.BENCHMARK_DIR
        / "challenging_bands.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(
            handle
        )
        writer.writerow([
            "band",
            "wavelength_nm",
            "curvature",
        ])

        for band in difficulty[
            "challenging_bands"
        ]:
            writer.writerow([
                int(
                    band
                ),
                float(
                    wavelengths[
                        int(
                            band
                        )
                    ]
                ),
                float(
                    difficulty[
                        "spectral_curvature"
                    ][
                        int(
                            band
                        )
                    ]
                ),
            ])

    rows = []

    total = (
        len(
            config.SCENARIOS
        )
        * len(
            config.SEVERITY_LEVELS
        )
        * config.N_REPEATS
    )

    test_finite = np.isfinite(
        test
    )
    valid_target = (
        ~np.asarray(
            test_invalid,
            dtype=bool,
        )
        & test_finite
    )

    with tqdm(
        total=total,
        desc=(
            "Creating saturation simulation masks"
        ),
    ) as progress:
        for (
            scenario_index,
            scenario,
        ) in enumerate(
            config.SCENARIOS
        ):
            for severity in (
                config.SEVERITY_LEVELS
            ):
                for repeat in range(
                    1,
                    config.N_REPEATS + 1,
                ):
                    seed = (
                        config.SEED
                        + scenario_index
                        * 100000
                        + severity
                        * 1000
                        + repeat
                        * 37
                    )

                    rng = (
                        np.random.default_rng(
                            seed
                        )
                    )

                    raw_mask = (
                        generate_saturation_mask(
                            test.shape,
                            scenario,
                            severity,
                            rng,
                            difficulty[
                                "challenging_pixels"
                            ],
                            difficulty[
                                "challenging_bands"
                            ],
                        )
                    )

                    dropped_invalid_spatial_pixels = 0

                    if (
                        scenario
                        == "whole_spectrum_pixels"
                    ):
                        (
                            mask,
                            dropped_invalid_spatial_pixels,
                        ) = enforce_strict_whole_spectrum_mask(
                            raw_mask,
                            valid_target,
                        )
                    else:
                        mask = (
                            raw_mask
                            & valid_target
                        )

                    filename = (
                        f"{scenario}"
                        f"__s{severity:02d}"
                        f"__r{repeat:02d}.npz"
                    )

                    save_mask_file(
                        mask_dir
                        / filename,
                        mask,
                        {
                            "scenario": scenario,
                            "severity": severity,
                            "repeat": repeat,
                            "seed": seed,
                        },
                    )

                    affected_pixels = int(
                        np.count_nonzero(
                            np.any(
                                mask,
                                axis=2,
                            )
                        )
                    )
                    masked_voxels = int(
                        np.count_nonzero(
                            mask
                        )
                    )

                    rows.append({
                        "scenario": scenario,
                        "severity": severity,
                        "repeat": repeat,
                        "seed": seed,
                        "masked_voxels": masked_voxels,
                        "affected_spatial_pixels": affected_pixels,
                        "masked_fraction": (
                            masked_voxels
                            / mask.size
                        ),
                        "mean_masked_bands_per_affected_pixel": (
                            masked_voxels
                            / affected_pixels
                            if affected_pixels
                            else 0.0
                        ),
                        "dropped_invalid_spatial_pixels": (
                            dropped_invalid_spatial_pixels
                        ),
                    })

                    if repeat == 1:
                        prefix = (
                            f"{scenario}"
                            f"__s{severity:02d}"
                        )

                        save_spatial_preview(
                            mask,
                            preview_dir
                            / f"{prefix}__spatial.png",
                        )
                        save_spectral_preview(
                            mask,
                            preview_dir
                            / f"{prefix}__spectral.png",
                        )

                    progress.update(
                        1
                    )

    summary_path = (
        config.BENCHMARK_DIR
        / "mask_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            rows
        )

    info = [
        "HSI SATURATION RECONSTRUCTION DATASET",
        "",
        f"Experiment: {config.EXPERIMENT_NAME}",
        (
            "Global training datasets: "
            + ", ".join(
                config.TRAIN_DATASET_FOLDERS
            )
        ),
        (
            "Active held-out test dataset: "
            + config.ACTIVE_TEST_DATASET
        ),
        (
            "All configured held-out datasets: "
            + ", ".join(config.TEST_DATASET_FOLDERS)
        ),
        (
            "Training split per training acquisition: "
            f"{config.TRAIN_IMAGE_TRAIN_FRACTION:.0%} train / "
            f"{config.TRAIN_IMAGE_VAL_FRACTION:.0%} validation"
        ),
        (
            "Held-out acquisition fraction used for benchmark: "
            f"{config.HELD_OUT_TEST_FRACTION:.0%}"
        ),
        (
            f"Held-out test cube shape: {test.shape}"
        ),
        f"Scenarios: {len(config.SCENARIOS)}",
        f"Severity levels: {config.SEVERITY_LEVELS}",
        f"Repeats per severity: {config.N_REPEATS}",
        f"Total fixed masks: {total}",
        "",
        (
            "The held-out test acquisition is never used for model "
            "training or validation."
        ),
        (
            "Global normalization statistics are computed only from "
            "the training partitions of the training acquisitions."
        ),
        (
            "Training datasets are sampled uniformly so one acquisition "
            "does not dominate solely because of its size."
        ),
        "",
        (
            "Severity increases corruption size, density, missing spectral "
            "width, or placement difficulty, depending on the scenario."
        ),
        (
            "High severities are preferentially placed on strong spatial "
            "gradients and high-curvature spectral regions."
        ),
        "",
        (
            "single_detector_line always contains exactly one detector "
            "element."
        ),
        (
            "Its severity is controlled by increasingly difficult "
            "spatial/spectral placement."
        ),
        (
            "mixed_saturation adds corruption families progressively "
            "from severity 1 to 5."
        ),
        (
            "Complete-band loss is included only in mixed_saturation "
            "severity 5."
        ),
        (
            "For whole_spectrum_pixels, every retained affected spatial "
            "pixel is masked in every spectral band. Spatial pixels whose "
            "original spectrum contains any invalid ground-truth band are "
            "removed from this scenario rather than converted into partial "
            "spectral targets."
        ),
        "",
        (
            "Evaluation samples complete affected spatial pixels so that "
            "all masked bands of each selected spectrum remain grouped "
            "for RMSE and SAM."
        ),
    ]

    (
        config.BENCHMARK_DIR
        / "dataset_info.txt"
    ).write_text(
        "\n".join(
            info
        ),
        encoding="utf-8",
    )

    print("")
    print(
        "Global dataset preparation completed."
    )
    print(
        summary_path
    )


if __name__ == "__main__":
    main()
