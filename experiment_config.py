from __future__ import annotations

from dataclasses import dataclass
import argparse
import os
import re
from pathlib import Path
from typing import Iterable

DATASET_PATTERN = re.compile(r"^(SWIR|VNIR)[1-9][0-9]*$")
SUPPORTED_DATASETS = [
    *(f"SWIR{i}" for i in range(1, 11)),
    *(f"VNIR{i}" for i in range(1, 11)),
]


def discover_folder_split(root: Path | None = None) -> tuple[list[str], list[str]]:
    root = root or Path(__file__).resolve().parent / "input_raw"
    groups = []
    for role in ("train", "test"):
        folder = root / role
        if not folder.is_dir():
            raise ValueError(f"Missing data directory: {folder}")
        names = sorted((p.name.upper() for p in folder.iterdir() if p.is_dir()),
                       key=lambda x: (x[:4], int(re.search(r"[0-9]+$", x).group()) if DATASET_PATTERN.fullmatch(x) else 0))
        invalid = [name for name in names if not DATASET_PATTERN.fullmatch(name)]
        if invalid:
            raise ValueError(f"Invalid acquisition names in {folder}: {', '.join(invalid)}")
        if not names:
            raise ValueError(f"No acquisitions found in {folder}")
        groups.append(names)
    train, tests = groups
    overlap = set(train) & set(tests)
    if overlap:
        raise ValueError("Acquisitions in both train and test: " + ", ".join(sorted(overlap)))
    if len(train) < 4:
        raise ValueError("TPSSR training requires at least four training acquisitions: two for model fitting and two for source-only validation.")
    if len({name[:4] for name in train + tests}) != 1:
        raise ValueError("Train and test must use the same sensor (SWIR or VNIR).")
    return train, tests

PRESETS = {
    "custom": {"description": "Discover acquisitions from input_raw/train and input_raw/test"},
    "swir_8x2": {
        "tests": ["SWIR5", "SWIR10"],
        "description": "8 SWIR train + 2 SWIR test (one from each 5-core region)",
    },
    "swir_region_a_to_b": {
        "train": [f"SWIR{i}" for i in range(1, 6)],
        "tests": [f"SWIR{i}" for i in range(6, 11)],
        "description": "Train on SWIR1-5 and test on SWIR6-10",
    },
    "swir_region_b_to_a": {
        "train": [f"SWIR{i}" for i in range(6, 11)],
        "tests": [f"SWIR{i}" for i in range(1, 6)],
        "description": "Train on SWIR6-10 and test on SWIR1-5",
    },
    "vnir_8x2": {
        "tests": ["VNIR5", "VNIR10"],
        "description": "8 VNIR train + 2 VNIR test (one from each 5-core region)",
    },
}


def _flatten(values: Iterable[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            part = part.strip().upper()
            if part:
                out.append(part)
    return out


def _unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for value in values:
        value = str(value).upper()
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def sensor_of(name: str) -> str:
    name = str(name).upper()
    if name.startswith("SWIR"):
        return "SWIR"
    if name.startswith("VNIR"):
        return "VNIR"
    raise ValueError(f"Unknown dataset name: {name}")


def sensor_universe(sensor: str) -> list[str]:
    sensor = str(sensor).upper()
    return [f"{sensor}{i}" for i in range(1, 11)]


def _validate_dataset_names(values: list[str]) -> None:
    bad = [x for x in values if not DATASET_PATTERN.fullmatch(x)]
    if bad:
        raise ValueError(
            "Unsupported dataset(s): "
            + ", ".join(bad)
            + ". Acquisition names must follow SWIRn or VNIRn."
        )


@dataclass(frozen=True)
class ExperimentSpec:
    train: tuple[str, ...]
    tests: tuple[str, ...]
    active_test: str
    experiment_name: str
    sensor: str
    folder_mode: bool = False

    @property
    def primary_test(self) -> str:
        # Kept as the internal model/checkpoint tag for backward compatibility
        # with the original TPSSR training code.
        return self.tests[0]

    @property
    def model_tag(self) -> str:
        return self.primary_test.lower()

    def with_active_test(self, dataset: str) -> "ExperimentSpec":
        dataset = str(dataset).upper()
        if dataset not in self.tests:
            raise ValueError(
                f"Active test dataset {dataset} is not in configured test set: "
                + ", ".join(self.tests)
            )
        return ExperimentSpec(
            train=self.train,
            tests=self.tests,
            active_test=dataset,
            experiment_name=self.experiment_name,
            sensor=self.sensor,
            folder_mode=self.folder_mode,
        )


def default_experiment_name(train: list[str], tests: list[str]) -> str:
    sensor = sensor_of((train or tests)[0]).lower()
    train_part = "-".join(x.replace(sensor.upper(), "") for x in train)
    test_part = "-".join(x.replace(sensor.upper(), "") for x in tests)
    return f"{sensor}__train_{train_part}__test_{test_part}"


def _spec_from_environment(active_override: str | None = None) -> ExperimentSpec | None:
    train_env = os.environ.get("HSI_TRAIN_DATASETS")
    tests_env = os.environ.get("HSI_TEST_DATASETS")
    if not train_env or not tests_env:
        return None
    train = _unique(_flatten([train_env]))
    tests = _unique(_flatten([tests_env]))
    active = (
        str(active_override).upper()
        if active_override
        else os.environ.get("HSI_ACTIVE_TEST_DATASET", tests[0]).upper()
    )
    name = os.environ.get(
        "HSI_EXPERIMENT_NAME",
        default_experiment_name(train, tests),
    )
    return _finalize_spec(train, tests, active, name,
                          folder_mode=os.environ.get("HSI_INPUT_LAYOUT") == "folders")


def _finalize_spec(
    train: list[str],
    tests: list[str],
    active_test: str | None,
    experiment_name: str | None,
    folder_mode: bool = False,
) -> ExperimentSpec:
    train = _unique(train)
    tests = _unique(tests)
    _validate_dataset_names(train + tests)
    if not train:
        raise ValueError("At least one training dataset is required.")
    if not tests:
        raise ValueError("At least one test dataset is required.")

    sensors = {sensor_of(x) for x in train + tests}
    if len(sensors) != 1:
        raise ValueError(
            "One experiment must use a single sensor type. "
            "Do not mix SWIR and VNIR in the same train/test experiment."
        )
    sensor = next(iter(sensors))
    overlap = sorted(set(train) & set(tests))
    if overlap:
        raise ValueError(
            "Training and test sets overlap: " + ", ".join(overlap)
        )

    active = str(active_test or tests[0]).upper()
    if active not in tests:
        raise ValueError(
            f"Active test dataset {active} must be one of: {', '.join(tests)}"
        )
    name = str(experiment_name or default_experiment_name(train, tests))
    return ExperimentSpec(
        train=tuple(train),
        tests=tuple(tests),
        active_test=active,
        experiment_name=name,
        sensor=sensor,
        folder_mode=folder_mode,
    )


def resolve_experiment(args, *, allow_legacy_dataset: bool = True) -> ExperimentSpec:
    # Child/worker processes inherit a fully resolved experiment through env.
    env_spec = _spec_from_environment(getattr(args, "active_test", None))
    explicit_train = _flatten(getattr(args, "train_datasets", None))
    explicit_tests = _flatten(getattr(args, "test_datasets", None))
    preset = getattr(args, "preset", None)
    legacy_dataset = getattr(args, "dataset", None)
    explicit_name = getattr(args, "experiment_name", None)

    from_folders = getattr(args, "from_folders", False)
    folder_root = Path(__file__).resolve().parent / "input_raw"
    if from_folders:
        if any((explicit_train, explicit_tests, legacy_dataset, preset not in (None, "custom"))):
            raise ValueError("--from-folders cannot be combined with explicit dataset lists or a preset.")
        train, tests = discover_folder_split(folder_root)
        return _finalize_spec(train, tests, getattr(args, "active_test", None), explicit_name, folder_mode=True)

    if not any((explicit_train, explicit_tests, preset, legacy_dataset, explicit_name)):
        if env_spec is not None:
            return env_spec
        if (folder_root / "train").is_dir() and (folder_root / "test").is_dir():
            train, tests = discover_folder_split(folder_root)
            return _finalize_spec(train, tests, getattr(args, "active_test", None), None, folder_mode=True)

    train: list[str] = []
    tests: list[str] = []

    if preset:
        info = PRESETS[preset]
        train = list(info.get("train", []))
        tests = list(info.get("tests", []))

    if explicit_tests:
        tests = explicit_tests
    if explicit_train:
        train = explicit_train

    # Backward-compatible single-fold mode used by the original clean project.
    if legacy_dataset and not train and not tests:
        ds = str(legacy_dataset).upper()
        _validate_dataset_names([ds])
        sensor = sensor_of(ds)
        # Preserve the original 5-acquisition LOIO behavior for --dataset.
        legacy_universe = [f"{sensor}{i}" for i in range(1, 6)]
        if ds not in legacy_universe:
            # For SWIR6..10 / VNIR6..10, use the 10-acquisition universe.
            legacy_universe = sensor_universe(sensor)
        tests = [ds]
        train = [x for x in legacy_universe if x != ds]

    # If only tests are supplied, derive train as the complement within 1..10.
    # This makes the desired 8/2 and 5/5 experiments concise.
    if tests and not train:
        _validate_dataset_names(tests)
        sensor = sensor_of(tests[0])
        if any(sensor_of(x) != sensor for x in tests):
            raise ValueError("All test datasets must use the same sensor.")
        train = [x for x in sensor_universe(sensor) if x not in set(tests)]

    # If only training datasets are supplied, use the sensor-complement as tests.
    if train and not tests:
        _validate_dataset_names(train)
        sensor = sensor_of(train[0])
        if any(sensor_of(x) != sensor for x in train):
            raise ValueError("All training datasets must use the same sensor.")
        tests = [x for x in sensor_universe(sensor) if x not in set(train)]

    if not train and not tests and env_spec is not None:
        return env_spec
    if preset == "custom" and not train and not tests:
        train, tests = discover_folder_split(folder_root)

    if not train or not tests:
        raise ValueError(
            "Specify --preset, --test-datasets, --train-datasets, or the legacy --dataset option."
        )

    return _finalize_spec(
        train,
        tests,
        getattr(args, "active_test", None),
        explicit_name,
        folder_mode=(preset == "custom" or os.environ.get("HSI_INPUT_LAYOUT") == "folders"),
    )


def add_experiment_args(
    parser: argparse.ArgumentParser,
    *,
    include_legacy_dataset: bool = True,
) -> None:
    parser.add_argument(
        "--from-folders", action="store_true",
        help="Read acquisition names from input_raw/train and input_raw/test.",
    )
    if include_legacy_dataset:
        parser.add_argument(
            "--dataset",
            default=None,
            help=(
                "Backward-compatible held-out acquisition selector. "
                "For new datasets use --from-folders."
            ),
        )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=None,
        help="Ready-made experiment split.",
    )
    parser.add_argument(
        "--train-datasets",
        nargs="+",
        default=None,
        help="Explicit training acquisitions, e.g. SWIR1 SWIR2 ...",
    )
    parser.add_argument(
        "--test-datasets",
        nargs="+",
        default=None,
        help=(
            "Explicit held-out acquisitions. For automatic discovery use --from-folders."
        ),
    )
    parser.add_argument(
        "--active-test",
        default=None,
        help="Internal/single-test selector. Normally omitted by the user.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional custom folder name for models/results.",
    )


def experiment_cli_args(spec: ExperimentSpec, *, active_test: str | None = None) -> list[str]:
    active = str(active_test or spec.active_test).upper()
    return [
        "--train-datasets",
        *spec.train,
        "--test-datasets",
        *spec.tests,
        "--active-test",
        active,
        "--experiment-name",
        spec.experiment_name,
    ]


def describe(spec: ExperimentSpec) -> str:
    return (
        f"Experiment: {spec.experiment_name}\n"
        f"Sensor: {spec.sensor}\n"
        f"Train ({len(spec.train)}): {', '.join(spec.train)}\n"
        f"Test ({len(spec.tests)}): {', '.join(spec.tests)}\n"
        f"Active test: {spec.active_test}\n"
        f"Internal model tag: {spec.primary_test}"
    )


def spectral_validation_cores(spec: ExperimentSpec) -> tuple[str, str]:
    """Two acquisition-held-out validation cores from the training partition.

    Preserve the original 8x2 split exactly. Region transfer uses two cores
    from the source region, never from the target region.
    """
    choices = {
        ("SWIR5", "SWIR10"): ("SWIR4", "SWIR9"),
        ("SWIR6", "SWIR7", "SWIR8", "SWIR9", "SWIR10"): ("SWIR4", "SWIR5"),
        ("SWIR1", "SWIR2", "SWIR3", "SWIR4", "SWIR5"): ("SWIR9", "SWIR10"),
        ("VNIR5", "VNIR10"): ("VNIR4", "VNIR9"),
    }
    names = choices.get(tuple(spec.tests))
    if not spec.folder_mode and names is not None and set(names).issubset(set(spec.train)):
        return names
    if len(spec.train) < 4:
        raise ValueError("At least four training acquisitions are needed for source-only validation.")
    return tuple(spec.train[-2:])


def selected_lineaware_region(spec: ExperimentSpec, router: str) -> bool:
    """Select split-specific Line-aware checkpoints for final TPSSR routing."""
    if router in ('learned_router3_lineaware', 'learned_router3_lineaware_mse', 'learned_router3_lineaware_rankcv', 'learned_router3_lineaware_regretcv', 'learned_router2_lineaware'):
        allowed = {
            ('SWIR5','SWIR10'): ('SWIR1','SWIR2','SWIR3','SWIR4','SWIR6','SWIR7','SWIR8','SWIR9'),
            ('SWIR6','SWIR7','SWIR8','SWIR9','SWIR10'): ('SWIR1','SWIR2','SWIR3','SWIR4','SWIR5'),
            ('SWIR1','SWIR2','SWIR3','SWIR4','SWIR5'): ('SWIR6','SWIR7','SWIR8','SWIR9','SWIR10'),
            ('VNIR5','VNIR10'): ('VNIR1','VNIR2','VNIR3','VNIR4','VNIR6','VNIR7','VNIR8','VNIR9'),
        }
        if spec.folder_mode:
            return len(spec.train) >= 4
        known = allowed.get(spec.tests)
        return spec.train == known if known is not None else len(spec.train) >= 4
    if router != 'learned_router3_aligned':
        return False
    swir8 = (spec.train == ('SWIR1','SWIR2','SWIR3','SWIR4','SWIR6','SWIR7','SWIR8','SWIR9')
             and spec.tests == ('SWIR5','SWIR10'))
    a2b_trial = (os.environ.get('TPSSR_A2B_LINEAWARE_TEST') == '1'
                 and spec.train == ('SWIR1','SWIR2','SWIR3','SWIR4','SWIR5')
                 and spec.tests == ('SWIR6','SWIR7','SWIR8','SWIR9','SWIR10'))
    b2a_trial = (os.environ.get('TPSSR_B2A_LINEAWARE_TEST') == '1'
                 and spec.train == ('SWIR6','SWIR7','SWIR8','SWIR9','SWIR10')
                 and spec.tests == ('SWIR1','SWIR2','SWIR3','SWIR4','SWIR5'))
    return swir8 or a2b_trial or b2a_trial


def selected_results_folder(spec: ExperimentSpec, router: str) -> str:
    """Keep accepted Line-aware results distinct from previous checkpoint runs."""
    base = {
        'v23':'tpssr', 'neighbor_v1':'tpssr_neighbor_v1',
        'neighbor_v2':'tpssr_neighbor_v2', 'learned_fusion':'tpssr_learned_fusion',
        'learned_fusion_guarded':'tpssr_learned_fusion_guarded',
        'learned_router':'tpssr_learned_router',
        'learned_router3':'tpssr_learned_router3',
        'learned_router3_rmse':'tpssr_learned_router3_rmse',
        'learned_router3_aligned':'tpssr_learned_router3_aligned',
        'learned_router3_lineaware':'tpssr_learned_router3_lineaware',
        'learned_router3_lineaware_mse':'tpssr_learned_router3_lineaware_mse',
        'learned_router3_lineaware_rankcv':'tpssr_learned_router3_lineaware_rankcv',
        'learned_router3_lineaware_regretcv':'tpssr_learned_router3_lineaware_regretcv',
        'learned_router2_lineaware':'tpssr_learned_router2_lineaware',
    }[router]
    if router in ('learned_router3_lineaware', 'learned_router3_lineaware_mse', 'learned_router3_lineaware_rankcv', 'learned_router3_lineaware_regretcv', 'learned_router2_lineaware'):
        return base
    if (os.environ.get('TPSSR_A2B_LINEAWARE_TEST') == '1'
            and spec.train == ('SWIR1','SWIR2','SWIR3','SWIR4','SWIR5')
            and spec.tests == ('SWIR6','SWIR7','SWIR8','SWIR9','SWIR10')):
        return base + '_lineaware_a2b_trial'
    if (os.environ.get('TPSSR_B2A_LINEAWARE_TEST') == '1'
            and router == 'learned_router3_aligned'
            and spec.train == ('SWIR6','SWIR7','SWIR8','SWIR9','SWIR10')
            and spec.tests == ('SWIR1','SWIR2','SWIR3','SWIR4','SWIR5')):
        return base + '_lineaware_b2a_trial'
    return base + ('_lineaware' if selected_lineaware_region(spec, router) else '')
