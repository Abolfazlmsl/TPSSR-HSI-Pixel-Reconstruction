from pathlib import Path
import os

# Root visible to the user. Internal code lives under core/.
PROJECT_DIR = Path(
    os.environ.get(
        "TPSSR_PROJECT_DIR",
        str(Path(__file__).resolve().parent.parent),
    )
).resolve()

SUPPORTED_DATASETS = [
    *(f"SWIR{i}" for i in range(1, 11)),
    *(f"VNIR{i}" for i in range(1, 11)),
]


def _parse_dataset_env(name, default=None):
    raw = os.environ.get(name)
    if raw is None:
        return list(default or [])
    values = []
    for part in raw.replace(",", " ").split():
        item = part.strip().upper()
        if item and item not in values:
            values.append(item)
    return values


# -----------------------------------------------------------------------------
# Experiment split
# -----------------------------------------------------------------------------
# HSI_TEST_DATASET is deliberately the PRIMARY held-out acquisition and stays
# fixed for one trained model. The original TPSSR training code uses this value
# in checkpoint folder names. HSI_ACTIVE_TEST_DATASET can change when the same
# trained model is evaluated on several held-out acquisitions.
TEST_DATASET_FOLDERS = _parse_dataset_env("HSI_TEST_DATASETS")
if not TEST_DATASET_FOLDERS:
    TEST_DATASET_FOLDERS = [os.environ.get("HSI_TEST_DATASET", "SWIR1").upper()]

TEST_DATASET_FOLDER = os.environ.get(
    "HSI_TEST_DATASET", TEST_DATASET_FOLDERS[0]
).upper()
if TEST_DATASET_FOLDER not in TEST_DATASET_FOLDERS:
    TEST_DATASET_FOLDERS = [TEST_DATASET_FOLDER, *TEST_DATASET_FOLDERS]

ACTIVE_TEST_DATASET = os.environ.get(
    "HSI_ACTIVE_TEST_DATASET", TEST_DATASET_FOLDERS[0]
).upper()

for _name in [*TEST_DATASET_FOLDERS, ACTIVE_TEST_DATASET]:
    if _name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset {_name!r}. Choose SWIR1..SWIR10 or VNIR1..VNIR10."
        )

SENSOR = "SWIR" if TEST_DATASET_FOLDER.startswith("SWIR") else "VNIR"
if any(("SWIR" if x.startswith("SWIR") else "VNIR") != SENSOR for x in TEST_DATASET_FOLDERS):
    raise ValueError("One experiment cannot mix SWIR and VNIR test acquisitions.")

TRAIN_DATASET_FOLDERS = _parse_dataset_env("HSI_TRAIN_DATASETS")
if not TRAIN_DATASET_FOLDERS:
    # Backward-compatible original 5-acquisition LOIO default.
    default_universe = [f"{SENSOR}{i}" for i in range(1, 6)]
    if TEST_DATASET_FOLDER not in default_universe:
        default_universe = [f"{SENSOR}{i}" for i in range(1, 11)]
    TRAIN_DATASET_FOLDERS = [
        x for x in default_universe if x != TEST_DATASET_FOLDER
    ]

for _name in TRAIN_DATASET_FOLDERS:
    if _name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported training dataset {_name!r}.")
    if ("SWIR" if _name.startswith("SWIR") else "VNIR") != SENSOR:
        raise ValueError("One experiment cannot mix SWIR and VNIR training acquisitions.")

_overlap = set(TRAIN_DATASET_FOLDERS) & set(TEST_DATASET_FOLDERS)
if _overlap:
    raise ValueError(
        "Training and test datasets overlap: " + ", ".join(sorted(_overlap))
    )
if ACTIVE_TEST_DATASET not in TEST_DATASET_FOLDERS:
    raise ValueError(
        f"ACTIVE_TEST_DATASET={ACTIVE_TEST_DATASET} is not in TEST_DATASET_FOLDERS."
    )

# The legacy deep/Geometry training scripts validate that ALL_SENSOR_DATASETS minus
# TEST_DATASET_FOLDER equals the training set. Keep this compatibility universe
# intentionally limited to training acquisitions + the primary held-out tag.
ALL_SENSOR_DATASETS = list(TRAIN_DATASET_FOLDERS) + [TEST_DATASET_FOLDER]
ALL_SWIR_DATASETS = ALL_SENSOR_DATASETS  # backward-compatible historical name

# Data preparation only needs all training acquisitions and the currently active
# held-out acquisition. Re-running prepare_dataset.py for the next held-out core
# reuses calibrated training files and creates a separate benchmark mask folder.
PREPARE_DATASETS = list(TRAIN_DATASET_FOLDERS) + [ACTIVE_TEST_DATASET]

EXPERIMENT_NAME = os.environ.get(
    "HSI_EXPERIMENT_NAME",
    "global_" + "_".join(TRAIN_DATASET_FOLDERS) + "__test_" + "_".join(TEST_DATASET_FOLDERS),
)

INPUT_ROOT = PROJECT_DIR / "input_raw"
PROCESSED_DATA_ROOT = PROJECT_DIR / "processed_data"
DATASET_FOLDER = ACTIVE_TEST_DATASET
INPUT_DIR = INPUT_ROOT / ACTIVE_TEST_DATASET
PROCESSED_DATA_DIR = PROCESSED_DATA_ROOT / ACTIVE_TEST_DATASET
BASE_DATASET_DIR = PROCESSED_DATA_DIR

# One model family is shared by all held-out acquisitions in the experiment.
MODELS_DIR = PROJECT_DIR / "trained_models" / EXPERIMENT_NAME
# Masks/results are separate for each held-out acquisition.
SATURATION_DATA_DIR = (
    PROJECT_DIR / "simulated_saturation" / EXPERIMENT_NAME / ACTIVE_TEST_DATASET
)
BENCHMARK_DIR = SATURATION_DATA_DIR
RESULTS_DIR = PROJECT_DIR / "results" / EXPERIMENT_NAME / ACTIVE_TEST_DATASET

# Each training acquisition is split spatially along y.
TRAIN_IMAGE_TRAIN_FRACTION = 0.80
TRAIN_IMAGE_VAL_FRACTION = 0.20

# The held-out acquisition is never used for training.
# Only its final block is used for the fixed 200-mask benchmark so the
# computational scale remains comparable with the previous experiment.
HELD_OUT_TEST_FRACTION = 0.20

# Kept for backward compatibility with utilities that may inspect them.
TRAIN_FRACTION = TRAIN_IMAGE_TRAIN_FRACTION
VAL_FRACTION = TRAIN_IMAGE_VAL_FRACTION
TEST_FRACTION = HELD_OUT_TEST_FRACTION

# All acquisitions must represent the same SWIR spectral grid.
WAVELENGTH_MATCH_ATOL_NM = 0.50

RAW_CLIPPING_CODE = 65535
EPS = 1e-6

SEED = 4207

SCENARIOS = [
    "isolated_voxels",
    "single_detector_line",
    "multiple_detector_lines",
    "partial_vertical_lines",
    "spatial_patch",
    "consecutive_spectral_bands",
    "spatio_spectral_block",
    "whole_spectrum_pixels",
    "band_wide_saturation",
    "mixed_saturation",
]

SEVERITY_LEVELS = [1, 2, 3, 4, 5]
N_REPEATS = 4

# Evaluation is sampled by spatial pixel so that all masked bands from the
# same affected spectrum stay grouped.
MAX_EVAL_SPATIAL_PIXELS = 5000
MAX_EVAL_VOXELS = 250000

# Synthetic saturation targeting.
HARD_SPATIAL_QUANTILE = 0.75
HARD_SPECTRAL_QUANTILE = 0.70
SPARSE_COORD_THRESHOLD = 500000

# PCA.
PCA_COMPONENTS = 40
PCA_TRAIN_SPECTRA = 200000
PCA_ITERATIONS = 10

# Random Forest.
RF_PATCH_Y = 7
RF_PATCH_X = 7
RF_PATCH_B = 15
RF_TRAIN_SAMPLES = 80000
RF_N_TREES = 250
RF_TREE_BATCH = 25
RF_MAX_DEPTH = 30
RF_N_JOBS = -1

# Published DDS2M baseline.
# Official implementation: https://github.com/miaoyuchun/DDS2M
DDS2M_REPO_DIR = PROJECT_DIR / "external" / "DDS2M"
DDS2M_GIT_URL = "https://github.com/miaoyuchun/DDS2M.git"

# DDS2M is self-supervised and is optimized independently for each corrupted
# benchmark tile. These defaults keep the original public implementation's
# diffusion schedule and command-line defaults.
DDS2M_GPU_IDS = [0, 1]
# Backward-compatible single-GPU alias.
DDS2M_GPU_ID = DDS2M_GPU_IDS[0]
DDS2M_TILE_SIZE = 128
DDS2M_TILE_STRIDE = 64
DDS2M_TIMESTEPS = 3000
DDS2M_START_POINT = 1500
DDS2M_RANK = 10
DDS2M_SIGMA_0 = 0.0
DDS2M_ETA = 0.95
DDS2M_ETA_B = 1.0
DDS2M_BETA = 0.0
DDS2M_INNER_ITERATIONS = 0
DDS2M_LEARNING_RATE = 5e-4

# Set to an integer such as 1 for a quick installation/smoke test.
# Keep None for the complete 200-mask benchmark.
DDS2M_METHOD_NAME = "dds2m_t3000_sp1500"
DDS2M_MAX_MASKS = None
DDS2M_RESUME = True
DDS2M_VERBOSE_SUBPROCESS = False
DDS2M_KEEP_OFFICIAL_OUTPUTS = False

# Published PFGIN baseline.
# Official implementation: https://github.com/yinhuwu/PFGIN
PFGIN_REPO_DIR = PROJECT_DIR / "external" / "PFGIN"
PFGIN_GIT_URL = "https://github.com/yinhuwu/PFGIN.git"

# PFGIN uses 16-band hyperspectral windows in the official repository.
PFGIN_SPECTRAL_WINDOW = 16
PFGIN_SPECTRAL_STRIDE = 8
PFGIN_PATCH_SIZE = 128
PFGIN_TILE_STRIDE = 64
PFGIN_BATCH_SIZE = 4
PFGIN_TEST_BATCH_SIZE = 4
PFGIN_GPU_IDS = [0]
PFGIN_NUM_WORKERS = 4

# The public PFGIN config uses 40,000 optimization iterations.
# Keep SMOKE_TEST=True for the first installation check; then change it to False
# and rerun train.py. Training resumes from the smoke-test checkpoint.
PFGIN_SMOKE_TEST = False
PFGIN_SMOKE_ITERS = 50
PFGIN_MAX_ITERS = 40000
PFGIN_LEARNING_RATE = 1e-4
PFGIN_D2G_LR = 0.1
PFGIN_BETA1 = 0.0
PFGIN_BETA2 = 0.9
PFGIN_EVAL_INTERVAL = 2000
PFGIN_VAL_SAMPLES = 64
PFGIN_RESUME = True

# Official loss weights from the public PFGIN checkpoints/config.yml.
PFGIN_L1_LOSS_WEIGHT = 1.0
PFGIN_FM_LOSS_WEIGHT = 10.0
PFGIN_STYLE_LOSS_WEIGHT = 250.0
PFGIN_CONTENT_LOSS_WEIGHT = 0.1
PFGIN_ADV_LOSS_WEIGHT = 0.1
PFGIN_GAN_LOSS = "nsgan"
PFGIN_KERNEL_SIZE = [3, 7, 3, 11]
PFGIN_MODE = "ss"

# Robust per-band normalization is fitted ONLY on the training acquisitions.
PFGIN_SCALER_SPECTRA = 200000
PFGIN_SCALER_LOW_Q = 0.005
PFGIN_SCALER_HIGH_Q = 0.995

# Set to 1 for a quick test of the evaluation pipeline; None evaluates all masks.
PFGIN_MAX_MASKS = None


# Context margin used by sparse masked-voxel evaluation.
SPARSE_ROUTER_PATCH_SIZE = 31
