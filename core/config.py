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
    *(f"SWIR{i}" for i in range(1, 6)),
    *(f"VNIR{i}" for i in range(1, 6)),
]

TEST_DATASET_FOLDER = os.environ.get("HSI_TEST_DATASET", "SWIR1").upper()
if TEST_DATASET_FOLDER not in SUPPORTED_DATASETS:
    raise ValueError(
        f"Unsupported held-out dataset {TEST_DATASET_FOLDER!r}. "
        f"Choose one of: {', '.join(SUPPORTED_DATASETS)}"
    )

SENSOR = "SWIR" if TEST_DATASET_FOLDER.startswith("SWIR") else "VNIR"
ALL_SENSOR_DATASETS = [f"{SENSOR}{i}" for i in range(1, 6)]
# Backward-compatible name used by the original preparation code.
ALL_SWIR_DATASETS = ALL_SENSOR_DATASETS

TRAIN_DATASET_FOLDERS = [
    name for name in ALL_SENSOR_DATASETS if name != TEST_DATASET_FOLDER
]

EXPERIMENT_NAME = (
    "global_" + "_".join(TRAIN_DATASET_FOLDERS) + "__test_" + TEST_DATASET_FOLDER
)

INPUT_ROOT = PROJECT_DIR / "input_raw"
PROCESSED_DATA_ROOT = PROJECT_DIR / "processed_data"
DATASET_FOLDER = TEST_DATASET_FOLDER
INPUT_DIR = INPUT_ROOT / TEST_DATASET_FOLDER
PROCESSED_DATA_DIR = PROCESSED_DATA_ROOT / TEST_DATASET_FOLDER
BASE_DATASET_DIR = PROCESSED_DATA_DIR

SATURATION_DATA_DIR = PROJECT_DIR / "simulated_saturation" / EXPERIMENT_NAME
BENCHMARK_DIR = SATURATION_DATA_DIR
MODELS_DIR = PROJECT_DIR / "trained_models" / EXPERIMENT_NAME
RESULTS_DIR = PROJECT_DIR / "results" / EXPERIMENT_NAME

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

# Confidence-aware sparse hybrid router.
SPARSE_ROUTER_PATCH_SIZE = 31
SPARSE_ROUTER_TRAIN_SAMPLES_PER_EPOCH = 12000
SPARSE_ROUTER_VAL_SAMPLES = 4000
SPARSE_ROUTER_EPOCHS = 70
SPARSE_ROUTER_BATCH_SIZE = 8
SPARSE_ROUTER_TEST_BATCH_SIZE = 16
SPARSE_ROUTER_LEARNING_RATE = 1.8e-4
SPARSE_ROUTER_WEIGHT_DECAY = 1e-4
SPARSE_ROUTER_NUM_WORKERS = 4
SPARSE_ROUTER_EARLY_STOPPING_PATIENCE = 15
SPARSE_ROUTER_MIN_EPOCHS = 25
SPARSE_ROUTER_GRAD_CLIP = 1.0
SPARSE_ROUTER_EMA_DECAY = 0.999
SPARSE_ROUTER_USE_TTA = True
SPARSE_ROUTER_RESUME = True

SPARSE_ROUTER_MODEL_DIM = 192
SPARSE_ROUTER_ATTENTION_HEADS = 8
SPARSE_ROUTER_TRANSFORMER_LAYERS = 6
SPARSE_ROUTER_FF_MULTIPLIER = 4
SPARSE_ROUTER_DROPOUT = 0.05

SPARSE_ROUTER_TEMPERATURE = 0.10

SPARSE_ROUTER_UNCERTAINTY_SCORE_WEIGHT = 1.40
SPARSE_ROUTER_BIAS_SCALE = 0.35
SPARSE_ROUTER_ANALYTICAL_CONFIDENCE_WEIGHT = 0.35
SPARSE_ROUTER_ERROR_FLOOR = 1e-4

SPARSE_ROUTER_DERIVATIVE_LOSS_WEIGHT = 0.18
SPARSE_ROUTER_SAM_LOSS_WEIGHT = 0.08
SPARSE_ROUTER_CONTEXT_LOSS_WEIGHT = 0.015
SPARSE_ROUTER_DEEP_LOSS_WEIGHT = 0.18

SPARSE_ROUTER_CLASSIFICATION_LOSS_WEIGHT = 0.45
SPARSE_ROUTER_UNCERTAINTY_LOSS_WEIGHT = 0.30
SPARSE_ROUTER_RANKING_LOSS_WEIGHT = 0.12
SPARSE_ROUTER_REGRET_LOSS_WEIGHT = 0.35
SPARSE_ROUTER_ENTROPY_LOSS_WEIGHT = 0.01
SPARSE_ROUTER_RANKING_MARGIN = 0.08

# Texture Restoration Router V2.
# The analytical spectral/spatial experts and 31x31 router context above are unchanged.
# Only the learned deep expert uses the larger patch and the settings below.
TEXTURE_V2_DEEP_PATCH_SIZE = 63
TEXTURE_V2_TRAIN_SAMPLES_PER_EPOCH = 4000
TEXTURE_V2_VAL_SAMPLES = 800
TEXTURE_V2_EPOCHS = 70
TEXTURE_V2_BATCH_SIZE = 2
TEXTURE_V2_TEST_BATCH_SIZE = 4
TEXTURE_V2_NUM_WORKERS = 2
TEXTURE_V2_LEARNING_RATE = 1.5e-4
TEXTURE_V2_WEIGHT_DECAY = 1.0e-4
TEXTURE_V2_GRAD_CLIP = 1.0
TEXTURE_V2_EMA_DECAY = 0.999
TEXTURE_V2_EARLY_STOPPING_PATIENCE = 12
TEXTURE_V2_MIN_EPOCHS = 24
TEXTURE_V2_RESUME = True
TEXTURE_V2_USE_TTA = True
TEXTURE_V2_MAX_MASKS = 1
TEXTURE_V2_TEST_GPU = 0

TEXTURE_V2_BASE_CHANNELS = 72
TEXTURE_V2_MID_CHANNELS = 112
TEXTURE_V2_BOTTLENECK_CHANNELS = 192
TEXTURE_V2_BASIS_CHANNELS = 64
TEXTURE_V2_SPATIAL_ATTENTION_HEADS = 4
TEXTURE_V2_SPATIAL_ATTENTION_LAYERS = 2

# Deep-expert objectives. Main accuracy losses remain dominant; texture terms are modest.
TEXTURE_V2_HARD_EXAMPLE_BOOST = 2.0
TEXTURE_V2_HARD_EXAMPLE_SCALE = 0.45
TEXTURE_V2_DEEP_CENTER_WEIGHT = 0.30
TEXTURE_V2_PATCH_MASKED_WEIGHT = 0.30
TEXTURE_V2_PATCH_SPECTRAL_DERIVATIVE_WEIGHT = 0.15
TEXTURE_V2_PATCH_SPECTRAL_SECOND_DERIVATIVE_WEIGHT = 0.06
TEXTURE_V2_PATCH_SPATIAL_GRADIENT_WEIGHT = 0.12
TEXTURE_V2_PATCH_SSIM_WEIGHT = 0.08
TEXTURE_V2_PATCH_HAAR_WEIGHT = 0.08
TEXTURE_V2_DEEP_SAM_WEIGHT = 0.05

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

# -----------------------------------------------------------------------------
# Hybrid Sparse Router v2: Spectral + Spatial + Random Forest + Deep expert
# -----------------------------------------------------------------------------
SPARSE_ROUTER_RF_TRAIN_SAMPLES_PER_EPOCH = 6000
SPARSE_ROUTER_RF_VAL_SAMPLES = 2000
SPARSE_ROUTER_RF_EPOCHS = 30
SPARSE_ROUTER_RF_BATCH_SIZE = 8
SPARSE_ROUTER_RF_TEST_BATCH_SIZE = 16
SPARSE_ROUTER_RF_NUM_WORKERS = 2
SPARSE_ROUTER_RF_EARLY_STOPPING_PATIENCE = 8
SPARSE_ROUTER_RF_MIN_EPOCHS = 10
SPARSE_ROUTER_RF_GRAD_CLIP = 1.0
SPARSE_ROUTER_RF_EMA_DECAY = 0.999
SPARSE_ROUTER_RF_USE_TTA = True
SPARSE_ROUTER_RF_RESUME = True

# The v2 model is warm-started from the already-trained v1 Sparse Router.
# The backbone therefore uses a smaller learning rate than the routing heads.
SPARSE_ROUTER_RF_BACKBONE_LR = 3e-5
SPARSE_ROUTER_RF_HEAD_LR = 1.2e-4
SPARSE_ROUTER_RF_WEIGHT_DECAY = 1e-4

# Random Forest prediction is CPU-side. Limit threads per DDP process to avoid
# two GPU processes each trying to use every CPU core.
SPARSE_ROUTER_RF_PREDICT_N_JOBS = 4

SPARSE_ROUTER_RF_CLASSIFICATION_LOSS_WEIGHT = 0.55
SPARSE_ROUTER_RF_UNCERTAINTY_LOSS_WEIGHT = 0.30
SPARSE_ROUTER_RF_RANKING_LOSS_WEIGHT = 0.12
SPARSE_ROUTER_RF_REGRET_LOSS_WEIGHT = 0.50
SPARSE_ROUTER_RF_ENTROPY_LOSS_WEIGHT = 0.01

# Important routing mistakes receive more weight than near-ties.
# The multiplier ranges from 1 to 1 + this value.
SPARSE_ROUTER_RF_ORACLE_MARGIN_WEIGHT = 2.0

# RF confidence is estimated from the fraction of observed values in the
# 7x7x15 RF context. It is only metadata; the learned uncertainty estimator
# remains the main routing signal.
SPARSE_ROUTER_RF_CONFIDENCE_FLOOR = 0.05
SPARSE_ROUTER_RF_CONFIDENCE_PRIOR_WEIGHT = 0.10


# Four-expert Texture Restoration Router RF V2.
# Keeps spectral + spatial + Random Forest experts and upgrades only the deep expert.
TEXTURE_RF_V2_DEEP_PATCH_SIZE = 63
TEXTURE_RF_V2_TRAIN_SAMPLES_PER_EPOCH = 4500
TEXTURE_RF_V2_VAL_SAMPLES = 1000
TEXTURE_RF_V2_EPOCHS = 60
TEXTURE_RF_V2_BATCH_SIZE = 2
TEXTURE_RF_V2_TEST_BATCH_SIZE = 2
TEXTURE_RF_V2_NUM_WORKERS = 2
TEXTURE_RF_V2_ROUTER_LR = 1.0e-4
TEXTURE_RF_V2_DEEP_LR = 1.2e-4
TEXTURE_RF_V2_WEIGHT_DECAY = 1.0e-4
TEXTURE_RF_V2_GRAD_CLIP = 1.0
TEXTURE_RF_V2_EMA_DECAY = 0.999
TEXTURE_RF_V2_EARLY_STOPPING_PATIENCE = 10
TEXTURE_RF_V2_MIN_EPOCHS = 20
TEXTURE_RF_V2_START_FROM_SCRATCH = True
TEXTURE_RF_V2_RESUME = False
TEXTURE_RF_V2_USE_TTA = True
TEXTURE_RF_V2_TEST_GPU = 0
# Set to None after the first successful one-mask smoke benchmark.
TEXTURE_RF_V2_MAX_MASKS = None

TEXTURE_RF_V2_BASE_CHANNELS = 72
TEXTURE_RF_V2_MID_CHANNELS = 112
TEXTURE_RF_V2_BOTTLENECK_CHANNELS = 192
TEXTURE_RF_V2_BASIS_CHANNELS = 64
TEXTURE_RF_V2_SPATIAL_ATTENTION_HEADS = 4
TEXTURE_RF_V2_SPATIAL_ATTENTION_LAYERS = 2

# Strengthened Deep V3 spectral-sequence branch.
TEXTURE_RF_V2_SPECTRAL_MODEL_DIM = 192
TEXTURE_RF_V2_SPECTRAL_HEADS = 4
TEXTURE_RF_V2_SPECTRAL_LAYERS = 4
TEXTURE_RF_V2_SPECTRAL_FF_MULT = 4
TEXTURE_RF_V2_GAP_HIDDEN = 64

# Deep expert is weighted most where spectral, spatial, and RF are all weak.
TEXTURE_RF_V2_HARD_EXAMPLE_BOOST = 2.5
TEXTURE_RF_V2_HARD_EXAMPLE_SCALE = 0.45
TEXTURE_RF_V2_DEEP_CENTER_WEIGHT = 0.40
TEXTURE_RF_V2_PATCH_MASKED_WEIGHT = 0.30
TEXTURE_RF_V2_PATCH_SPECTRAL_DERIVATIVE_WEIGHT = 0.15
TEXTURE_RF_V2_PATCH_SPECTRAL_SECOND_DERIVATIVE_WEIGHT = 0.06
TEXTURE_RF_V2_PATCH_SPATIAL_GRADIENT_WEIGHT = 0.12
TEXTURE_RF_V2_PATCH_SSIM_WEIGHT = 0.08
TEXTURE_RF_V2_PATCH_HAAR_WEIGHT = 0.08
TEXTURE_RF_V2_DEEP_SAM_WEIGHT = 0.05

# Extra supervision for the three previously weak hard-case families.
TEXTURE_RF_V2_WHOLE_SPECTRUM_LOSS_BOOST = 2.2
TEXTURE_RF_V2_CONSECUTIVE_BANDS_LOSS_BOOST = 1.7
TEXTURE_RF_V2_SPATIO_SPECTRAL_LOSS_BOOST = 1.9
TEXTURE_RF_V2_WHOLE_SAM_BOOST = 1.8
TEXTURE_RF_V2_CONSECUTIVE_CONTINUITY_BOOST = 2.0
TEXTURE_RF_V2_SPATIO_CONTINUITY_BOOST = 1.8
TEXTURE_RF_V2_DEEP_CENTER_D1_WEIGHT = 0.12
TEXTURE_RF_V2_DEEP_CENTER_D2_WEIGHT = 0.05

# Post-router texture-consistency refiner. The four-way router remains the
# initial estimator; this small module applies only a bounded residual on
# missing center bands.
TEXTURE_RF_V2_REFINER_HIDDEN = 96
TEXTURE_RF_V2_REFINER_LR = 2.5e-4
TEXTURE_RF_V2_REFINER_GATE_FLOOR = 0.20
TEXTURE_RF_V2_REFINER_GATE_ERROR_SCALE = 0.35
TEXTURE_RF_V2_REFINER_GATE_TEXTURE_SCALE = 0.30
TEXTURE_RF_V2_REFINER_MAX_DELTA_NORMALIZED = 0.75
TEXTURE_RF_V2_ROUTER_BASE_LOSS_WEIGHT = 0.20
TEXTURE_RF_V2_REFINER_NO_WORSE_WEIGHT = 0.08
TEXTURE_RF_V2_REFINER_DELTA_WEIGHT = 0.001
TEXTURE_RF_V2_REFINER_RESIDUAL_SUPERVISION_WEIGHT = 0.80
TEXTURE_RF_V2_REFINER_GATE_SUPERVISION_WEIGHT = 0.12
TEXTURE_RF_V2_FINAL_TEXTURE_GRADIENT_WEIGHT = 0.24
TEXTURE_RF_V2_FINAL_TEXTURE_LAPLACIAN_WEIGHT = 0.16
TEXTURE_RF_V2_FINAL_SPECTRAL_D2_WEIGHT = 0.06

# Test-time texture diagnostics and representative output figures.
TEXTURE_RF_V2_COMPUTE_TEXTURE_METRICS = True
TEXTURE_RF_V2_TEXTURE_SSIM_MAX_SAMPLES = 20000
TEXTURE_RF_V2_EXPORT_EXAMPLES = True
TEXTURE_RF_V2_EXPORT_CROP_HEIGHT = 96
TEXTURE_RF_V2_EXPORT_CROP_WIDTH = 96
TEXTURE_RF_V2_EXPORT_MAX_SPATIAL_PIXELS = 5000
TEXTURE_RF_V2_EXPORT_CASES = [
    ("whole_spectrum_pixels", 5, 1),
    ("consecutive_spectral_bands", 5, 1),
    ("spatio_spectral_block", 5, 1),
    ("mixed_saturation", 5, 1),
]

# -----------------------------------------------------------------------------
# V8 dedicated trainable region-level HSI inpainter.
# This is a separate component of the final method, trained from scratch on
# synthetic large connected missing regions. It does NOT average V5 predictions.
# -----------------------------------------------------------------------------
REGION_V8_ENABLED = True
REGION_V8_PATCH_SIZE = 127
REGION_V8_BASE_CHANNELS = 48
REGION_V8_MID_CHANNELS = 72
REGION_V8_BOTTLENECK_CHANNELS = 112
REGION_V8_BASIS_CHANNELS = 48
REGION_V8_SPATIAL_HEADS = 4
REGION_V8_SPATIAL_LAYERS = 2
REGION_V8_SPECTRAL_DIM = 128
REGION_V8_SPECTRAL_HEADS = 4
REGION_V8_SPECTRAL_LAYERS = 3

REGION_V8_TRAIN_SAMPLES_PER_EPOCH = 3000
REGION_V8_VAL_SAMPLES = 400
REGION_V8_EPOCHS = 50
REGION_V8_BATCH_SIZE = 1
REGION_V8_NUM_WORKERS = 2
REGION_V8_LEARNING_RATE = 2.0e-4
REGION_V8_WEIGHT_DECAY = 1.0e-4
REGION_V8_GRAD_CLIP = 1.0
REGION_V8_EMA_DECAY = 0.999
REGION_V8_MIN_EPOCHS = 18
REGION_V8_EARLY_STOPPING_PATIENCE = 8
REGION_V8_START_FROM_SCRATCH = True

# Region-inpainting objectives. Pixel fidelity remains dominant while spatial
# structure, high-frequency texture and spectral continuity are explicit.
REGION_V8_SPATIAL_GRAD_WEIGHT = 0.28
REGION_V8_SPECTRAL_D1_WEIGHT = 0.18
REGION_V8_SPECTRAL_D2_WEIGHT = 0.08
REGION_V8_SSIM_WEIGHT = 0.16
REGION_V8_HAAR_WEIGHT = 0.16
REGION_V8_SAM_WEIGHT = 0.08

# Geometry-only activation at inference. No synthetic scenario name is used.
# 0.60 lets large spatio-spectral blocks use the region expert as well as
# whole-spectrum regions; small/easy corruption remains with V5.
REGION_V8_MIN_MISSING_BAND_FRACTION = 0.60
REGION_V8_MIN_COMPONENT_PIXELS = 6
REGION_V8_BLEND = 1.0
REGION_V8_INFERENCE_BATCH_SIZE = 1
REGION_V8_TILE_STRIDE = 48
REGION_V8_USE_TTA = False

# -----------------------------------------------------------------------------
# V9 retrieval-guided trainable region HSI inpainter.
# Uses a healthy reference patch retrieved from the same acquisition/cube using
# observed context only. The base V5 router/refiner remains unchanged.
# -----------------------------------------------------------------------------
REGION_V9_ENABLED = True
REGION_V9_PATCH_SIZE = 127
REGION_V9_BASE_CHANNELS = 48
REGION_V9_MID_CHANNELS = 72
REGION_V9_BOTTLENECK_CHANNELS = 112
REGION_V9_BASIS_CHANNELS = 48
REGION_V9_SPATIAL_HEADS = 4
REGION_V9_SPATIAL_LAYERS = 2
REGION_V9_SPECTRAL_DIM = 128
REGION_V9_SPECTRAL_HEADS = 4
REGION_V9_SPECTRAL_LAYERS = 3
REGION_V9_REFERENCE_HEADS = 4
REGION_V9_CROSS_ATTENTION_GRID = 10
REGION_V9_REFERENCE_CONTEXT_STRENGTH = 0.65
REGION_V9_REFERENCE_FILM_STRENGTH = 0.15

REGION_V9_TRAIN_SAMPLES_PER_EPOCH = 3000
REGION_V9_VAL_SAMPLES = 400
REGION_V9_EPOCHS = 50
REGION_V9_BATCH_SIZE = 1
REGION_V9_NUM_WORKERS = 2
REGION_V9_LEARNING_RATE = 2.0e-4
REGION_V9_WEIGHT_DECAY = 1.0e-4
REGION_V9_GRAD_CLIP = 1.0
REGION_V9_EMA_DECAY = 0.999
REGION_V9_MIN_EPOCHS = 18
REGION_V9_EARLY_STOPPING_PATIENCE = 8
REGION_V9_START_FROM_SCRATCH = True

# Same reconstruction objectives used by V8, now with real healthy-reference
# guidance rather than asking the network to invent all fine texture itself.
REGION_V9_SPATIAL_GRAD_WEIGHT = 0.28
REGION_V9_SPECTRAL_D1_WEIGHT = 0.18
REGION_V9_SPECTRAL_D2_WEIGHT = 0.08
REGION_V9_SSIM_WEIGHT = 0.16
REGION_V9_HAAR_WEIGHT = 0.16
REGION_V9_SAM_WEIGHT = 0.08

# Retrieval bank and descriptor settings.
REGION_V9_DESCRIPTOR_BANDS = 16
REGION_V9_DESCRIPTOR_SPATIAL_BINS = 4
REGION_V9_BANK_MAX_PATCHES_PER_CUBE = 160
REGION_V9_INFERENCE_BANK_MAX_PATCHES = 96
REGION_V9_RETRIEVAL_TOP_K = 4
REGION_V9_RETRIEVAL_MIN_CENTER_DISTANCE = 102
REGION_V9_RETRIEVAL_DISTANCE_SCALE = 0.35
REGION_V9_REFERENCE_MIN_VALID_FRACTION = 0.995
REGION_V9_REFERENCE_MAX_MISSING_FRACTION = 0.002
REGION_V9_REFERENCE_DROPOUT = 0.08

# Geometry-only activation. No benchmark scenario label is inspected.
REGION_V9_MIN_MISSING_BAND_FRACTION = 0.60
REGION_V9_MIN_COMPONENT_PIXELS = 6
REGION_V9_BLEND = 1.0
REGION_V9_INFERENCE_BATCH_SIZE = 1
REGION_V9_TILE_STRIDE = 48
REGION_V9_USE_TTA = False

# -----------------------------------------------------------------------------
# V10 structure-guided + retrieval-guided trainable region HSI inpainter.
# V9 retrieval is retained, but the model first completes wavelength-dependent
# spatial structure maps and injects them into the reconstruction decoder.
# -----------------------------------------------------------------------------
REGION_V10_ENABLED = True
REGION_V10_PATCH_SIZE = 127
REGION_V10_BASE_CHANNELS = 48
REGION_V10_MID_CHANNELS = 72
REGION_V10_BOTTLENECK_CHANNELS = 112
REGION_V10_BASIS_CHANNELS = 48
REGION_V10_SPATIAL_HEADS = 4
REGION_V10_SPATIAL_LAYERS = 2
REGION_V10_SPECTRAL_DIM = 128
REGION_V10_SPECTRAL_HEADS = 4
REGION_V10_SPECTRAL_LAYERS = 3
REGION_V10_REFERENCE_HEADS = 4
REGION_V10_CROSS_ATTENTION_GRID = 10
REGION_V10_REFERENCE_CONTEXT_STRENGTH = 0.65
REGION_V10_REFERENCE_FILM_STRENGTH = 0.15

# Explicit structure-continuation branch. Eight wavelength groups retain
# wavelength-dependent texture instead of collapsing HSI to one grayscale map.
REGION_V10_STRUCTURE_GROUPS = 8
REGION_V10_STRUCTURE_BASE_CHANNELS = 32
REGION_V10_STRUCTURE_MID_CHANNELS = 48
REGION_V10_STRUCTURE_BOTTLENECK_CHANNELS = 64
REGION_V10_STRUCTURE_HEADS = 4
REGION_V10_STRUCTURE_LAYERS = 2
REGION_V10_STRUCTURE_CONTEXT_GRID = 8
REGION_V10_STRUCTURE_FEATURE_STRENGTH = 0.45
REGION_V10_STRUCTURE_OUTPUT_STRENGTH = 0.35

REGION_V10_TRAIN_SAMPLES_PER_EPOCH = 3000
REGION_V10_VAL_SAMPLES = 400
REGION_V10_EPOCHS = 55
REGION_V10_BATCH_SIZE = 1
REGION_V10_NUM_WORKERS = 2
REGION_V10_LEARNING_RATE = 1.8e-4
REGION_V10_WEIGHT_DECAY = 1.0e-4
REGION_V10_GRAD_CLIP = 1.0
REGION_V10_EMA_DECAY = 0.999
REGION_V10_MIN_EPOCHS = 20
REGION_V10_EARLY_STOPPING_PATIENCE = 9
REGION_V10_START_FROM_SCRATCH = True

# Main reconstruction objectives.
REGION_V10_SPATIAL_GRAD_WEIGHT = 0.30
REGION_V10_SPECTRAL_D1_WEIGHT = 0.18
REGION_V10_SPECTRAL_D2_WEIGHT = 0.08
REGION_V10_SSIM_WEIGHT = 0.16
REGION_V10_HAAR_WEIGHT = 0.18
REGION_V10_SAM_WEIGHT = 0.08

# Explicit structure-completion supervision. The gradient/laplacian/boundary
# terms force line/edge continuation through the missing region rather than
# merely improving average intensity.
REGION_V10_STRUCTURE_PIXEL_WEIGHT = 0.22
REGION_V10_STRUCTURE_GRAD_WEIGHT = 0.34
REGION_V10_STRUCTURE_LAPLACIAN_WEIGHT = 0.18
REGION_V10_STRUCTURE_BOUNDARY_WEIGHT = 0.20
REGION_V10_STRUCTURE_OBSERVED_CONTEXT_WEIGHT = 0.04

# Retrieval bank and descriptor settings (same leak-free logic as V9).
REGION_V10_DESCRIPTOR_BANDS = 16
REGION_V10_DESCRIPTOR_SPATIAL_BINS = 4
REGION_V10_BANK_MAX_PATCHES_PER_CUBE = 160
REGION_V10_INFERENCE_BANK_MAX_PATCHES = 96
REGION_V10_RETRIEVAL_TOP_K = 4
REGION_V10_RETRIEVAL_MIN_CENTER_DISTANCE = 102
REGION_V10_RETRIEVAL_DISTANCE_SCALE = 0.35
REGION_V10_REFERENCE_MIN_VALID_FRACTION = 0.995
REGION_V10_REFERENCE_MAX_MISSING_FRACTION = 0.002
REGION_V10_REFERENCE_DROPOUT = 0.08

# Geometry-only activation; scenario names are never inspected.
REGION_V10_MIN_MISSING_BAND_FRACTION = 0.60
REGION_V10_MIN_COMPONENT_PIXELS = 6
REGION_V10_BLEND = 1.0
REGION_V10_INFERENCE_BATCH_SIZE = 1
REGION_V10_TILE_STRIDE = 48
REGION_V10_USE_TTA = False


# -----------------------------------------------------------------------------
# V12 texture-preserving extensions for the V10 region expert.
# IMPORTANT: activation geometry is intentionally unchanged. V12 still uses
# REGION_V10_MIN_MISSING_BAND_FRACTION = 0.60, so the new expert is targeted at
# the same very-hard/whole-spectrum regions that V10 handled.
# -----------------------------------------------------------------------------
REGION_V12_NUM_REFERENCES = 3
REGION_V12_RETRIEVAL_POOL_K = 8
REGION_V12_DESCRIPTOR_BANDS = 20
REGION_V12_DESCRIPTOR_SPATIAL_BINS = 6

# Multi-scale dense reference transfer. Reference features are aligned and
# injected at bottleneck, mid-resolution and full-resolution decoder scales.
REGION_V12_REFERENCE_BOTTLENECK_STRENGTH = 0.30
REGION_V12_REFERENCE_MID_STRENGTH = 0.36
REGION_V12_REFERENCE_HIGH_STRENGTH = 0.48
REGION_V12_REFERENCE_MAX_OFFSET_BOTTLENECK = 3.0
REGION_V12_REFERENCE_MAX_OFFSET_MID = 5.0
REGION_V12_REFERENCE_MAX_OFFSET_HIGH = 7.0
REGION_V12_REFERENCE_DROPOUT = 0.08
REGION_V12_REFERENCE_GLOBAL_DROPOUT = 0.04

# Texture-distribution supervision. These losses complement the deterministic
# pixel/gradient/SSIM/Haar objectives and specifically penalize over-smoothing.
REGION_V12_TEXTURE_AUTOCORR_WEIGHT = 0.10
REGION_V12_TEXTURE_FOURIER_WEIGHT = 0.055
REGION_V12_TEXTURE_GROUPS = 8
REGION_V12_TEXTURE_LAGS = (1, 2, 4)
REGION_V12_FOURIER_EPS = 1.0e-4

# -----------------------------------------------------------------------------
# V13 context-aligned raw patch transfer for the V12/V10 region expert.
# Activation geometry remains unchanged on purpose:
# REGION_V10_MIN_MISSING_BAND_FRACTION = 0.60
# The new path only changes how texture is reconstructed inside the same hard
# region cases; the already-good easy cases remain on the base router.
# -----------------------------------------------------------------------------
REGION_V13_NUM_REFERENCES = 3
REGION_V13_RETRIEVAL_POOL_K = 12
REGION_V13_DESCRIPTOR_BANDS = 24
REGION_V13_DESCRIPTOR_SPATIAL_BINS = 6
REGION_V13_BANK_MAX_PATCHES_PER_CUBE = 240
REGION_V13_INFERENCE_BANK_MAX_PATCHES = 192
# Training references are kept fully disjoint from the target patch to prevent
# synthetic-mask ground-truth leakage. At inference a smaller distance is safe
# because candidate patches are rejected if they overlap the real benchmark mask.
REGION_V13_TRAIN_MIN_CENTER_DISTANCE = 136
REGION_V13_INFERENCE_MIN_CENTER_DISTANCE = 64

# Raw HSI texture transfer. A dense flow is predicted from low-dimensional
# wavelength-group structure, then the *actual reference HSI patch* is warped,
# radiometrically calibrated from observed context, and blended into the hole.
REGION_V13_TRANSFER_GROUPS = 8
REGION_V13_TRANSFER_HIDDEN = 48
REGION_V13_TRANSFER_MAX_OFFSET_PX = 10.0
REGION_V13_TRANSFER_CONTEXT_SCALE = 1.20
REGION_V13_TRANSFER_MIN_ALPHA = 0.32
REGION_V13_TRANSFER_MAX_ALPHA = 0.78
REGION_V13_TRANSFER_FEATURE_STRENGTH = 0.28
REGION_V13_TRANSFER_RESIDUAL_SCALE = 0.12
REGION_V13_TRANSFER_SCORE_FLOOR = 1.0e-4
REGION_V13_TRANSFER_STD_SCALE_MIN = 0.55
REGION_V13_TRANSFER_STD_SCALE_MAX = 1.80

# Texture supervision added on top of the V12 losses. This term directly
# supervises fine spatial residuals in the missing region rather than only their
# global distribution.
REGION_V13_HIGHPASS_WEIGHT = 0.22
REGION_V13_HIGHPASS_KERNEL = 5

# -----------------------------------------------------------------------------
# V15 local directional PatchMatch texture transfer
# -----------------------------------------------------------------------------
# V15 is applied only to spatial pixels where most of the spectrum is missing.
# This keeps easy/partial-band cases on the established V13 reconstruction path.
REGION_V15_LOCAL_MIN_BAND_MISSING_FRACTION = 0.80
REGION_V15_SOURCE_MAX_BAND_MISSING_FRACTION = 0.02

# Observed annulus used to estimate local structure and score source offsets.
REGION_V15_RING_INNER = 3
REGION_V15_RING_OUTER = 14
REGION_V15_MIN_RING_PIXELS = 24
REGION_V15_MIN_CALIBRATION_PIXELS = 20

# Search mostly along the dominant local structure tangent, with a small
# perpendicular allowance and cardinal-direction safety candidates.
REGION_V15_SEARCH_RADIUS = 48
REGION_V15_SEARCH_STEP = 2
REGION_V15_PERP_MAX = 8
REGION_V15_PERP_STEP = 2
# Coarser search during training to keep epochs practical.
REGION_V15_SEARCH_STEP_TRAIN = 4
REGION_V15_PERP_STEP_TRAIN = 4
REGION_V15_MIN_SOURCE_COVERAGE = 0.995

# Candidate score: observed boundary intensity + gradient continuity + weak
# geometric regularization toward the detected structure direction.
REGION_V15_GRAD_SCORE_WEIGHT = 0.45
REGION_V15_PERP_SCORE_WEIGHT = 0.18
REGION_V15_DISTANCE_SCORE_WEIGHT = 0.04
REGION_V15_SCORE_CONFIDENCE_SCALE = 1.60
REGION_V15_MIN_CONFIDENCE = 0.10

# Keep neural/V13 low-frequency radiometry, but carry the actual high-frequency
# spatial texture from the selected local healthy source.
REGION_V15_LOWPASS_KERNEL = 7
REGION_V15_LOCAL_LOW_BLEND = 0.30
REGION_V15_LOCAL_HF_GAIN = 1.00
REGION_V15_LOCAL_ALPHA_MIN = 0.72
REGION_V15_LOCAL_ALPHA_MAX = 0.98

# V15 checkpoints are saved separately. Training can initialize from the V13
# model so the already learned spectral/spatial backbone is not discarded.
REGION_V15_START_FROM_V13 = True
REGION_V15_MODEL_FOLDER = "region_inpainter_v15_2_hf_phase_aligned_patchmatch"

# V15.1 robust local-source search
REGION_V15_ADAPTIVE_MARGIN = 4
REGION_V15_FALLBACK_MIN_SOURCE_COVERAGE = 0.85
REGION_V15_FALLBACK_MIN_RING_PIXELS = 10
REGION_V15_MIN_ORIENTATION_PIXELS = 12
REGION_V15_COVERAGE_SCORE_WEIGHT = 3.0
REGION_V15_VALID_SOURCE_CONFIDENCE_FLOOR = 0.12
REGION_V15_NO_SOURCE_SCORE = 999.0


# -----------------------------------------------------------------------------
# V15.2: phase-aligned, high-frequency-only local texture transfer
# -----------------------------------------------------------------------------
# Fine integer-offset refinement around the V15.1 coarse source location.
REGION_V152_PHASE_REFINE_RADIUS = 3
REGION_V152_PHASE_RING_OUTER = 6
REGION_V152_PHASE_MIN_RING_PIXELS = 12
REGION_V152_PHASE_INTENSITY_WEIGHT = 0.55
REGION_V152_PHASE_GRAD_WEIGHT = 0.45
REGION_V152_PHASE_SHIFT_PENALTY = 0.02
REGION_V152_PHASE_CONFIDENCE_SCALE = 0.80

# Preserve V13 low-frequency content exactly; import only local high-frequency
# residuals from the selected healthy source.
REGION_V152_LOWPASS_KERNEL = 7
REGION_V152_HF_GAIN = 1.00
REGION_V152_HF_SCALE_MIN = 0.65
REGION_V152_HF_SCALE_MAX = 1.35
REGION_V152_HF_SCALE_MIN_PIXELS = 12

# More conservative fusion than V15.1.
REGION_V152_ALPHA_MIN = 0.20
REGION_V152_ALPHA_MAX = 0.60
REGION_V152_VALID_SOURCE_CONFIDENCE_FLOOR = 0.08
REGION_V152_SCORE_CONF_WEIGHT = 0.30
REGION_V152_PHASE_CONF_WEIGHT = 0.55
REGION_V152_COVERAGE_CONF_WEIGHT = 0.15

# -----------------------------------------------------------------------------
# V15.3: frozen V13 + trainable structure continuation + V15.2 HF transfer
# -----------------------------------------------------------------------------
REGION_V153_ENABLED = True
REGION_V153_START_FROM_V13 = True
REGION_V153_FREEZE_V13 = True
# Keep the legacy key because test.py/train.py already use it as the region
# checkpoint folder selector.
REGION_V15_MODEL_FOLDER = "region_inpainter_v15_6_endpoint_traced_crack_hf"

# Structure-continuation branch architecture.
REGION_V153_STRUCTURE_HIDDEN = 48
REGION_V153_STRUCTURE_BASIS = 12
REGION_V153_STRUCTURE_HEADS = 4
REGION_V153_STRUCTURE_LAYERS = 2
REGION_V153_STRUCTURE_GRID = 8
REGION_V153_STRUCTURE_MIN_BAND_MISSING_FRACTION = 0.80
REGION_V153_STRUCTURE_MAX_DELTA = 0.65
REGION_V153_STRUCTURE_GATE_MAX = 0.75
# When a strong structure correction is being made, slightly suppress local HF
# injection so texture cannot erase a newly continued thin crack/ridge.
REGION_V153_LOCAL_STRUCTURE_SUPPRESSION = 0.35

# Fine-tuning schedule. The original V13 backbone is frozen; only the new V15.3
# branch is optimized.
REGION_V153_TRAIN_SAMPLES_PER_EPOCH = 3000
REGION_V153_VAL_SAMPLES = 500
REGION_V153_EPOCHS = 35
REGION_V153_MIN_EPOCHS = 12
REGION_V153_EARLY_STOPPING_PATIENCE = 8
REGION_V153_LEARNING_RATE = 2.0e-4
REGION_V153_WEIGHT_DECAY = 1.0e-4
REGION_V153_GRAD_CLIP = 1.0
REGION_V153_EMA_DECAY = 0.999
REGION_V153_START_FROM_SCRATCH = True

# V15.3-specific structural objectives. These are applied to the structure-
# refined prediction, not to the frozen V13 structure branch.
REGION_V153_STRUCTURE_COMPLETION_WEIGHT = 0.70
REGION_V153_ORIENTATION_WEIGHT = 0.30
REGION_V153_RESIDUAL_REG_WEIGHT = 0.008
REGION_V153_GATE_REG_WEIGHT = 0.002


# -----------------------------------------------------------------------------
# V15.5: frozen V13 + multi-directional thin-line continuation + V15.2 HF
# -----------------------------------------------------------------------------
REGION_V155_ENABLED = True
REGION_V155_START_FROM_V13 = True
REGION_V155_FREEZE_V13 = True
REGION_V155_START_FROM_SCRATCH = True

# Twelve local line directions: 0..165 degrees in 15-degree increments.
REGION_V155_NUM_DIRECTIONS = 12
REGION_V155_RAY_LENGTH = 22
REGION_V155_RAY_DECAY = 0.045
REGION_V155_MIN_RAY_SUPPORT = 0.12
REGION_V155_DETAIL_KERNEL = 9
REGION_V155_SELECTOR_HIDDEN = 48
REGION_V155_SELECTOR_NULL_BIAS = -0.10
REGION_V155_SELECTOR_DIRECTION_BIAS = 0.0

# Direct thin-structure continuation is intentionally stronger than the generic
# residual, so narrow cracks/ridges are not averaged away.
REGION_V155_THIN_GATE_MAX = 0.88
REGION_V155_THIN_MAX_DELTA = 0.80
REGION_V155_SPECTRAL_MODULATION = 0.25
REGION_V155_SMOOTH_HIDDEN = 40
REGION_V155_SMOOTH_BASIS = 10
REGION_V155_SMOOTH_MAX_DELTA = 0.28
REGION_V155_SMOOTH_GATE_MAX = 0.35
REGION_V155_STRUCTURE_MIN_BAND_MISSING_FRACTION = 0.80

# Protect newly continued thin lines from later HF transplantation.
REGION_V155_LOCAL_THIN_SUPPRESSION = 0.60

# Fine-tuning: original V13 stays frozen; only structure_continuation_v155 trains.
REGION_V155_TRAIN_SAMPLES_PER_EPOCH = 3000
REGION_V155_VAL_SAMPLES = 500
REGION_V155_EPOCHS = 42
REGION_V155_MIN_EPOCHS = 16
REGION_V155_EARLY_STOPPING_PATIENCE = 10
REGION_V155_LEARNING_RATE = 1.0e-4
REGION_V155_WEIGHT_DECAY = 1.0e-4
REGION_V155_GRAD_CLIP = 1.0
REGION_V155_EMA_DECAY = 0.999

# Structure/topology-specific objectives.
REGION_V155_STRUCTURE_COMPLETION_WEIGHT = 0.45
REGION_V155_ORIENTATION_WEIGHT = 0.25
REGION_V155_THIN_LINE_WEIGHT = 0.55
REGION_V155_GUIDANCE_WEIGHT = 0.22
REGION_V155_DIRECTION_ENTROPY_WEIGHT = 0.0005
REGION_V155_RESIDUAL_REG_WEIGHT = 0.001
REGION_V155_GATE_REG_WEIGHT = 0.0

# V15.5 anti-collapse supervision.
# The gate is trained from ground-truth thin-structure strength and the selector
# is trained to choose the directional bridge that best matches that structure.
REGION_V155_THIN_GATE_INIT_BIAS = 0.0
REGION_V155_SELECTOR_ACTIVE_FLOOR = 0.18
REGION_V155_DIRECTION_CONFIDENCE_LOG_PRIOR = 0.30
REGION_V155_GATE_SUPERVISION_WEIGHT = 0.70
REGION_V155_DIRECTION_SUPERVISION_WEIGHT = 0.55
REGION_V155_NULL_SUPERVISION_WEIGHT = 0.20
REGION_V155_GATE_TARGET_STRENGTH_RATIO = 1.05
REGION_V155_GATE_TARGET_TEMPERATURE = 0.22
REGION_V155_GATE_TARGET_ABS_FRACTION = 0.10
REGION_V155_GATE_TARGET_ABS_TEMPERATURE = 0.04
REGION_V155_GATE_POSITIVE_BOOST = 2.0
REGION_V155_DIRECTION_TARGET_TEMPERATURE = 0.18
REGION_V155_DIRECTION_CONFIDENCE_WEIGHT = 0.55
REGION_V155_DIRECTION_MIN_TARGET_GATE = 0.20
REGION_V155_ENDPOINT_WEIGHT = 0.18
REGION_V155_ENDPOINT_RING = 2


# =============================================================================
# V15.6: explicit endpoint-to-endpoint crack/ridge tracing + structure-aware HF
# =============================================================================
REGION_V156_ENABLED = True
REGION_V156_START_FROM_V155 = True
REGION_V156_FREEZE_V13 = True
REGION_V156_START_FROM_SCRATCH = True
REGION_V156_SOURCE_V155_FOLDER = "region_inpainter_v15_5_frozen_v13_supervised_thinline_hf"

# Observed-boundary endpoint extraction and path tracing.
REGION_V156_ENDPOINT_RING = 3
REGION_V156_ENDPOINT_HIDDEN = 48
REGION_V156_ENDPOINT_DETAIL_RATIO = 1.00
REGION_V156_ENDPOINT_DETAIL_TEMPERATURE = 0.30
REGION_V156_ENDPOINT_GRAD_RATIO = 0.85
REGION_V156_ENDPOINT_GRAD_TEMPERATURE = 0.35
REGION_V156_ENDPOINT_CONF_GAIN = 2.20
REGION_V156_ENDPOINT_AGREEMENT_SCALE = 1.15
REGION_V156_ENDPOINT_SIGN_TEMPERATURE = 0.35
REGION_V156_PATH_CONFIDENCE_LOG_PRIOR = 0.65
REGION_V156_SELECTOR_ACTIVE_FLOOR = 0.12
REGION_V156_ENDPOINT_EVIDENCE_FLOOR = 0.08
REGION_V156_ENDPOINT_GATE_INIT_BIAS = -0.25
REGION_V156_ENDPOINT_GATE_MAX = 0.92
REGION_V156_ENDPOINT_MAX_DELTA = 0.85
REGION_V156_SPECTRAL_MODULATION = 0.20

# Structure-aware protection of the successful V15.2 local HF transfer.
REGION_V156_LOCAL_COMPATIBILITY_SCALE = 0.80
REGION_V156_LOCAL_COMPATIBILITY_FLOOR = 0.12
REGION_V156_LOCAL_PATH_SUPPRESSION = 0.20

# Fine-tuning: initialize inherited continuation weights from trained V15.5,
# keep the original V13 backbone frozen, and train only structure_continuation_v156.
REGION_V156_TRAIN_SAMPLES_PER_EPOCH = 3000
REGION_V156_VAL_SAMPLES = 500
REGION_V156_EPOCHS = 30
REGION_V156_MIN_EPOCHS = 10
REGION_V156_EARLY_STOPPING_PATIENCE = 8
REGION_V156_LEARNING_RATE = 6.0e-5
REGION_V156_RESUME_LEARNING_RATE = 3.5e-5
REGION_V156_WEIGHT_DECAY = 1.0e-4
REGION_V156_GRAD_CLIP = 1.0
REGION_V156_EMA_DECAY = 0.999

# Reuse V15.5 ground-truth gate/direction supervision, but put more emphasis
# on endpoint/path continuity.
REGION_V156_STRUCTURE_COMPLETION_WEIGHT = 0.40
REGION_V156_ORIENTATION_WEIGHT = 0.30
REGION_V156_THIN_LINE_WEIGHT = 0.60
REGION_V156_GUIDANCE_WEIGHT = 0.28
REGION_V156_GATE_SUPERVISION_WEIGHT = 0.65
REGION_V156_DIRECTION_SUPERVISION_WEIGHT = 0.60
REGION_V156_ENDPOINT_WEIGHT = 0.35
REGION_V156_RESIDUAL_REG_WEIGHT = 0.0008
REGION_V156_GATE_REG_WEIGHT = 0.0


# =============================================================================
# V16 FINAL: structure-first topology completion + topology-aware safe HF
# =============================================================================
REGION_V16_ENABLED = True
REGION_V16_MODEL_FOLDER = "region_inpainter_v16_structure_first_topology_hf"
REGION_V16_START_FROM_SCRATCH = True

# Frozen V13 base + dedicated topology branch.
REGION_V16_EDGE_BASE_CHANNELS = 48
REGION_V16_EDGE_MID_CHANNELS = 72
REGION_V16_EDGE_BOTTLENECK_CHANNELS = 112
REGION_V16_EDGE_HEADS = 4
REGION_V16_EDGE_LAYERS = 2
REGION_V16_EDGE_GRID = 8
REGION_V16_HARD_MISSING_BAND_FRACTION = 0.80
REGION_V16_CONTEXT_RING = 4
REGION_V16_STRUCTURE_MAX_DELTA = 0.90
REGION_V16_STRUCTURE_CONFIDENCE_FLOOR = 0.0
REGION_V16_SPECTRAL_MODULATION = 0.20
REGION_V16_TOPOLOGY_INIT_BIAS = -0.75
REGION_V16_CONFIDENCE_INIT_BIAS = -0.35

# Soft topology targets from multi-band structure.
REGION_V16_TOPOLOGY_DETAIL_KERNEL = 9
REGION_V16_EDGE_TARGET_RATIO = 1.05
REGION_V16_EDGE_TARGET_TEMPERATURE = 0.30
REGION_V16_LINE_TARGET_RATIO = 1.15
REGION_V16_LINE_TARGET_TEMPERATURE = 0.28
REGION_V16_TOPOLOGY_POSITIVE_BOOST = 2.0
REGION_V16_CLDICE_ITERATIONS = 7
REGION_V16_STRUCTURE_TOPOLOGY_BOOST = 1.5
REGION_V16_BOUNDARY_RING = 2
REGION_V16_CONFIDENCE_ERROR_SCALE = 0.35

# Topology-aware protection of the successful V15.2 HF transplant.
REGION_V16_HF_ALPHA_CAP = 0.48
REGION_V16_HF_COMPATIBILITY_FLOOR = 0.10
REGION_V16_LOW_CONFIDENCE_HF_FACTOR = 0.25

# Final training schedule. Only topology_edge_v16.* is trainable.
REGION_V16_TRAIN_SAMPLES_PER_EPOCH = 3600
REGION_V16_VAL_SAMPLES = 600
REGION_V16_EPOCHS = 36
REGION_V16_MIN_EPOCHS = 12
REGION_V16_EARLY_STOPPING_PATIENCE = 9
REGION_V16_LEARNING_RATE = 8.0e-5
REGION_V16_RESUME_LEARNING_RATE = 4.8e-5
REGION_V16_WEIGHT_DECAY = 1.0e-4
REGION_V16_GRAD_CLIP = 1.0
REGION_V16_EMA_DECAY = 0.999

# Loss weights. These explicitly supervise topology and connectivity rather
# than relying on generic intensity/image losses to discover cracks implicitly.
REGION_V16_STRUCTURE_WEIGHT = 0.80
REGION_V16_TOPOLOGY_BCE_WEIGHT = 0.45
REGION_V16_TOPOLOGY_DICE_WEIGHT = 0.35
REGION_V16_CLDICE_WEIGHT = 0.55
REGION_V16_ORIENTATION_WEIGHT = 0.30
REGION_V16_BOUNDARY_WEIGHT = 0.30
REGION_V16_CONFIDENCE_WEIGHT = 0.20
REGION_V16_RESIDUAL_REG_WEIGHT = 0.0008

# -----------------------------------------------------------------------------
# Final TPSSR simple/hard router (same geometry criterion as the legacy model)
# -----------------------------------------------------------------------------
FINAL_HARD_MIN_MISSING_FRACTION = 0.60
FINAL_HARD_MIN_COMPONENT_PIXELS = 6
FINAL_HARD_CONNECTIVITY = 4

# Visual examples produced automatically after a full test run.
FINAL_EXPORT_EXAMPLES = True
FINAL_EXPORT_CROP_HEIGHT = 96
FINAL_EXPORT_CROP_WIDTH = 96
FINAL_EXPORT_MAX_SPATIAL_PIXELS = 5000
FINAL_EXPORT_CASES = [
    ("isolated_voxels", 5, 1),
    ("single_detector_line", 5, 1),
    ("multiple_detector_lines", 5, 1),
    ("partial_vertical_lines", 5, 1),
    ("spatial_patch", 5, 1),
    ("consecutive_spectral_bands", 5, 1),
    ("spatio_spectral_block", 5, 1),
    ("whole_spectrum_pixels", 5, 1),
    ("mixed_saturation", 5, 1),
]

# =============================================================================
# V30 GLOBAL THREE-EXPERT ROUTER: Spectral | Deep-v3.1 | V27->V27.5
# =============================================================================
# No hand-written 60% / component-size decision is used by V30 inference.
THREE_ROUTER_MODEL_DIM = 192
THREE_ROUTER_ATTENTION_HEADS = 8
THREE_ROUTER_TRANSFORMER_LAYERS = 6
THREE_ROUTER_FF_MULTIPLIER = 4
THREE_ROUTER_DROPOUT = 0.05

# A deterministic oracle cache evaluates the three frozen experts once.  Router
# epochs reuse those labels, avoiding a V27/V27.5 forward pass every epoch.
THREE_ROUTER_CACHE_TRAIN_SAMPLES = 6000
THREE_ROUTER_CACHE_VAL_SAMPLES = 1500
THREE_ROUTER_CACHE_BATCH_SIZE = 1
THREE_ROUTER_EPOCHS = 30
THREE_ROUTER_BATCH_SIZE = 8
THREE_ROUTER_NUM_WORKERS = 2
THREE_ROUTER_LEARNING_RATE = 1.2e-4
THREE_ROUTER_WEIGHT_DECAY = 1.0e-4
THREE_ROUTER_GRAD_CLIP = 1.0
THREE_ROUTER_EMA_DECAY = 0.999
THREE_ROUTER_MIN_EPOCHS = 12
THREE_ROUTER_EARLY_STOPPING_PATIENCE = 8

# Error-aware routing score.  The router predicts expected error for every
# expert and adds only a bounded learned correction.  Spectral confidence is
# used as a prior only for the analytical expert.
THREE_ROUTER_TEMPERATURE = 0.10
THREE_ROUTER_UNCERTAINTY_SCORE_WEIGHT = 1.40
THREE_ROUTER_BIAS_SCALE = 0.35
THREE_ROUTER_SPECTRAL_CONFIDENCE_WEIGHT = 0.35

# Strong router supervision.
THREE_ROUTER_ORACLE_MARGIN_WEIGHT = 2.0
THREE_ROUTER_RANKING_MARGIN = 0.08
THREE_ROUTER_CLASSIFICATION_LOSS_WEIGHT = 0.70
THREE_ROUTER_UNCERTAINTY_LOSS_WEIGHT = 0.35
THREE_ROUTER_RANKING_LOSS_WEIGHT = 0.18
THREE_ROUTER_REGRET_LOSS_WEIGHT = 0.65
THREE_ROUTER_RECON_LOSS_WEIGHT = 0.25
THREE_ROUTER_ENTROPY_LOSS_WEIGHT = 0.01

# =============================================================================
# V31 COMPONENT ROUTER + PARTIAL-OBSERVATION ADAPTER
# =============================================================================
# V31 keeps the successful ORIGINAL V27/V27.5 architecture unchanged for true
# whole-spectrum holes.  A separate adapter can use bands that remain observed
# inside partial corruption.  The learned router then chooses ONE expert for
# each complete 4-connected spatial corruption component.

V31_PARTIAL_ADAPTER_TRAIN_SAMPLES = 1400
V31_PARTIAL_ADAPTER_VAL_SAMPLES = 350
V31_PARTIAL_ADAPTER_EPOCHS = 10
V31_PARTIAL_ADAPTER_BATCH_SIZE = 1
V31_PARTIAL_ADAPTER_NUM_WORKERS = 2
V31_PARTIAL_ADAPTER_LR = 1.2e-4
V31_PARTIAL_ADAPTER_WEIGHT_DECAY = 1.0e-5
V31_PARTIAL_ADAPTER_GRAD_CLIP = 1.5
V31_PARTIAL_ADAPTER_MAX_RESIDUAL = 0.22
V31_PARTIAL_ADAPTER_MIN_MISSING_FRACTION = 0.20
V31_PARTIAL_ADAPTER_MAX_MISSING_FRACTION = 0.95
V31_PARTIAL_ADAPTER_WHOLE_PROB = 0.12

# Fixed frozen-expert oracle bank used to train the component router.
V31_COMPONENT_ROUTER_CACHE_TRAIN_SAMPLES = 4000
V31_COMPONENT_ROUTER_CACHE_VAL_SAMPLES = 1000
V31_COMPONENT_ROUTER_EPOCHS = 30
V31_COMPONENT_ROUTER_BATCH_SIZE = 8
V31_COMPONENT_ROUTER_NUM_WORKERS = 2
V31_COMPONENT_ROUTER_LR = 1.2e-4
V31_COMPONENT_ROUTER_WEIGHT_DECAY = 1.0e-4
V31_COMPONENT_ROUTER_GRAD_CLIP = 1.0
V31_COMPONENT_ROUTER_EMA_DECAY = 0.999
V31_COMPONENT_ROUTER_MIN_EPOCHS = 12
V31_COMPONENT_ROUTER_PATIENCE = 8
V31_COMPONENT_ROUTER_PREVIEW_PIXELS = 32
V31_COMPONENT_ROUTER_ORACLE_PIXELS = 48
V31_COMPONENT_ROUTER_DEEP_BATCH = 8
