# Final TPSSR — Clean Source Project

This is the cleaned source-code version of the final TPSSR model. Experimental V13/V16/V27 variants, old router experiments, audits, and abandoned branches have been removed.

## Final architecture

```text
Corrupted HSI + mask
        |
        v
Stage 1 router
   Spectral <-> Deep-v3.1
        |
        v
Hard-case gate
   | No                    | Yes
   v                       v
keep Stage-1 output   Frozen V27 Stage-A
                           |
                           v
                    V13 TopologyFirst V1
                           |
                           v
                     final reconstruction
```

V27 supplies **structural guidance only** in the final Region branch. V27's own final reconstructed pixels are not used. V13 TopologyFirst V1 synthesizes the final Region values.

## Included comparison methods

Five benchmark methods are included:

1. Spectral interpolation
2. Spatial interpolation
3. Spatio-spectral interpolation
4. PCA
5. Random Forest

Two external published methods are also included:

- PFGIN
- DDS2M

The uploaded project already contained the external source trees; they are preserved under `external/`.

## Clean project layout

```text
TPSSR_Final_Source/
├── prepare_dataset.py
├── train.py
├── test.py
├── evaluate.py
├── benchmarks.py
├── external_methods.py
├── run_all.py
├── requirements.txt
├── core/
│   ├── config.py
│   ├── data_preparation/
│   ├── reconstruction_methods/
│   │   ├── texture_restoration_router_rf_v2/   # only final training files
│   │   ├── baselines/
│   │   ├── pca/
│   │   ├── random_forest/
│   │   ├── pfgin/
│   │   └── dds2m/
│   └── src/                                    # only final runtime dependencies
├── external/
│   ├── PFGIN/
│   └── DDS2M/
├── input_raw/
├── processed_data/
├── simulated_saturation/
├── trained_models/
└── results/
```

## 1. Data placement

For one sensor family, all five acquisitions must exist because one acquisition is held out and the other four are used for training.

Example for SWIR:

```text
input_raw/
├── SWIR1/
├── SWIR2/
├── SWIR3/
├── SWIR4/
└── SWIR5/
```

Each acquisition folder should contain one ENVI measurement plus its matching white and dark references. The preparation code identifies the main, `whiteref`, and `darkref` headers automatically.

The same structure is supported for `VNIR1` ... `VNIR5`.

## 2. Install Python dependencies

Use the same CUDA/PyTorch environment in which TPSSR was developed, then install the remaining packages:

```bash
pip install -r requirements.txt
```

DDS2M also contains its original requirements file:

```text
external/DDS2M/requirements.txt
```

## 3. Prepare a fold

Example: SWIR1 is the held-out test acquisition; SWIR2-5 are training/validation acquisitions.

```bash
python prepare_dataset.py --dataset SWIR1
```

This prepares the calibrated data and generates the fixed benchmark:

- 10 corruption scenarios
- 5 severities
- 4 repeats
- 200 masks total

## 4. Train Final TPSSR from the beginning

```bash
python train.py --dataset SWIR1 --gpus 2
```

The clean training sequence is:

```text
1. Deep-v3 teacher
2. Deep-v3.1
3. V27 learned correspondence / Stage-A
4. V13 TopologyFirst V1
5. Final hierarchical router
```

Existing checkpoints are skipped automatically. To rerun every stage:

```bash
python train.py --dataset SWIR1 --gpus 2 --force
```

## 5. Test Final TPSSR

```bash
python test.py \
  --dataset SWIR1 \
  --gpus 2 \
  --max-masks 200 \
  --deep-batch 16 \
  --region-batch 32
```

The test code now explicitly prints global progress such as:

```text
[GPU 0] Mask 37/200 (18.5%) | spatio_spectral_block S4 R1
```

The run is resume-safe. Completed masks are reused unless `--force` is supplied.

Main output:

```text
results/<fold>/final_tpssr/test_metrics.csv
```

Per-mask metrics include RMSE, MAE, R², SAM, SSIM, coverage, route fractions, component counts, and runtime.

## 6. Evaluate + export visual examples

```bash
python evaluate.py --dataset SWIR1 --visuals 3
```

Evaluation creates:

```text
results/<fold>/final_tpssr/
├── SUMMARY.txt
├── summary.json
├── scenario_summary.csv
├── severity_summary.csv
└── visuals/
    ├── 01_whole_spectrum_pixels_....png
    ├── 02_spatio_spectral_block_....png
    └── 03_mixed_saturation_....png
```

Each visual is a side-by-side comparison of:

```text
Original | Corrupted | Final TPSSR
```

The default examples are hard Severity-5 cases and can be changed in `evaluate.py`.

## 7. Run the five benchmark methods

Run one method:

```bash
python benchmarks.py --dataset SWIR1 --method spectral --stage both
python benchmarks.py --dataset SWIR1 --method spatial --stage both
python benchmarks.py --dataset SWIR1 --method spatio_spectral --stage both
python benchmarks.py --dataset SWIR1 --method pca --stage both
python benchmarks.py --dataset SWIR1 --method random_forest --stage both
```

Run all five:

```bash
python benchmarks.py --dataset SWIR1 --method all --stage both
```

PCA and Random Forest have global training stages. Spectral, Spatial, and Spatio-spectral are analytical and therefore skip training.

The benchmark test scripts also report `Mask i/200` progress.

## 8. External methods

### PFGIN

```bash
python external_methods.py --dataset SWIR1 --method pfgin --stage both
```

PFGIN performs global training and then benchmark testing.

### DDS2M

```bash
python external_methods.py --dataset SWIR1 --method dds2m --stage both
```

DDS2M does not have a global training stage in this wrapper; it performs its self-supervised optimization per corrupted case/tile during testing.

Both external test wrappers report mask progress and retain their existing resume/cache behavior.

## 9. Optional one-command workflow

Final TPSSR only:

```bash
python run_all.py --dataset SWIR1 --prepare --final
```

Final TPSSR + five benchmarks:

```bash
python run_all.py --dataset SWIR1 --prepare --final --benchmarks
```

Everything, including PFGIN and DDS2M:

```bash
python run_all.py --dataset SWIR1 --everything
```

The last command can take a very long time because the external methods are expensive.

## Final SWIR1 reference result

The finalized end-to-end architecture was previously validated on the fixed 200-mask SWIR1 benchmark with full coverage. The clean project preserves that architecture and evaluation protocol; fresh training can vary slightly because neural-network optimization is stochastic.
