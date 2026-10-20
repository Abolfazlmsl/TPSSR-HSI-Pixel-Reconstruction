# TPSSR — Hyperspectral Pixel Reconstruction

This source distribution preserves the **Learned Router3 Aligned** configuration selected for the SWIR experiments. It combines three reconstruction experts: spectral interpolation, a learned spectral–neighbor fusion model, and geometry/topology-based Region reconstruction. A source-trained quality router selects among eligible experts using observed information only.

The model also retains the frozen band-gated CNN/Transformer backbone and the refined Region output. The candidate Adaptive spectral gate and experimental Region Spectral Guidance are **not** included in the selected inference route.

## Installation without losing previous experiments

This is a **code-only distribution**. It does **not** contain `processed_data/`, `trained_models/`, `simulated_saturation/`, `input_raw/`, or `results/`. Checkpoint and result folders must remain in place. Save a backup of the running project before extracting this archive into the same project directory; files with matching code paths can replace the old versions.

No migration of checkpoint files is performed. Checkpoints from different splits (SWIR 8×2, A→B, B→A) must not be exchanged. The existing experiment folders should remain intact, including the six-core shadow Region checkpoints used during router calibration.

The archived source has no cached bytecode, diagnostic probe scripts, or stored DDS2M trial logs. The independently authored external baseline implementations remain under `external/` with their documentation and must retain applicable attribution and licenses.

## Dataset splits

| Preset | Source acquisitions | Held-out test acquisitions | Source calibration/validation acquisitions |
|---|---|---|---|
| `swir_8x2` | SWIR1–4 and SWIR6–9 | **SWIR5, SWIR10** | SWIR4, SWIR9 |
| `swir_region_a_to_b` | SWIR1–5 | SWIR6–10 | SWIR4, SWIR5 |
| `swir_region_b_to_a` | SWIR6–10 | SWIR1–5 | SWIR9, SWIR10 |
| `vnir_8x2` | VNIR1–4 and VNIR6–9 | VNIR5, VNIR10 | VNIR4, VNIR9 |

Under `swir_8x2`, the Router's auxiliary (shadow) Region experts are fitted on SWIR1–3 and SWIR6–8; their held-out source calibration cores are SWIR4 and SWIR9. **SWIR5 and SWIR10 never enter this fit.** The original eight-source Region model and the separately trained spectral models are retained. The Router's training labels use a refined Region prediction produced by the shadow model.

For the main SWIR 8×2 test, a total budget of **200 masks** means **100 masks for SWIR5 and 100 for SWIR10**, not 200 for each. Preserve the existing deterministic mask files when conducting paired comparisons.

## Verify the selected model (two GPUs)

Run these commands at the project root, in the environment used for the accepted TPSSR checkpoints:

```bash
python preflight.py --preset swir_8x2
python test.py --preset swir_8x2 --router learned_router3_aligned --gpus 2 --max-masks 200
python evaluate.py --preset swir_8x2 --router learned_router3_aligned --visuals 0
```

`preflight.py` checks for the learned fusion and three-expert router checkpoints as well as the base network weights. The `test.py` and `evaluate.py` commands require `--router learned_router3_aligned`: the baseline V2.3 inference mode is still available explicitly for historical comparisons.

The combined outputs are written under:

```
results/swir__train_1-2-3-4-6-7-8-9__test_5-10/combined/tpssr_learned_router3_aligned/
```

Do not use `--force` unless the intention is to overwrite completed mask-level results.

## Training sequence for a genuinely new split

If checkpoints for that split already exist, keep them; the trainers normally skip accepted checkpoint files. For a new split, the ordered stages are:

```bash
python preflight.py --preset swir_8x2 --data-only
python train.py --preset swir_8x2 --stage all --gpus 2 --dry-run
python train.py --preset swir_8x2 --stage all --gpus 2
python train.py --preset swir_8x2 --stage fusion --gpus 2
python train.py --preset swir_8x2 --stage router3_aligned --gpus 2
python preflight.py --preset swir_8x2
```

The `all` stage trains the accepted spectral backbone, confidence gate, Geometry, and Topology. It does not automatically train learned fusion or the three-expert router; those are separate stages. The router stage may fit the auxiliary shadow Region model if the corresponding checkpoint files are absent. This can be computationally expensive. In normal post-training evaluation none of these steps needs to be repeated.

## Reference methods and paired comparisons

The source tree retains the classical/reference benchmark entrypoints and the wrappers for PFGIN and DDS2M. Existing baseline runs should be reused rather than recomputed without a specific reason.

```bash
python benchmarks.py --preset swir_8x2 --method spectral --stage test --max-masks 200
python plot_method_comparison.py --experiment-dir results/swir__train_1-2-3-4-6-7-8-9__test_5-10
```

For the manual-gated comparator, use the same masks:

```bash
python test.py --preset swir_8x2 --router learned_fusion_guarded --gpus 2 --max-masks 200
python evaluate.py --preset swir_8x2 --router learned_fusion_guarded --visuals 0
```

**Reference results from the completed 200-mask SWIR 8×2 tests (macro-averaged per mask):**

| Mode | RMSE | Coverage |
|---|---:|---:|
| TPSSR V2.3 | 0.007788 | 100% |
| Learned Fusion Guarded (manual branch selection) | 0.005106 | 100% |
| Learned Router3 Aligned (learned branch selection) | 0.005119 | 100% |

The learned router is approximately 0.25% above the best manual comparator in macro-RMSE; the two are not claimed to be statistically equivalent. The learned router is preferred for the selected architecture because branch selection is data-driven. Report the statistical protocol honestly: SWIR5 and SWIR10 are separate from weight fitting in the final 8×2 run, but their earlier results informed architecture development. This is **not** a never-seen independent dataset.

Reported RMSE, MAE, R² and SAM aggregates in `COMBINED_SUMMARY.txt` are means of valid per-mask metrics, not voxel-pooled quantities. A method with incomplete coverage cannot be compared as if it made the same predictions as a 100%-coverage method. For the final paper, compare identical masks and common reconstructed voxels where relevant. TPSSR SSIM is reported only when its mask-specific calculation is valid.

## Code map

- `train.py`: four base training stages, learned neighbor fusion and three-expert Router calibration.
- `training/neighbor_fusion.py`: learned spectral and local-donor fusion.
- `training/three_expert_router.py`: source-only Router data/labels, fitting and checkpoint verification.
- `core/src/pipeline.py`: frozen experts and selected inference routes.
- `core/src/spatial_inference.py`: Geometry/Topology Region inference.
- `test.py`: mask-level reconstruction and progress reporting.
- `evaluate.py`: per-acquisition and combined metric summaries and visuals.
- `preflight.py`: data and checkpoint layout validation.
- `benchmarks.py`, `external_methods.py`: benchmark tools, not part of the final learned Router.

## Software checks

```bash
python -m compileall -q train.py test.py evaluate.py training core
python -m pytest -q tests
python train.py --preset swir_8x2 --stage router3_aligned --gpus 2 --dry-run
```

These checks validate syntax, selected interfaces and the training plan. They do **not** reproduce the numerical results without the exact processed data, masks, checkpoints and server environment.
