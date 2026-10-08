PFGIN integration for the HSI saturation-reconstruction benchmark
=================================================================

Published method
----------------
Y. Wu, J. Zhang, D. Liu, "Predictive Filtering Integrated Generative Remote Sensing Hyperspectral Image Inpainting," IEEE Geoscience and Remote Sensing Letters, 2025.
Official code: https://github.com/yinhuwu/PFGIN

What this integration does
--------------------------
- Uses the official PFGIN InpaintGenerator_SS / InpaintingModel implementation.
- Keeps the official 16-band spectral window and [3,7,3,11] predictive spatial-spectral kernels.
- Keeps the public repository's GAN/L1/content/style loss weights.
- Replaces the repository's hard-coded Windows TIFF loader with this project's calibrated SWIR arrays.
- Trains ONLY on TRAIN_DATASET_FOLDERS and validates on their validation blocks.
- Tests on the held-out acquisition and the exact same benchmark masks used by the other methods.
- Fits normalization only from training acquisitions, preventing test-ground-truth leakage.

Installation
------------
1) Extract this patch into the project root (merge folders; do not delete data/results).
2) From the project root run:

   python apply_pfgin_config.py
   python reconstruction_methods/pfgin/setup_official.py

3) First smoke-test training (config default PFGIN_SMOKE_TEST=True):

   python reconstruction_methods/pfgin/train.py

This runs only 50 iterations to reveal dependency/shape/CUDA problems.

4) If the smoke test succeeds, set in config.py:

   PFGIN_SMOKE_TEST = False

Then run train.py again. It resumes from the smoke checkpoint and continues toward 40,000 iterations.

5) For a one-mask evaluation keep:

   PFGIN_MAX_MASKS = 1

and run:

   python reconstruction_methods/pfgin/test.py

6) After the model is fully trained, set:

   PFGIN_MAX_MASKS = None

and run test.py for all benchmark masks.

Important
---------
Do NOT use the 50-iteration smoke checkpoint for paper comparisons. The public PFGIN configuration trains to 40,000 iterations.

The evaluation/compare_methods.py script already discovers metrics.csv files automatically, so no compare_methods.py change is required.
