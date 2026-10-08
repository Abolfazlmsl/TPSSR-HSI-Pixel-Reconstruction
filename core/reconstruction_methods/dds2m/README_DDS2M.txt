DDS2M BASELINE INTEGRATION
==========================

This folder integrates the official ICCV 2023 DDS2M implementation into the
same saturation benchmark used by the other reconstruction methods.

Official method:
Y. Miao, L. Zhang, L. Zhang, D. Tao,
"DDS2M: Self-Supervised Denoising Diffusion Spatio-Spectral Model for
Hyperspectral Image Restoration", ICCV 2023.

The official source code is NOT copied into this project. setup_official.py
clones the authors' public repository into external/DDS2M.

FAIRNESS DETAILS
----------------

- The same 200 benchmark masks are reused.
- The same select_eval_coords() sampling is reused.
- DDS2M receives no training images; it remains self-supervised.
- Hidden saturation values are zeroed before DDS2M sees a tile.
- Evaluation uses the final x_recon, not the upstream x_best. The public
  reproduction code chooses x_best with clean-reference PSNR, which would be
  oracle selection for our held-out benchmark.
- The public code assumes square images, so the test block is evaluated through
  overlapping square tiles while keeping the original voxel coordinates.

FIRST SETUP
-----------

From the project root:

    python reconstruction_methods/dds2m/setup_official.py

If imports from the official repository are missing, either use a separate
compatible environment or run:

    python reconstruction_methods/dds2m/setup_official.py --install-requirements

QUICK SMOKE TEST
----------------

Temporarily set in config.py:

    DDS2M_MAX_MASKS = 1

Then run:

    python reconstruction_methods/dds2m/test.py

After it works, restore:

    DDS2M_MAX_MASKS = None

FULL RUN
--------

    python reconstruction_methods/dds2m/test.py

Output:

    results/<experiment>/dds2m/metrics.csv

Then:

    python evaluation/compare_methods.py

RUNTIME
-------

DDS2M performs self-supervised optimization during reconstruction and is much
slower than the analytical/PCA/RF methods. The public completion entry point
uses 3000 diffusion steps by default. The integration caches completed tiles,
so an interrupted run can resume.

Cache:

    results/<experiment>/dds2m/cache/

PUBLICATION SETTINGS
--------------------

Current defaults keep the public command-line settings:

    DDS2M_TIMESTEPS = 3000
    DDS2M_START_POINT = 1500
    DDS2M_RANK = 10
    DDS2M_ETA = 0.95
    DDS2M_ETA_B = 1.0

DDS2M_SIGMA_0 = 0.0 because our benchmark tests missing/saturated-value
reconstruction without adding a second Gaussian-noise corruption.

DDS2M_TILE_SIZE = 128 is used initially to make the 200-mask benchmark more
practical. The public completion config used 256 x 256 inputs. For a final
publication run, if memory and runtime permit, test:

    DDS2M_TILE_SIZE = 256
    DDS2M_TILE_STRIDE = 128
