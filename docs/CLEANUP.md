# Repository maintenance

The public workflow is exposed by `tpssr.py`; the underlying train/test scripts
remain in place because source code and checkpoints depend on these paths.

The archive contains one-off diagnostics, comparisons, obsolete PCA/RF variants,
and an embedded copy of the external benchmarks. The current benchmark runner
uses **standard spectral PCA** and **multiscale RF** only.

Data, model weights, generated plots and evaluation caches are ignored by Git.
Do not upload private acquisition files or unpublished model checkpoints
without a separate data-sharing decision.

A successful source-level regression run does not verify GPU inference or
reproduce published numerical results. Validate a checkpoint-backed 200-mask
run before citing new metrics.
