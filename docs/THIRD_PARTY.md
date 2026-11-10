# External benchmark implementations

PFGIN and DDS2M are independent methods developed by their respective authors.
Their source code is **not included** in this repository. The setup scripts
clone the upstream repositories into `external/`.

- PFGIN: <https://github.com/yinhuwu/PFGIN>
- DDS2M: <https://github.com/miaoyuchun/DDS2M>

Follow the licenses and attribution requirements in the upstream repositories.
The root `LICENSE` covers original TPSSR code, not third-party repositories.

Install (from the project root):

```bash
python core/reconstruction_methods/pfgin/setup_official.py
python core/reconstruction_methods/dds2m/setup_official.py
```

The DDS2M setup script applies a small disk-storage patch in the cloned checkout.
The evaluation adapter also uses overlapping square tiles and avoids selecting
reconstructions using held-out ground truth. PFGIN uses 16-band spectral input
windows with project-specific preprocessing and masking. See the adapter sources
in `core/reconstruction_methods/` for implementation details.
