import numpy as np
import torch
from torch.utils.data import Dataset

from src.training_corruptions import generate_training_mask
from .common import normalize_cube_patch


class RandomPFGINDataset(Dataset):
    def __init__(self, cubes, lo, hi, patch_size, spectral_window, length, seed):
        self.cubes = cubes
        self.lo = np.asarray(lo, dtype=np.float32)
        self.hi = np.asarray(hi, dtype=np.float32)
        self.patch = int(patch_size)
        self.window = int(spectral_window)
        self.length = int(length)
        self.seed = int(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + int(index) * 104729)
        cube = self.cubes[int(rng.integers(0, len(self.cubes)))]
        h, w, bands = cube.shape
        if bands < self.window:
            raise RuntimeError(f"PFGIN requires at least {self.window} spectral bands; got {bands}.")

        size = min(self.patch, h, w)
        y0 = int(rng.integers(0, max(1, h - size + 1)))
        x0 = int(rng.integers(0, max(1, w - size + 1)))
        b0 = int(rng.integers(0, max(1, bands - self.window + 1)))

        patch = np.asarray(cube[y0:y0 + size, x0:x0 + size, b0:b0 + self.window], dtype=np.float32)
        if size < self.patch:
            py = self.patch - patch.shape[0]
            px = self.patch - patch.shape[1]
            patch = np.pad(patch, [(0, py), (0, px), (0, 0)], mode="reflect")

        finite = np.isfinite(patch)
        patch = normalize_cube_patch(patch, self.lo[b0:b0 + self.window], self.hi[b0:b0 + self.window])
        mask, _, _ = generate_training_mask(patch.shape, rng)
        # Hide any invalid source values from the network as well.
        mask |= ~finite

        image = torch.from_numpy(np.transpose(patch, (2, 0, 1)).copy())
        mask_t = torch.from_numpy(np.transpose(mask.astype(np.float32), (2, 0, 1)).copy())
        return image, mask_t
