from pathlib import Path
import sys

HERE = Path(__file__).resolve()
PROJECT_ROOT = next(parent for parent in HERE.parents if (parent / "config.py").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.evaluation_utils import load_global_training_data
from reconstruction_methods.pfgin.common import (
    require_official_repo,
    import_official_models,
    make_official_config,
    unwrap,
    fit_scaler,
    load_scaler,
)
from reconstruction_methods.pfgin.dataset import RandomPFGINDataset


def checkpoint_payload(model, iteration, best_val):
    return {
        "iteration": int(iteration),
        "best_val": float(best_val),
        "generator": unwrap(model.generator).state_dict(),
        "discriminator": unwrap(model.discriminator).state_dict(),
        "gen_optimizer": model.gen_optimizer.state_dict(),
        "dis_optimizer": model.dis_optimizer.state_dict(),
    }


def save_checkpoint(path, model, iteration, best_val):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, iteration, best_val), path)


def load_checkpoint(path, model, device):
    payload = torch.load(path, map_location=device)
    unwrap(model.generator).load_state_dict(payload["generator"])
    unwrap(model.discriminator).load_state_dict(payload["discriminator"])
    model.gen_optimizer.load_state_dict(payload["gen_optimizer"])
    model.dis_optimizer.load_state_dict(payload["dis_optimizer"])
    model.iteration = int(payload.get("iteration", 0))
    return model.iteration, float(payload.get("best_val", np.inf))


@torch.no_grad()
def validate(model, loader, device, max_batches=8):
    model.eval()
    errors = []
    for batch_index, (images, masks) in enumerate(loader):
        if batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        outputs = model(images, masks)
        target_mask = masks > 0.5
        if target_mask.any():
            diff = outputs[target_mask] - images[target_mask]
            errors.append(float(torch.sqrt(torch.mean(diff * diff)).cpu()))
    model.train()
    return float(np.mean(errors)) if errors else np.inf


def main():
    repo = require_official_repo(config)
    official_models = import_official_models(repo)

    if not torch.cuda.is_available():
        raise RuntimeError("PFGIN training is configured for CUDA. No CUDA GPU was found.")

    gpu_ids = list(config.PFGIN_GPU_IDS)
    device = torch.device(f"cuda:{gpu_ids[0]}")
    torch.cuda.set_device(gpu_ids[0])
    torch.backends.cudnn.benchmark = True

    train_cubes, val_cubes, _ = load_global_training_data()
    model_dir = config.MODELS_DIR / "pfgin"
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = model_dir / "scaler.npz"

    if scaler_path.exists():
        lo, hi = load_scaler(scaler_path)
    else:
        print("Fitting PFGIN per-band normalization from TRAIN acquisitions only...")
        lo, hi = fit_scaler(train_cubes, config, scaler_path)

    max_iters = int(config.PFGIN_SMOKE_ITERS if config.PFGIN_SMOKE_TEST else config.PFGIN_MAX_ITERS)
    train_length = max_iters * int(config.PFGIN_BATCH_SIZE)
    train_ds = RandomPFGINDataset(
        train_cubes, lo, hi,
        config.PFGIN_PATCH_SIZE,
        config.PFGIN_SPECTRAL_WINDOW,
        train_length,
        config.SEED + 1000,
    )
    val_ds = RandomPFGINDataset(
        val_cubes, lo, hi,
        config.PFGIN_PATCH_SIZE,
        config.PFGIN_SPECTRAL_WINDOW,
        int(config.PFGIN_VAL_SAMPLES),
        config.SEED + 500000,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config.PFGIN_BATCH_SIZE),
        shuffle=False,
        num_workers=int(config.PFGIN_NUM_WORKERS),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config.PFGIN_BATCH_SIZE),
        shuffle=False,
        num_workers=min(2, int(config.PFGIN_NUM_WORKERS)),
        pin_memory=True,
    )

    official_cfg = make_official_config(config, model_dir, device, training=True)
    model = official_models.InpaintingModel(official_cfg).to(device)
    if len(gpu_ids) > 1:
        model.generator = torch.nn.DataParallel(model.generator, device_ids=gpu_ids)
        model.discriminator = torch.nn.DataParallel(model.discriminator, device_ids=gpu_ids)

    latest = model_dir / "latest.pt"
    best = model_dir / "best.pt"
    start_iter = 0
    best_val = np.inf
    if config.PFGIN_RESUME and latest.exists():
        start_iter, best_val = load_checkpoint(latest, model, device)
        print(f"Resuming PFGIN from iteration {start_iter}.")

    if start_iter >= max_iters:
        print(f"Checkpoint already reached {start_iter} iterations (target {max_iters}).")
        if config.PFGIN_SMOKE_TEST:
            print("Smoke test is complete. Set PFGIN_SMOKE_TEST=False and rerun train.py for the full 40k training.")
        return

    model.train()
    pbar = tqdm(total=max_iters, initial=start_iter, desc="PFGIN training")
    iteration = start_iter

    for images, masks in train_loader:
        if iteration >= max_iters:
            break
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs, gen_loss, dis_loss, _logs = model.process(images, masks)
        if not torch.isfinite(gen_loss) or not torch.isfinite(dis_loss):
            raise RuntimeError(f"Non-finite PFGIN loss at iteration {iteration + 1}.")
        model.backward(gen_loss, dis_loss)

        iteration += 1
        model.iteration = iteration
        pbar.update(1)
        if iteration % 20 == 0:
            pbar.set_postfix(gen=f"{gen_loss.item():.4g}", dis=f"{dis_loss.item():.4g}")

        do_eval = (iteration % int(config.PFGIN_EVAL_INTERVAL) == 0) or iteration == max_iters
        if config.PFGIN_SMOKE_TEST and iteration == max_iters:
            do_eval = True

        if do_eval:
            val_rmse = validate(model, val_loader, device)
            print(f"\nPFGIN validation masked RMSE @ {iteration}: {val_rmse:.6f}")
            if val_rmse < best_val:
                best_val = val_rmse
                save_checkpoint(best, model, iteration, best_val)
                print("Saved new best PFGIN checkpoint.")
            save_checkpoint(latest, model, iteration, best_val)

    pbar.close()
    save_checkpoint(latest, model, iteration, best_val)
    if not best.exists():
        save_checkpoint(best, model, iteration, best_val)

    print("\nPFGIN training completed.")
    print(f"Latest: {latest}")
    print(f"Best:   {best}")
    if config.PFGIN_SMOKE_TEST:
        print("\nSMOKE TEST ONLY. Set PFGIN_SMOKE_TEST=False and rerun train.py; it will resume toward 40,000 iterations.")


if __name__ == "__main__":
    main()
