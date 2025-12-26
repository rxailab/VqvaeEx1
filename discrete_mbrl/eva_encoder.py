#!/usr/bin/env python3
"""
Standalone AE/VQ-VAE encoder-decoder evaluation for MiniGrid using env.render() frames.

Key fix:
- MiniGrid obs["image"] is 7x7x3 (symbolic) -> too small for your conv AE
- Use env.render() (rgb_array) and resize to a reasonable size (default 64x64)

Repo-specific:
- model_construction.py is in the same folder
- construct_ae_model signature: construct_ae_model(input_dim, args, load=True, latent_activation=False)
- make_ae_v1 expects input_dim as CHW tuple (C,H,W)
- model_construction/training_helpers expect args.wandb/args.comet_ml/etc. -> patched defaults
"""

import os
import sys
import argparse
from typing import Any, Dict, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


# --- Ensure local imports resolve from this folder (discrete_mbrl/) ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)


def try_make_env(env_name: str, seed: int = 0):
    """
    Prefer project helper make_env; fallback to gymnasium.make(render_mode="rgb_array").
    """
    try:
        from env_helpers import make_env  # type: ignore
        env = make_env(env_name, seed=seed)
        return env
    except Exception:
        import gymnasium as gym  # type: ignore
        env = gym.make(env_name, render_mode="rgb_array")
        try:
            env.reset(seed=seed)
        except TypeError:
            pass
        return env


def ensure_args_fields(args: argparse.Namespace, defaults: Dict[str, Any]) -> argparse.Namespace:
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args


def patch_eval_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        # tracking/logging toggles
        "wandb": False,
        "comet_ml": False,
        "tensorboard": False,
        "mlflow": False,

        # saving/resume flags
        "save": False,
        "save_dir": "",
        "log_dir": "",
        "model_dir": "",
        "results_dir": "",
        "resume": False,
        "resume_path": "",
        "debug": False,

        # experiment identity
        "exp_name": "eval",
        "run_name": "eval",
        "project": "",
        "group": "",
        "tags": "",

        # AE-specific
        "ae_model_version": getattr(args, "ae_model_version", 1),
        "ae_model_hash": "",
        "ae_model_name": "",

        # Trainer/optimizer defaults (constructor expects these)
        "learning_rate": 1e-4,
        "ae_grad_clip": 1.0,
        "grad_clip": 1.0,
        "weight_decay": 0.0,
        "beta1": 0.9,
        "beta2": 0.999,

        # Often referenced batch/epochs even if unused here
        "b": 32,
        "batch_size": 32,
        "epochs": 1,

        "seed": getattr(args, "seed", 0),
    }
    return ensure_args_fields(args, defaults)



def try_construct_ae(input_dim, args: argparse.Namespace) -> torch.nn.Module:
    from model_construction import construct_ae_model  # local file
    args = patch_eval_defaults(args)

    out = construct_ae_model(input_dim, args, load=False)

    # Many repos return (model, trainer) or (encoder, decoder, model, trainer).
    if isinstance(out, torch.nn.Module):
        return out

    if isinstance(out, (tuple, list)):
        # Prefer the first torch.nn.Module we can find
        for item in out:
            if isinstance(item, torch.nn.Module):
                return item

        # Or a dict-like tuple element
        for item in out:
            if isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, torch.nn.Module):
                        return v

        raise RuntimeError(
            f"construct_ae_model returned a tuple/list but no nn.Module found. "
            f"Types: {[type(x) for x in out]}"
        )

    raise RuntimeError(f"Unexpected return type from construct_ae_model: {type(out)}")



def load_checkpoint_into_model(model, ckpt_path, device, strict=True):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = None
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "ae_state_dict", "net"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state_dict = ckpt[k]
                break
        if state_dict is None and all(isinstance(k, str) for k in ckpt.keys()):
            state_dict = ckpt
    else:
        state_dict = ckpt

    cleaned = {}
    for k, v in state_dict.items():
        nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        cleaned[nk] = v

    model.load_state_dict(cleaned, strict=strict)

def render_frame(env) -> np.ndarray:
    """
    Get an RGB frame (H,W,3) uint8 from env.render().
    """
    frame = env.render()
    if frame is None:
        raise RuntimeError(
            "env.render() returned None. Ensure env was created with render_mode='rgb_array' "
            "(fallback does this; your make_env might not)."
        )
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError(f"Unexpected render frame shape: {frame.shape}")
    if frame.dtype != np.uint8:
        frame = frame.clip(0, 255).astype(np.uint8)
    return frame


def resize_hwc_uint8(img: np.ndarray, size: int) -> np.ndarray:
    """
    Resize HWC uint8 RGB to (size, size, 3).
    Use NEAREST to keep grid-like style crisp.
    """
    pil = Image.fromarray(img)
    pil = pil.resize((size, size), resample=Image.NEAREST)
    out = np.array(pil, dtype=np.uint8)
    return out


def hwc_uint8_to_torch(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    HWC uint8 -> torch float32 BCHW in [0,1]
    """
    x = torch.from_numpy(img).to(device=device)
    x = x.float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
    return x


def torch_to_hwc_uint8(x: torch.Tensor) -> np.ndarray:
    """
    torch BCHW/CHW float -> HWC uint8
    """
    if x.ndim == 4:
        x = x[0]
    x = x.detach().float().cpu()

    # Some decoders output [-1,1] or other ranges; clamp to [0,1] best-effort
    # If your recon looks too dark/bright, tell me and we’ll align normalization precisely.
    x = x.clamp(0, 1)

    x = x.permute(1, 2, 0).numpy()
    x = (x * 255.0).round().clip(0, 255).astype(np.uint8)
    return x


@torch.no_grad()
def reconstruct(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Handle common AE/VQ-VAE forward outputs.
    """
    model.eval()
    out = model(x)
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        return out[0]
    if isinstance(out, dict):
        for key in ["recon", "reconstruction", "x_recon", "x_hat", "decoded"]:
            if key in out and isinstance(out[key], torch.Tensor):
                return out[key]
    raise RuntimeError("Model forward output not recognized; adapt reconstruct() for your AE.")


def make_grid(images: np.ndarray, ncols: int) -> np.ndarray:
    """
    images: (N,H,W,C)
    """
    N, H, W, C = images.shape
    ncols = max(1, ncols)
    nrows = int(np.ceil(N / ncols))
    grid = np.zeros((nrows * H, ncols * W, C), dtype=np.uint8)
    for i in range(N):
        r, c = divmod(i, ncols)
        grid[r * H:(r + 1) * H, c * W:(c + 1) * W, :] = images[i]
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="MiniGrid-SimpleCrossingS9N1-v0")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--ncols", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="recon.png")
    parser.add_argument("--show", action="store_true")

    # IMPORTANT: render frame size to feed AE (match your AE training size)
    parser.add_argument("--img_size", type=int, default=64, help="Resize env.render() frames to this square size")

    # AE constructor args
    parser.add_argument("--ae_model_type", type=str, default="vqvae")
    parser.add_argument("--ae_model_version", type=int, default=1)
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--filter_size", type=int, default=9)

    # compatibility flags (ignored in eval)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--comet_ml", action="store_true")

    args = parser.parse_args()
    args = patch_eval_defaults(args)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device={device}")

    env = try_make_env(args.env_name, seed=args.seed)
    obs, _ = env.reset(seed=args.seed) if hasattr(env, "reset") else (env.reset(), {})

    # Build AE with CHW tuple matching resized render frames
    input_dim = (3, args.img_size, args.img_size)
    print(f"[INFO] using input_dim={input_dim} (CHW) from --img_size={args.img_size}")

    ae = try_construct_ae(input_dim, args).to(device)
    load_checkpoint_into_model(ae, args.ckpt, device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    originals, recons = [], []

    for _ in range(args.steps):
        frame = render_frame(env)                 # HWC, uint8
        frame = resize_hwc_uint8(frame, args.img_size)
        originals.append(frame)

        x = hwc_uint8_to_torch(frame, device)     # BCHW float [0,1]
        xhat = reconstruct(ae, x)
        recon = torch_to_hwc_uint8(xhat)
        recons.append(recon)

        action = env.action_space.sample() if hasattr(env, "action_space") else 0
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, _, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            obs, _, done, _ = step_out

        if done:
            obs, _ = env.reset(seed=args.seed)

    originals = np.stack(originals, axis=0)
    recons = np.stack(recons, axis=0)

    orig_grid = make_grid(originals, ncols=args.ncols)
    recon_grid = make_grid(recons, ncols=args.ncols)

    sep_h = 8
    sep = np.ones((sep_h, orig_grid.shape[1], 3), dtype=np.uint8) * 255
    combined = np.concatenate([orig_grid, sep, recon_grid], axis=0)

    plt.figure(figsize=(max(8, args.ncols * 1.2), 10))
    plt.imshow(combined)
    plt.axis("off")
    plt.title(f"Top: env.render() resized to {args.img_size} | Bottom: Reconstruction\nenv={args.env_name}")
    plt.tight_layout()
    plt.savefig(args.save_path, dpi=200)
    print(f"[OK] Saved comparison image to: {args.save_path}")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
