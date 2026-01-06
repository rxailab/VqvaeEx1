#!/usr/bin/env python3
"""
debug_vqvae_recon.py (FULL UPDATED)

This version is designed for your repo layout:
  <repo_root>/
    discrete_mbrl/
      env_helpers.py
      model_construction.py
      training_helpers.py
      data_logging.py
      ...

It isolates the Autoencoder (VQ-VAE) and produces an AE-only recon grid,
BUT also prints the critical diagnostics that explain "all-black recon":

- Confirms obs range/shape
- Loads the correct checkpoint for args.env_name (prefers discrete_mbrl/models/<env_name>/)
- Runs encode->decode on the SAME observation space the AE was trained on (CHW float [0,1])
- Prints:
    - encode output stats
    - reconstruction stats
    - (if available) quantizer codebook stats + index usage (unique indices / histogram)
- Saves: vqvae_recon_debug.png (or --out)

This script does NOT rely on env.render() unless forced. It uses env observation.

Usage (use SAME args as training for correct AE construction):
python debug_vqvae_recon.py \
  --env_name MiniGrid-SimpleCrossingS9N1-v0 \
  --ae_model_type vqvae \
  --device cuda \
  --codebook_size 512 \
  --embedding_dim 128 \
  --latent_dim 128 \
  --filter_size 9 \
  --n_frames 32 \
  --batch 16 \
  --out vqvae_recon_debug.png
"""

import os
import sys
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# =============================================================================
# Import-path fix for repo layout
# =============================================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

cur = THIS_DIR
repo_root = None
for _ in range(10):
    if os.path.isdir(os.path.join(cur, "discrete_mbrl")):
        repo_root = cur
        break
    cur = os.path.dirname(cur)

if repo_root is None:
    raise RuntimeError(f"Could not locate 'discrete_mbrl' folder by walking up from: {THIS_DIR}")

DISCRETE_MBRL_DIR = os.path.join(repo_root, "discrete_mbrl")
if DISCRETE_MBRL_DIR not in sys.path:
    sys.path.insert(0, DISCRETE_MBRL_DIR)

print("[debug] repo_root =", repo_root)
print("[debug] sys.path[0] =", sys.path[0])

# Repo modules
from env_helpers import make_env
from model_construction import construct_ae_model
from training_helpers import make_argparser, process_args
from data_logging import init_experiment


# =============================================================================
# Helpers
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _obs_from_reset(reset_out: Any):
    return reset_out[0] if isinstance(reset_out, tuple) else reset_out


def _step_unpack(step_out: Tuple):
    # gymnasium: obs, reward, terminated, truncated, info
    if len(step_out) == 5:
        obs, _, terminated, truncated, _ = step_out
        done = terminated or truncated
        return obs, done
    # old gym: obs, reward, done, info
    obs, _, done, _ = step_out
    return obs, done


def _is_hwc_image(x) -> bool:
    return isinstance(x, np.ndarray) and x.ndim == 3 and x.shape[-1] in (1, 3, 4)


def _is_chw_image(x) -> bool:
    return isinstance(x, np.ndarray) and x.ndim == 3 and x.shape[0] in (1, 3, 4)


def obs_to_hwc_float01(obs: np.ndarray) -> np.ndarray:
    """
    Convert obs to HWC float32 in [0,1].
    Supports CHW or HWC.
    """
    if not isinstance(obs, np.ndarray) or obs.ndim != 3:
        raise ValueError(f"Expected 3D ndarray obs, got type={type(obs)} ndim={getattr(obs,'ndim',None)}")

    if _is_chw_image(obs) and not _is_hwc_image(obs):
        obs = np.transpose(obs, (1, 2, 0))  # CHW->HWC

    if obs.shape[-1] == 4:
        obs = obs[..., :3]
    if obs.shape[-1] == 1:
        obs = np.repeat(obs, 3, axis=-1)

    obs = obs.astype(np.float32)
    if obs.max() > 1.5:  # likely 0..255
        obs = obs / 255.0

    obs = np.clip(obs, 0.0, 1.0)
    return obs


def to_torch_nchw(frames_hwc_float01: np.ndarray, device: str) -> torch.Tensor:
    """
    (N,H,W,3) float32 -> (N,3,H,W) float32
    """
    x = torch.from_numpy(frames_hwc_float01).permute(0, 3, 1, 2).contiguous()
    return x.to(device=device, dtype=torch.float32)


def tensor_to_uint8_hwc(x: torch.Tensor) -> np.ndarray:
    """
    x: (N,C,H,W) float [0,1] -> uint8 (N,H,W,C)
    """
    x = x.detach().cpu().clamp(0, 1)
    x = (x * 255.0).round().byte()
    return x.permute(0, 2, 3, 1).numpy()


def save_grid(real_uint8: np.ndarray, recon_uint8: np.ndarray, out_path: str, n_cols: int = 8, title: str = ""):
    n = min(len(real_uint8), len(recon_uint8))
    real_uint8 = real_uint8[:n]
    recon_uint8 = recon_uint8[:n]

    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(2 * n_rows, n_cols, figsize=(n_cols * 2.0, 2 * n_rows * 2.0))
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    axes = axes.reshape(2 * n_rows, n_cols)

    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            for rr in [0, 1]:
                ax = axes[2 * r + rr, c]
                ax.axis("off")
                if idx >= n:
                    continue
                if rr == 0:
                    ax.imshow(real_uint8[idx])
                    if c == 0:
                        ax.set_title("Real", fontsize=10)
                else:
                    ax.imshow(recon_uint8[idx])
                    if c == 0:
                        ax.set_title("Recon", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def collect_obs_frames(env, n_frames: int, max_steps_per_episode: int = 200) -> np.ndarray:
    """
    Collect frames directly from env OBS (what AE was trained on).
    Returns (N,H,W,3) float32 in [0,1].
    """
    frames: List[np.ndarray] = []

    reset_out = env.reset()
    obs = _obs_from_reset(reset_out)

    if not isinstance(obs, np.ndarray) or obs.ndim != 3:
        raise RuntimeError(
            "Environment observation is not a 3D image array. "
            f"Got type={type(obs)}, ndim={getattr(obs,'ndim',None)}. "
            "This debug script assumes the AE was trained on pixel observations."
        )

    step = 0
    while len(frames) < n_frames:
        frames.append(obs_to_hwc_float01(obs))

        action = env.action_space.sample()
        step_out = env.step(action)
        obs, done = _step_unpack(step_out)
        step += 1

        if done or step >= max_steps_per_episode:
            reset_out = env.reset()
            obs = _obs_from_reset(reset_out)
            step = 0

    return np.stack(frames, axis=0).astype(np.float32)


# =============================================================================
# Checkpoint finding (env-specific)
# =============================================================================
def find_ae_checkpoint_for_env(env_name: str) -> str:
    """
    Prefer:
      <repo_root>/discrete_mbrl/models/<env_name>/*.pt or *.pth

    Picks best/final/latest if present; else most recent mtime.
    """
    models_root = Path(repo_root) / "discrete_mbrl" / "models"
    env_dir = models_root / env_name

    if not env_dir.exists():
        raise FileNotFoundError(f"Env model directory does not exist: {env_dir}")

    candidates = list(env_dir.glob("*.pt")) + list(env_dir.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found under: {env_dir}")

    def score(p: Path) -> tuple:
        n = p.name.lower()
        bonus = 0
        if "best" in n:
            bonus += 30
        if "final" in n:
            bonus += 20
        if "latest" in n:
            bonus += 15
        if "epoch" in n:
            bonus += 5
        return (bonus, p.stat().st_mtime)

    candidates.sort(key=score, reverse=True)
    return str(candidates[0])


def load_state_dict_best_effort(model: torch.nn.Module, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    # Most robust: find a dict of tensors we can treat as state_dict
    state_dict = None

    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "encoder"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
        if state_dict is None:
            # Might already be a raw state_dict
            if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
                state_dict = ckpt
            else:
                # Search for first dict value that looks like a state_dict
                for v in ckpt.values():
                    if isinstance(v, dict) and any(isinstance(t, torch.Tensor) for t in v.values()):
                        state_dict = v
                        break
    else:
        # non-dict -> could be state_dict itself
        state_dict = ckpt

    if state_dict is None:
        raise RuntimeError(f"Unrecognized checkpoint format: {type(ckpt)}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[debug] load_state_dict(strict=False): missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) and len(missing) <= 20:
        print("  missing keys:", missing)
    if len(unexpected) and len(unexpected) <= 20:
        print("  unexpected keys:", unexpected)


# =============================================================================
# Quantizer diagnostics
# =============================================================================
def try_get_code_indices(encoder: torch.nn.Module, xb: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Try to extract VQ code indices from encoder.
    Different repos implement different APIs.
    Returns LongTensor of shape (B, N_latents) or (B,H,W) etc, or None.
    """
    # Common: encoder.encode(x) returns indices (discrete) for VQVAE
    try:
        z = encoder.encode(xb)
        if isinstance(z, torch.Tensor):
            # In some repos encode returns indices directly (integer type or float)
            if z.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8):
                return z.long()
            # Sometimes encode returns float embeddings; not indices
        elif isinstance(z, (tuple, list)):
            # Sometimes returns (quantized, indices, loss) or similar
            for item in z:
                if isinstance(item, torch.Tensor) and item.dtype in (torch.int64, torch.int32, torch.uint8):
                    return item.long()
    except Exception:
        pass

    # Another pattern: encoder.quantizer(...) returns indices
    if hasattr(encoder, "quantizer"):
        try:
            # If we can get encoder's pre-quant embeddings, try typical path:
            # Some models have encoder.encoder(x) or encoder.encode_continuous(x)
            if hasattr(encoder, "encoder"):
                h = encoder.encoder(xb)
                out = encoder.quantizer(h)
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    idx = out[1]
                    if isinstance(idx, torch.Tensor):
                        return idx.long()
        except Exception:
            pass

    return None


def print_tensor_stats(name: str, t: torch.Tensor):
    t2 = t.detach()
    print(
        f"[debug] {name}: shape={tuple(t2.shape)} dtype={t2.dtype} "
        f"min={float(t2.min()):.6f} max={float(t2.max()):.6f} mean={float(t2.mean()):.6f} "
        f"abs_mean={float(t2.abs().mean()):.6f}"
    )


def print_code_usage(indices: torch.Tensor, codebook_size: Optional[int] = None, topk: int = 10):
    idx = indices.detach().cpu().long().view(-1).numpy()
    uniq, counts = np.unique(idx, return_counts=True)
    print(f"[debug] code usage: unique_codes={len(uniq)} / total_latents={len(idx)}")
    if codebook_size is not None:
        print(f"[debug] codebook_size={codebook_size}, usage_ratio={len(uniq)/float(codebook_size):.4f}")
    # print top-k most frequent codes
    order = np.argsort(-counts)
    print("[debug] top codes:", [(int(uniq[i]), int(counts[i])) for i in order[:topk]])


# =============================================================================
# Main
# =============================================================================
def main():
    parser = make_argparser()

    # Debug-only args
    parser.add_argument("--ckpt", type=str, default="", help="Override checkpoint path (optional)")
    parser.add_argument("--n_frames", type=int, default=32)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", type=str, default="vqvae_recon_debug.png")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    args = process_args(args)

    # Disable external logging for debug
    args.wandb = False
    args.comet_ml = False

    # Keep init_experiment for consistency with repo conventions (hashing/logging),
    # but we do not rely on it for checkpoint discovery.
    args = init_experiment("debug_vqvae_recon", args)

    set_seed(getattr(args, "seed", 0))

    device = getattr(args, "device", "cpu")
    if device != "cpu" and not torch.cuda.is_available():
        print("[debug] CUDA not available -> using cpu")
        device = "cpu"

    # Checkpoint selection
    if args.ckpt.strip():
        ckpt_path = args.ckpt.strip()
        print(f"[debug] Using override ckpt: {ckpt_path}")
    else:
        ckpt_path = find_ae_checkpoint_for_env(args.env_name)
        print(f"[debug] Auto-found AE ckpt: {ckpt_path}")

    # Create env (training wrappers)
    env = make_env(args.env_name)

    reset_out = env.reset()
    obs0 = _obs_from_reset(reset_out)

    print("[debug] obs type:", type(obs0))
    if isinstance(obs0, np.ndarray):
        print("[debug] obs shape:", obs0.shape, "dtype:", obs0.dtype, "min/max:", float(obs0.min()), float(obs0.max()))
    elif isinstance(obs0, dict):
        print("[debug] obs dict keys:", list(obs0.keys()))

    # Collect frames from obs-space (not render)
    frames_hwc = collect_obs_frames(env, n_frames=args.n_frames)
    print("[debug] Collected obs-frames:", frames_hwc.shape, frames_hwc.dtype,
          "min/max:", float(frames_hwc.min()), float(frames_hwc.max()))

    x = to_torch_nchw(frames_hwc, device=device)
    print_tensor_stats("input x", x)

    # Construct AE (same as training)
    obs_shape_for_model = obs0.shape if isinstance(obs0, np.ndarray) else frames_hwc.shape[1:]
    encoder, _ = construct_ae_model(obs_shape_for_model, args)
    encoder = encoder.to(device).eval()

    # Load weights
    load_state_dict_best_effort(encoder, ckpt_path, device=device)

    # Forward AE-only
    recons: List[torch.Tensor] = []
    with torch.no_grad():
        for bi in range(0, x.shape[0], args.batch):
            xb = x[bi:bi + args.batch]

            # Encode
            if hasattr(encoder, "encode"):
                z = encoder.encode(xb)
            else:
                z = None

            # Decode
            if hasattr(encoder, "decode") and z is not None:
                recon = encoder.decode(z if isinstance(z, torch.Tensor) else z[0])
            else:
                out = encoder(xb)
                if isinstance(out, (tuple, list)):
                    recon = out[0]
                elif isinstance(out, dict):
                    recon = out.get("recon", out.get("x_recon", None))
                    if recon is None:
                        raise RuntimeError(f"Encoder dict output keys: {list(out.keys())}")
                else:
                    recon = out

            # Diagnostics
            print_tensor_stats("recon", recon)

            # If recon size mismatch, resize for visualization (nearest to preserve grid)
            if recon.shape[-2:] != xb.shape[-2:]:
                print(f"[WARN] recon size != input size: input={tuple(xb.shape)} recon={tuple(recon.shape)} -> resizing for viz")
                recon = F.interpolate(recon, size=xb.shape[-2:], mode="nearest")

            # Quantizer diagnostics (best-effort)
            indices = try_get_code_indices(encoder, xb)
            if indices is not None:
                # Determine codebook size if possible
                cb_size = None
                if hasattr(encoder, "n_embeddings"):
                    cb_size = int(getattr(encoder, "n_embeddings"))
                elif hasattr(encoder, "quantizer") and hasattr(encoder.quantizer, "embedding"):
                    cb_size = int(encoder.quantizer.embedding.weight.shape[0])

                print_tensor_stats("code_indices", indices.float())
                print_code_usage(indices, codebook_size=cb_size, topk=10)

                # Codebook stats if available
                if hasattr(encoder, "quantizer") and hasattr(encoder.quantizer, "embedding"):
                    emb = encoder.quantizer.embedding.weight
                    print_tensor_stats("codebook_embedding", emb)

            recons.append(recon)

    recon_all = torch.cat(recons, dim=0)

    # Save grid
    real_uint8 = tensor_to_uint8_hwc(x)
    recon_uint8 = tensor_to_uint8_hwc(recon_all)

    save_grid(
        real_uint8,
        recon_uint8,
        args.out,
        n_cols=8,
        title=f"{args.env_name} | {args.ae_model_type} | AE-only Recon",
    )
    print(f"Saved: {args.out}")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
