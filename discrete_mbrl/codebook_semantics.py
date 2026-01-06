#!/usr/bin/env python3
"""
Codebook Semantics (Research Direction 1): Code -> pixel/patch meaning

For each VQ code c:
- frequency (coverage over latent positions)
- mean RGB over associated image patches
- mean patch visualization (average of aligned patches)
- example patches (contact sheet)
- spatial heatmap over latent grid (where code appears)

NEW (requested):
- Save per-observation latent index grids to disk:
    analysis/codebook_semantics/latent_codes.npy  (shape: [N, H_lat, W_lat], dtype uint16/int32)
  This lets you run downstream scripts (e.g., codebook_spatial_maps.py) without re-encoding.

Repo-native:
- uses make_argparser / process_args
- uses prepare_dataloader from your repo
- loads checkpoint via --ckpt (already in your arg parser)

Outputs saved under: analysis/codebook_semantics/
"""

import os
import math
import csv
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

# repo imports
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training_helpers import make_argparser, process_args
from model_construction import construct_ae_model
from data_helpers import prepare_dataloader
from data_logging import init_experiment


# -------------------------
# utils: robust index parsing
# -------------------------
def _extract_indices_from_encode_output(enc):
    """
    Try to extract code indices tensor from encoder.encode(...) output.
    Supports:
      - LongTensor directly
      - tuple/list where one element is LongTensor indices
    Returns: tensor or None
    """
    if torch.is_tensor(enc):
        return enc
    if isinstance(enc, (tuple, list)):
        # prefer int/long tensors
        for t in enc:
            if torch.is_tensor(t) and t.dtype in (torch.int64, torch.long):
                return t
        # fallback: first tensor
        for t in enc:
            if torch.is_tensor(t):
                return t
    return None


def _indices_to_grid(indices, latent_h=None, latent_w=None):
    """
    Convert indices to (B, H_lat, W_lat).
    Accepts indices shaped:
      - (B, H, W)
      - (B, L)
    If (B,L), tries to infer square or use provided latent_h/latent_w.
    """
    assert torch.is_tensor(indices), "indices must be a tensor"
    if indices.ndim == 3:
        return indices
    if indices.ndim != 2:
        raise ValueError(f"Unsupported indices shape {tuple(indices.shape)}")

    B, L = indices.shape
    if latent_h is not None and latent_w is not None and latent_h * latent_w == L:
        return indices.view(B, latent_h, latent_w)

    s = int(round(math.sqrt(L)))
    if s * s == L:
        return indices.view(B, s, s)

    # common fallback: 64->8x8, 81->9x9, 256->16x16
    if L == 64:
        return indices.view(B, 8, 8)
    if L == 81:
        return indices.view(B, 9, 9)
    if L == 256:
        return indices.view(B, 16, 16)

    raise ValueError(f"Cannot reshape flat indices of length {L} to a grid.")


def _to_uint8_hwc(x_chw_01):
    """(3,H,W) float in [0,1] -> uint8 (H,W,3)"""
    x = x_chw_01.detach().cpu().clamp(0, 1).numpy()
    x = (x.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return x


def _save_image_uint8(img_u8_hwc, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img_u8_hwc).save(path)


def _make_contact_sheet(patches_u8, ncols=8, pad=2, pad_val=0):
    """
    patches_u8: list of (H,W,3) uint8
    """
    if len(patches_u8) == 0:
        return None
    ph, pw, _ = patches_u8[0].shape
    n = len(patches_u8)
    ncols = min(ncols, n)
    nrows = int(math.ceil(n / ncols))

    H = nrows * ph + (nrows + 1) * pad
    W = ncols * pw + (ncols + 1) * pad
    canvas = np.full((H, W, 3), pad_val, dtype=np.uint8)

    for i, p in enumerate(patches_u8):
        r = i // ncols
        c = i % ncols
        y0 = pad + r * (ph + pad)
        x0 = pad + c * (pw + pad)
        canvas[y0:y0 + ph, x0:x0 + pw] = p
    return canvas


def _resize_nearest(img_u8_hwc, scale=8):
    """Nearest-neighbor upscale for tiny patches."""
    if scale == 1:
        return img_u8_hwc
    pil = Image.fromarray(img_u8_hwc)
    pil = pil.resize((img_u8_hwc.shape[1] * scale, img_u8_hwc.shape[0] * scale), resample=Image.NEAREST)
    return np.array(pil)


# -------------------------
# main analysis
# -------------------------
@torch.no_grad()
def analyze_codebook(args):
    # output dirs
    out_dir = os.path.join("analysis", "codebook_semantics")
    os.makedirs(out_dir, exist_ok=True)

    # dataloader
    test_loader = prepare_dataloader(
        args.env_name, "test",
        batch_size=args.batch_size,
        preprocess=args.preprocess,
        randomize=True,
        n=args.n_samples,
        n_preload=0,
        preload=False,
        extra_buffer_keys=getattr(args, "extra_buffer_keys", None),
    )

    # infer obs shape
    sample_batch = next(iter(test_loader))
    obs = sample_batch[0]  # (B,C,H,W) presumably
    if isinstance(obs, (list, tuple)):
        obs = obs[0]
    obs_shape = tuple(obs.shape[1:])  # (C,H,W)
    print(f"[info] obs={obs_shape}")

    # construct model
    encoder = construct_ae_model(obs_shape, args)[0]
    encoder = encoder.to(args.device).eval()

    # optional load ckpt (repo arg name is --ckpt)
    ckpt_path = getattr(args, "ckpt", None)
    if ckpt_path:
        print(f"[info] loading encoder checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        print(f"[info] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    # get codebook size
    K = getattr(args, "codebook_size", None)
    if hasattr(encoder, "n_embeddings"):
        K = int(encoder.n_embeddings)
    if K is None:
        raise RuntimeError("Could not infer codebook size (K). Make sure --codebook_size is set or encoder has n_embeddings.")
    print(f"[info] codes={K}")

    # --- First pass: infer latent grid size and patch size ---
    obs0 = sample_batch[0].to(args.device)[:1]  # (1,C,H,W)
    enc0 = encoder.encode(obs0)
    idx0 = _extract_indices_from_encode_output(enc0)
    if idx0 is None:
        raise RuntimeError("encoder.encode(...) did not return indices (LongTensor). This script is for VQ encoders.")
    grid0 = _indices_to_grid(idx0)  # (1,H_lat,W_lat)
    H_lat, W_lat = int(grid0.shape[1]), int(grid0.shape[2])
    C, H, W = obs_shape
    patch_h = H // H_lat
    patch_w = W // W_lat
    if patch_h <= 0 or patch_w <= 0:
        raise RuntimeError(f"Bad patch sizes inferred: patch_h={patch_h}, patch_w={patch_w}")
    print(f"[info] latent=(h={H_lat}, w={W_lat}) patch=(h={patch_h}, w={patch_w})")

    # NEW: allocate latent index storage
    target = int(args.n_samples)
    save_latents = bool(getattr(args, "save_latents", True))
    latents_dtype = getattr(args, "latent_indices_dtype", "uint16")
    if latents_dtype not in ("uint16", "int32"):
        raise ValueError("--latent_indices_dtype must be uint16 or int32")

    latent_codes = None
    if save_latents:
        # uint16 is enough for K<=65535 (your K=512)
        dtype = np.uint16 if latents_dtype == "uint16" else np.int32
        latent_codes = np.zeros((target, H_lat, W_lat), dtype=dtype)
        print(f"[info] will save latent indices: shape={(target, H_lat, W_lat)} dtype={latent_codes.dtype}")

    # storage
    counts = np.zeros((K,), dtype=np.int64)                   # total occurrences
    pos_counts = np.zeros((K, H_lat, W_lat), dtype=np.int64)  # spatial heatmaps
    sum_rgb = np.zeros((K, 3), dtype=np.float64)              # sum of patch mean RGB (for mean color)
    sum_patch = np.zeros((K, 3, patch_h, patch_w), dtype=np.float64)  # mean patch accumulation
    patch_counts = np.zeros((K,), dtype=np.int64)

    # reservoir sampling for examples
    max_examples = int(getattr(args, "examples_per_code", 32))
    rng = np.random.default_rng(0)
    examples = {c: [] for c in range(K)}     # list of uint8 patches
    seen_for_examples = np.zeros((K,), dtype=np.int64)

    print(f"[info] target samples={target} batch_size={args.batch_size}")
    print(f"[info] saving outputs to: {out_dir}")

    # iteration
    pbar = tqdm(total=target, desc="Collecting samples", unit="obs", ncols=120)
    collected = 0

    for batch in test_loader:
        obs_b = batch[0].to(args.device)  # (B,C,H,W)
        B = obs_b.shape[0]

        # encode to indices grid
        enc = encoder.encode(obs_b)
        idx = _extract_indices_from_encode_output(enc)
        if idx is None:
            raise RuntimeError("encoder.encode(...) did not yield indices in this batch.")
        grid = _indices_to_grid(idx, latent_h=H_lat, latent_w=W_lat)  # (B,H_lat,W_lat)
        grid_np = grid.detach().cpu().numpy().astype(np.int64)  # safe for counting / slicing

        obs_cpu = obs_b.detach().cpu()  # keep as torch for slicing

        # For each sample in batch
        for b in range(B):
            if collected >= target:
                break

            # NEW: store latent indices for this observation
            if save_latents and latent_codes is not None:
                # clip to valid range and cast
                if latent_codes.dtype == np.uint16:
                    latent_codes[collected] = np.clip(grid_np[b], 0, 65535).astype(np.uint16)
                else:
                    latent_codes[collected] = grid_np[b].astype(np.int32)

            # for each latent cell
            for i in range(H_lat):
                y0 = i * patch_h
                y1 = (i + 1) * patch_h
                for j in range(W_lat):
                    x0 = j * patch_w
                    x1 = (j + 1) * patch_w

                    c = int(grid_np[b, i, j])
                    if c < 0 or c >= K:
                        continue

                    patch = obs_cpu[b, :, y0:y1, x0:x1].clamp(0, 1)  # (3,ph,pw)

                    # count + spatial
                    counts[c] += 1
                    pos_counts[c, i, j] += 1

                    # patch stats
                    sum_patch[c] += patch.numpy()
                    patch_counts[c] += 1

                    # mean rgb of patch
                    rgb = patch.mean(dim=(1, 2)).numpy()
                    sum_rgb[c] += rgb

                    # reservoir for examples
                    seen_for_examples[c] += 1
                    if len(examples[c]) < max_examples:
                        examples[c].append(_to_uint8_hwc(patch))
                    else:
                        # standard reservoir sampling
                        t = int(seen_for_examples[c])
                        r = rng.integers(0, t)
                        if r < max_examples:
                            examples[c][r] = _to_uint8_hwc(patch)

            collected += 1
            pbar.update(1)

        if collected >= target:
            break

    pbar.close()

    # NEW: save latent indices
    if save_latents and latent_codes is not None:
        latent_path = os.path.join(out_dir, "latent_codes.npy")
        np.save(latent_path, latent_codes)
        print(f"[done] latent indices saved: {latent_path} (shape={latent_codes.shape}, dtype={latent_codes.dtype})")

        # Optional metadata (handy for downstream scripts)
        meta_path = os.path.join(out_dir, "latent_codes_meta.csv")
        with open(meta_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["N", "H_lat", "W_lat", "K", "patch_h", "patch_w"])
            w.writerow([int(target), int(H_lat), int(W_lat), int(K), int(patch_h), int(patch_w)])
        print(f"[done] latent meta saved: {meta_path}")

    used_codes = int((counts > 0).sum())
    dead_codes = K - used_codes
    print(f"[summary] used_codes = {used_codes}/{K} | dead_codes = {dead_codes}")

    # compute stats
    total_occ = int(counts.sum())
    freq = counts / max(total_occ, 1)

    mean_rgb = np.zeros((K, 3), dtype=np.float64)
    for c in range(K):
        if counts[c] > 0:
            mean_rgb[c] = sum_rgb[c] / counts[c]

    # save CSV
    csv_path = os.path.join(out_dir, "codebook_stats.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "count", "freq", "mean_r", "mean_g", "mean_b"])
        for c in range(K):
            w.writerow([c, int(counts[c]), float(freq[c]), float(mean_rgb[c, 0]), float(mean_rgb[c, 1]), float(mean_rgb[c, 2])])
    print(f"[done] csv: {csv_path}")

    # save per-code visuals
    # choose codes to render: all used, or top-N by count
    top_n = int(getattr(args, "top_n", 128))
    order = np.argsort(-counts)
    render_codes = [int(c) for c in order[:top_n] if counts[int(c)] > 0]

    # summary canvas: top 32 mean patches (for quick glance)
    summary_patches = []

    print("[info] saving mean patches / examples / heatmaps ...")
    for c in tqdm(render_codes, desc="Rendering codes", unit="code", ncols=120):
        if patch_counts[c] <= 0:
            continue

        mean_patch = (sum_patch[c] / patch_counts[c]).astype(np.float32)  # (3,ph,pw)
        mean_patch_t = torch.from_numpy(mean_patch).clamp(0, 1)
        mean_patch_u8 = _to_uint8_hwc(mean_patch_t)

        # save mean patch (upscaled)
        mean_patch_big = _resize_nearest(mean_patch_u8, scale=8)
        _save_image_uint8(mean_patch_big, os.path.join(out_dir, f"mean_patch_code{c:04d}.png"))

        # save contact sheet of examples
        ex = examples[c]
        if len(ex) > 0:
            ex_big = [_resize_nearest(p, scale=8) for p in ex]
            sheet = _make_contact_sheet(ex_big, ncols=8, pad=4, pad_val=10)
            if sheet is not None:
                _save_image_uint8(sheet, os.path.join(out_dir, f"examples_code{c:04d}.png"))

        # save heatmap (latent grid counts normalized)
        heat = pos_counts[c].astype(np.float32)
        if heat.max() > 0:
            heat = heat / heat.max()
        # convert to grayscale uint8 and upscale
        heat_u8 = (heat * 255.0).astype(np.uint8)
        heat_rgb = np.stack([heat_u8] * 3, axis=-1)  # (H_lat,W_lat,3)
        heat_big = _resize_nearest(heat_rgb, scale=32)
        _save_image_uint8(heat_big, os.path.join(out_dir, f"heatmap_code{c:04d}.png"))

        if len(summary_patches) < 32:
            summary_patches.append(mean_patch_big)

    # write summary image
    if len(summary_patches) > 0:
        summary = _make_contact_sheet(summary_patches, ncols=8, pad=6, pad_val=20)
        if summary is not None:
            _save_image_uint8(summary, os.path.join(out_dir, "summary.png"))

    print(f"[done] results saved to {out_dir}")


def main():
    parser = make_argparser()

    # semantics-specific args
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of observations to analyze")
    parser.add_argument("--top_n", type=int, default=128, help="Render top-N most frequent codes")
    parser.add_argument("--examples_per_code", type=int, default=32, help="Example patches stored per code")

    # NEW: latent index saving
    parser.add_argument(
        "--save_latents",
        action="store_true",
        help="Save per-observation latent index grids to analysis/codebook_semantics/latent_codes.npy",
    )
    parser.add_argument(
        "--no_save_latents",
        action="store_true",
        help="Disable saving latent index grids (overrides --save_latents).",
    )
    parser.add_argument(
        "--latent_indices_dtype",
        type=str,
        default="uint16",
        choices=["uint16", "int32"],
        help="Datatype for saved latent indices (uint16 recommended for K<=65535).",
    )

    args = parser.parse_args()
    args = process_args(args)

    # default behavior: SAVE latents unless explicitly disabled
    if getattr(args, "no_save_latents", False):
        args.save_latents = False
    else:
        # if user didn't pass --save_latents, we still default to saving
        args.save_latents = True

    # disable cloud logging by default
    args.wandb = False
    args.comet_ml = False
    args = init_experiment("codebook_semantics", args)

    analyze_codebook(args)


if __name__ == "__main__":
    main()
