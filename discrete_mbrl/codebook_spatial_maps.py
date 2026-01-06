#!/usr/bin/env python3
"""
Codebook Spatial Maps (Research Direction 1)

Consumes saved latent index grids from:
  analysis/codebook_semantics/latent_codes.npy

For each code:
- spatial frequency heatmap (H_lat x W_lat)
- normalized probability map
- spatial entropy
- center-of-mass
- summary CSV

NO model loading. NO encoder. Pure analysis.

Outputs:
  analysis/codebook_spatial_maps/
"""

import os
import csv
import math
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt


# -------------------------
# utils
# -------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_heatmap(img, path, scale=32):
    """
    img: (H,W) float in [0,1]
    """
    img_u8 = (img * 255.0).clip(0, 255).astype(np.uint8)
    img_rgb = np.stack([img_u8] * 3, axis=-1)
    pil = Image.fromarray(img_rgb)
    pil = pil.resize(
        (img_rgb.shape[1] * scale, img_rgb.shape[0] * scale),
        resample=Image.NEAREST,
    )
    pil.save(path)


def spatial_entropy(p):
    """
    p: normalized probability map (H,W), sum=1
    """
    eps = 1e-12
    p = p[p > eps]
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def center_of_mass(p):
    """
    p: normalized probability map (H,W)
    returns (y, x)
    """
    H, W = p.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    mass = p.sum()
    if mass <= 0:
        return (np.nan, np.nan)
    y = float((ys * p).sum() / mass)
    x = float((xs * p).sum() / mass)
    return y, x


# -------------------------
# main
# -------------------------
def main():
    in_dir = os.path.join("analysis", "codebook_semantics")
    out_dir = os.path.join("analysis", "codebook_spatial_maps")
    ensure_dir(out_dir)

    latent_path = os.path.join(in_dir, "latent_codes.npy")
    meta_path = os.path.join(in_dir, "latent_codes_meta.csv")

    assert os.path.exists(latent_path), f"Missing {latent_path}"

    print(f"[info] loading latent codes from {latent_path}")
    latents = np.load(latent_path)  # (N, H, W)
    N, H_lat, W_lat = latents.shape

    print(f"[info] latent_codes shape = {latents.shape}")

    # infer K
    K = int(latents.max()) + 1
    print(f"[info] inferred codebook size K={K}")

    # storage
    counts = np.zeros(K, dtype=np.int64)
    spatial_counts = np.zeros((K, H_lat, W_lat), dtype=np.int64)

    # accumulate
    print("[info] accumulating spatial statistics...")
    for n in tqdm(range(N), ncols=100):
        grid = latents[n]
        for i in range(H_lat):
            for j in range(W_lat):
                c = int(grid[i, j])
                if c < 0 or c >= K:
                    continue
                counts[c] += 1
                spatial_counts[c, i, j] += 1

    used_codes = int((counts > 0).sum())
    print(f"[summary] used_codes = {used_codes}/{K}")

    # CSV summary
    csv_path = os.path.join(out_dir, "codebook_spatial_stats.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "code",
            "count",
            "entropy",
            "center_y",
            "center_x",
        ])

        # render per-code heatmaps
        for c in tqdm(range(K), desc="Rendering maps", ncols=100):
            if counts[c] == 0:
                continue

            heat = spatial_counts[c].astype(np.float32)
            prob = heat / heat.sum()

            ent = spatial_entropy(prob)
            cy, cx = center_of_mass(prob)

            w.writerow([
                c,
                int(counts[c]),
                float(ent),
                float(cy),
                float(cx),
            ])

            # save heatmaps
            save_heatmap(
                heat / heat.max(),
                os.path.join(out_dir, f"heatmap_code{c:04d}.png"),
            )
            save_heatmap(
                prob,
                os.path.join(out_dir, f"prob_code{c:04d}.png"),
            )

    print(f"[done] spatial maps saved to {out_dir}")
    print(f"[done] csv: {csv_path}")

    # -------- summary plots --------
    print("[info] generating summary plots...")

    entropies = []
    for c in range(K):
        if counts[c] == 0:
            continue
        p = spatial_counts[c].astype(np.float32)
        p /= p.sum()
        entropies.append(spatial_entropy(p))

    plt.figure(figsize=(6, 4))
    plt.hist(entropies, bins=50)
    plt.title("Spatial Entropy of Codes")
    plt.xlabel("Entropy")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "entropy_histogram.png"))
    plt.close()

    print("[done] summary plots written")


if __name__ == "__main__":
    main()
