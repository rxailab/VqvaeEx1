#!/usr/bin/env python3
"""
Codebook Spatial Maps (Research Direction 1 — A–C)

Loads latent indices saved by codebook_semantics.py and performs:
A) Code Frequency vs Spatial Entropy scatter
B) Simple semantic labeling via count/entropy thresholds
C) KMeans-like clustering on spatial centers (pure numpy)

Input (expected):
  analysis/codebook_semantics/latent_indices.npz
    - indices: (N, H_lat, W_lat) int
    - (optional) H_lat, W_lat, K

Outputs:
  analysis/codebook_spatial_maps/codebook_spatial_stats.csv
  analysis/codebook_spatial_maps/analysis_spatial/*.png
"""

import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

# repo imports
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training_helpers import make_argparser, process_args
from data_logging import init_experiment


# -------------------------
# helpers
# -------------------------
def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def spatial_entropy(p_2d: np.ndarray, eps: float = 1e-12) -> float:
    """
    p_2d: nonnegative array, will be normalized to sum=1 (if sum>0)
    returns: entropy in nats
    """
    s = float(p_2d.sum())
    if s <= 0:
        return 0.0
    p = p_2d / s
    p = np.clip(p, eps, 1.0)
    return float(-(p * np.log(p)).sum())


def center_of_mass(counts_2d: np.ndarray):
    """
    counts_2d: (H,W) nonnegative
    returns (cx, cy) in grid coordinates (x=col, y=row)
    """
    total = float(counts_2d.sum())
    if total <= 0:
        return (np.nan, np.nan)

    H, W = counts_2d.shape
    ys = np.arange(H, dtype=np.float64)[:, None]
    xs = np.arange(W, dtype=np.float64)[None, :]

    cy = float((counts_2d * ys).sum() / total)
    cx = float((counts_2d * xs).sum() / total)
    return (cx, cy)


def simple_kmeans(X: np.ndarray, k: int = 5, n_iter: int = 50, seed: int = 0):
    """
    Minimal k-means (L2) in numpy.
    X: (N,D)
    returns labels (N,), centers (k,D)
    """
    rng = np.random.default_rng(seed)
    N, D = X.shape
    if N == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((k, D), dtype=np.float64)

    # init: pick random unique points
    idx = rng.choice(N, size=min(k, N), replace=False)
    centers = X[idx].copy()

    # if N < k, pad by repeating last
    if centers.shape[0] < k:
        pad = np.repeat(centers[-1][None, :], k - centers.shape[0], axis=0)
        centers = np.concatenate([centers, pad], axis=0)

    labels = np.zeros((N,), dtype=np.int64)

    for _ in range(n_iter):
        # assign
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # (N,k)
        new_labels = d2.argmin(axis=1)

        if np.all(new_labels == labels):
            break
        labels = new_labels

        # update
        for j in range(k):
            mask = (labels == j)
            if mask.any():
                centers[j] = X[mask].mean(axis=0)
            else:
                # re-seed empty cluster
                centers[j] = X[rng.integers(0, N)]

    return labels, centers


def label_code(count: int,
               ent: float,
               dead_count_thresh: int,
               static_entropy_thresh: float,
               dynamic_entropy_thresh: float) -> str:
    """
    Simple heuristic labels (you can refine later):
      - dead: count < dead_count_thresh
      - static-structure: entropy <= static_entropy_thresh
      - dynamic-agent: entropy >= dynamic_entropy_thresh
      - semi-static: otherwise
    """
    if count < dead_count_thresh:
        return "dead"
    if ent <= static_entropy_thresh:
        return "static-structure"
    if ent >= dynamic_entropy_thresh:
        return "dynamic-agent"
    return "semi-static"


# -------------------------
# main
# -------------------------
def analyze_spatial(args):
    # paths
    sem_dir = os.path.join("analysis", "codebook_semantics")
    in_npz = os.path.join(sem_dir, "latent_indices.npz")

    out_root = os.path.join("analysis", "codebook_spatial_maps")
    out_plot = os.path.join(out_root, "analysis_spatial")
    _safe_mkdir(out_root)
    _safe_mkdir(out_plot)

    if not os.path.exists(in_npz):
        raise FileNotFoundError(
            f"Missing {in_npz}. "
            f"Run codebook_semantics.py with latent index saving enabled."
        )

    data = np.load(in_npz, allow_pickle=False)
    indices = data["indices"]  # (N,H,W)
    if indices.ndim != 3:
        raise ValueError(f"Expected indices shape (N,H,W), got {indices.shape}")

    N, H_lat, W_lat = indices.shape

    # infer K (either stored or from max+1)
    if "K" in data.files:
        K = int(data["K"])
    else:
        K = int(indices.max()) + 1

    # thresholds (tunable)
    dead_count_thresh = int(getattr(args, "dead_count_thresh", 50))
    static_entropy_thresh = float(getattr(args, "static_entropy_thresh", 0.5))
    dynamic_entropy_thresh = float(getattr(args, "dynamic_entropy_thresh", 2.0))

    print("\n=== Codebook Spatial Stats Analysis (A–C) ===")
    print(f"npz_path: {in_npz}")
    print(f"N={N} latent=(H={H_lat}, W={W_lat}) inferred K={K}")

    # ------------------------------------------------------------
    # Build per-code spatial histograms: pos_counts[c, y, x]
    # ------------------------------------------------------------
    pos_counts = np.zeros((K, H_lat, W_lat), dtype=np.int64)
    counts = np.zeros((K,), dtype=np.int64)

    # fast counting
    # flatten grid positions
    flat = indices.reshape(N, H_lat * W_lat)
    for n in range(N):
        row = flat[n]
        # histogram codes in this obs
        # update total counts
        binc = np.bincount(row, minlength=K)
        counts += binc

        # update position counts (loop over positions, but H*W is small e.g. 81)
        for p, c in enumerate(row):
            y = p // W_lat
            x = p % W_lat
            pos_counts[c, y, x] += 1

    used_mask = counts > 0
    used_codes = int(used_mask.sum())
    dead_codes_zero = int((counts == 0).sum())

    print(f"total_codes: {K}")
    print(f"used_codes (count>0): {used_codes}")
    print(f"dead_codes (count==0): {dead_codes_zero}")

    # ------------------------------------------------------------
    # Compute entropy + centers per code
    # ------------------------------------------------------------
    ent = np.zeros((K,), dtype=np.float64)
    cx = np.full((K,), np.nan, dtype=np.float64)
    cy = np.full((K,), np.nan, dtype=np.float64)

    for c in range(K):
        if counts[c] <= 0:
            ent[c] = 0.0
            continue
        ent[c] = spatial_entropy(pos_counts[c].astype(np.float64))
        cx[c], cy[c] = center_of_mass(pos_counts[c].astype(np.float64))

    # ------------------------------------------------------------
    # A) Frequency vs Spatial Entropy
    # ------------------------------------------------------------
    # Use only codes that appear at least once (avoid log(0))
    xs = counts[used_mask].astype(np.float64)
    ys = ent[used_mask].astype(np.float64)

    plt.figure()
    plt.scatter(xs, ys)
    plt.xscale("log")
    plt.xlabel("Count (log scale)")
    plt.ylabel("Spatial Entropy")
    plt.title("Code Frequency vs Spatial Entropy")
    a_path = os.path.join(out_plot, "entropy_vs_count.png")
    plt.savefig(a_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------
    # B) Label codes by heuristic thresholds
    # ------------------------------------------------------------
    labels = np.array([
        label_code(int(counts[c]), float(ent[c]),
                   dead_count_thresh, static_entropy_thresh, dynamic_entropy_thresh)
        for c in range(K)
    ], dtype=object)

    # label counts (note: "dead" here means *rare*, not necessarily count==0)
    uniq, cnts = np.unique(labels, return_counts=True)
    label_hist = dict(zip(uniq.tolist(), cnts.tolist()))

    print("\nB) Label thresholds:")
    print(f"  dead_count_thresh: {dead_count_thresh}")
    print(f"  static_entropy_thresh: {static_entropy_thresh}")
    print(f"  dynamic_entropy_thresh: {dynamic_entropy_thresh}")

    print("\nB) Label counts:")
    for k in sorted(label_hist.keys()):
        print(f"  {k}: {label_hist[k]}")

    # plot centers colored by label (only codes with finite centers)
    finite = np.isfinite(cx) & np.isfinite(cy)
    Xc = cx[finite]
    Yc = cy[finite]
    Lc = labels[finite]

    # color mapping (fixed names)
    color_map = {
        "dead": "C0",
        "static-structure": "C1",
        "semi-static": "C2",
        "dynamic-agent": "C3",
    }

    plt.figure()
    for lab in ["dead", "static-structure", "semi-static", "dynamic-agent"]:
        m = (Lc == lab)
        if m.any():
            plt.scatter(Xc[m], Yc[m], label=lab)
    plt.gca().invert_yaxis()
    plt.xlabel("center_x (latent col)")
    plt.ylabel("center_y (latent row)")
    plt.title("Code Spatial Centers (colored by semantic label)")
    plt.legend()
    b_path = os.path.join(out_plot, "center_scatter_by_label.png")
    plt.savefig(b_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------
    # C) Cluster centers (k-means) to discover position groups
    # ------------------------------------------------------------
    # cluster only non-dead-by-zero? (use finite centers + count>0)
    X = np.stack([Xc, Yc], axis=1).astype(np.float64)
    k = int(getattr(args, "kmeans_k", 5))
    km_labels, km_centers = simple_kmeans(X, k=k, n_iter=100, seed=0)

    plt.figure()
    for j in range(k):
        m = (km_labels == j)
        if m.any():
            plt.scatter(X[m, 0], X[m, 1], label=f"cluster {j}")
    plt.scatter(km_centers[:, 0], km_centers[:, 1], marker="x", s=200, label="centers")
    plt.gca().invert_yaxis()
    plt.xlabel("center_x (latent col)")
    plt.ylabel("center_y (latent row)")
    plt.title(f"KMeans on Code Centers (k={k})")
    plt.legend()
    c_path = os.path.join(out_plot, "center_scatter_by_cluster.png")
    plt.savefig(c_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------
    # Save CSV (spatial stats per code)
    # ------------------------------------------------------------
    csv_path = os.path.join(out_root, "codebook_spatial_stats.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "code", "count", "spatial_entropy",
            "center_x", "center_y",
            "label",
        ])
        for c in range(K):
            w.writerow([
                int(c),
                int(counts[c]),
                float(ent[c]),
                float(cx[c]) if np.isfinite(cx[c]) else "",
                float(cy[c]) if np.isfinite(cy[c]) else "",
                str(labels[c]),
            ])

    print("\n[done] A–C plots:")
    print(f"  {a_path}")
    print(f"  {b_path}")
    print(f"  {c_path}")
    print(f"[done] csv: {csv_path}")


def main():
    parser = make_argparser()

    # spatial-maps specific knobs (A–C)
    parser.add_argument("--dead_count_thresh", type=int, default=50)
    parser.add_argument("--static_entropy_thresh", type=float, default=0.5)
    parser.add_argument("--dynamic_entropy_thresh", type=float, default=2.0)
    parser.add_argument("--kmeans_k", type=int, default=5)

    args = parser.parse_args()
    args = process_args(args)

    # disable cloud logging
    args.wandb = False
    args.comet_ml = False
    args = init_experiment("codebook_spatial_maps", args)

    analyze_spatial(args)


if __name__ == "__main__":
    main()
