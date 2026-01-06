#!/usr/bin/env python3
"""
Codebook Analyses (D/E/F) for VQ world models (repo-native)

Implements:
D) Code -> agent overlap score
   - For each latent patch assigned to code c, compute fraction of "agent pixels"
   - Outputs: CSV + histogram + top-K ranked codes by overlap

E) Code transition entropy
   - Uses (obs, next_obs) from your replay/dataloader (buffer-based)
   - For each code c, estimate P(next_code | c) at the SAME latent position
   - Outputs: CSV + scatter plots + summary stats
   - Also reports change rate: P(next != curr | c)

F) Ablation by code groups
   - Loads your code labels from analysis/codebook_spatial_maps/codebook_spatial_stats.csv
   - For each label group, replace those codes with a baseline code (default: most frequent)
   - Decode and save montages:
       - original recon
       - ablated recon (static removed, dynamic removed, etc.)

Repo-native:
- uses make_argparser / process_args
- uses construct_ae_model
- uses prepare_dataloader

Typical run (one line):
python codebook_analyses_def.py --env_name MiniGrid-SimpleCrossingS9N1-v0 --ae_model_type vqvae --trans_model_type discrete --codebook_size 512 --embedding_dim 128 --latent_dim 128 --filter_size 9 --device cuda --preprocess --batch_size 64 --n_samples 5000 --spatial_csv analysis/codebook_spatial_maps/codebook_spatial_stats.csv --out_dir analysis/codebook_semantics/analysis_def

Notes:
- D uses a simple "red triangle" threshold to detect agent pixels.
  If your agent color differs, adjust thresholds in make_agent_mask().
- E requires next_obs in the dataloader batch. If your loader doesn't return it,
  consider adding `extra_buffer_keys` or switching split to one that includes transitions.
"""

import os
import csv
import math
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# repo imports
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training_helpers import make_argparser, process_args
from model_construction import construct_ae_model
from data_helpers import prepare_dataloader
from data_logging import init_experiment


# -------------------------
# small utilities
# -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def _try_add_arg(parser, *args, **kwargs):
    """Safely add an argument if it doesn't already exist in shared parser."""
    try:
        parser.add_argument(*args, **kwargs)
    except Exception:
        pass


def save_uint8(img_u8_hwc, path):
    ensure_dir(os.path.dirname(path))
    Image.fromarray(img_u8_hwc).save(path)


def to_uint8_hwc(x_chw01: torch.Tensor):
    """(3,H,W) float [0,1] -> uint8 (H,W,3)"""
    x = x_chw01.detach().cpu().clamp(0, 1).numpy()
    x = (x.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return x


def resize_nearest_u8(img_u8_hwc, scale: int):
    if scale == 1:
        return img_u8_hwc
    pil = Image.fromarray(img_u8_hwc)
    pil = pil.resize((img_u8_hwc.shape[1] * scale, img_u8_hwc.shape[0] * scale), resample=Image.NEAREST)
    return np.array(pil)


def contact_sheet(imgs_u8, ncols=8, pad=6, pad_val=255):
    """imgs_u8: list of (H,W,3) uint8"""
    if len(imgs_u8) == 0:
        return None
    h, w, _ = imgs_u8[0].shape
    n = len(imgs_u8)
    ncols = min(ncols, n)
    nrows = int(math.ceil(n / ncols))
    H = nrows * h + (nrows + 1) * pad
    W = ncols * w + (ncols + 1) * pad
    canvas = np.full((H, W, 3), pad_val, dtype=np.uint8)
    for i, im in enumerate(imgs_u8):
        r = i // ncols
        c = i % ncols
        y0 = pad + r * (h + pad)
        x0 = pad + c * (w + pad)
        canvas[y0:y0 + h, x0:x0 + w] = im
    return canvas


def extract_indices(enc_out):
    """
    Extract LongTensor indices from encoder.encode output.
    Supports tensor or tuple/list containing a LongTensor.
    """
    if torch.is_tensor(enc_out):
        return enc_out
    if isinstance(enc_out, (tuple, list)):
        for t in enc_out:
            if torch.is_tensor(t) and t.dtype in (torch.long, torch.int64):
                return t
        for t in enc_out:
            if torch.is_tensor(t):
                return t
    return None


def indices_to_grid(indices: torch.Tensor, latent_h=None, latent_w=None):
    """
    Convert indices to (B,H_lat,W_lat). Accepts:
      - (B,H,W)
      - (B,L)
    """
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
    if L == 64:
        return indices.view(B, 8, 8)
    if L == 81:
        return indices.view(B, 9, 9)
    if L == 256:
        return indices.view(B, 16, 16)

    raise ValueError(f"Cannot reshape flat indices length {L} into grid.")


def grid_to_flat(grid_bhw: torch.Tensor):
    """(B,H,W) -> (B,H*W)"""
    return grid_bhw.view(grid_bhw.shape[0], -1)


def decode_from_indices_flat(encoder, indices_flat_bl: torch.Tensor):
    """
    Decode from indices. Most repo VQ decoders expect (B, n_latents) long.
    Returns recon (B,C,H,W)
    """
    return encoder.decode(indices_flat_bl.long())


# -------------------------
# D) Agent mask & overlap
# -------------------------
def make_agent_mask(obs_bchw01: torch.Tensor):
    """
    Heuristic agent detector for MiniGrid: red triangle.
    obs in (B,3,H,W), float in [0,1].

    Adjust thresholds if needed.
    """
    r = obs_bchw01[:, 0:1]
    g = obs_bchw01[:, 1:2]
    b = obs_bchw01[:, 2:3]
    # red-ish pixels
    mask = (r > 0.65) & (g < 0.35) & (b < 0.35)
    return mask.float()  # (B,1,H,W)


@torch.no_grad()
def analysis_D_agent_overlap(args, encoder, loader, H_lat, W_lat, patch_h, patch_w, K, out_dir):
    """
    For each latent patch: compute agent pixel fraction. Aggregate per code.
    """
    print("\n=== D) Code -> agent overlap ===")
    ensure_dir(out_dir)

    sum_agent_frac = np.zeros((K,), dtype=np.float64)
    patch_counts = np.zeros((K,), dtype=np.int64)
    hit_counts = np.zeros((K,), dtype=np.int64)  # how many patches contain any agent pixels (>0)

    target = int(args.n_samples)
    collected = 0
    pbar = tqdm(total=target, desc="D) agent-overlap", unit="obs", ncols=110)

    for batch in loader:
        obs = batch[0]
        if isinstance(obs, (list, tuple)):
            obs = obs[0]
        obs = obs.to(args.device).float().clamp(0, 1)  # (B,3,H,W)
        B = obs.shape[0]

        enc = encoder.encode(obs)
        idx = extract_indices(enc)
        if idx is None:
            raise RuntimeError("encoder.encode did not return indices (LongTensor).")
        grid = indices_to_grid(idx, H_lat, W_lat)  # (B,H_lat,W_lat)

        agent_mask = make_agent_mask(obs)  # (B,1,H,W)

        grid_cpu = grid.detach().cpu().numpy().astype(np.int64)
        agent_cpu = agent_mask.detach().cpu().numpy()  # (B,1,H,W)

        for b in range(B):
            if collected >= target:
                break

            for i in range(H_lat):
                y0, y1 = i * patch_h, (i + 1) * patch_h
                for j in range(W_lat):
                    x0, x1 = j * patch_w, (j + 1) * patch_w
                    c = int(grid_cpu[b, i, j])
                    if c < 0 or c >= K:
                        continue

                    patch_agent = agent_cpu[b, 0, y0:y1, x0:x1]
                    frac = float(patch_agent.mean())
                    sum_agent_frac[c] += frac
                    patch_counts[c] += 1
                    if frac > 0.0:
                        hit_counts[c] += 1

            collected += 1
            pbar.update(1)

        if collected >= target:
            break

    pbar.close()

    mean_agent_frac = np.zeros((K,), dtype=np.float64)
    hit_rate = np.zeros((K,), dtype=np.float64)
    for c in range(K):
        if patch_counts[c] > 0:
            mean_agent_frac[c] = sum_agent_frac[c] / patch_counts[c]
            hit_rate[c] = hit_counts[c] / patch_counts[c]

    # Save CSV
    csv_path = os.path.join(out_dir, "D_agent_overlap.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "patch_count", "mean_agent_frac", "hit_rate"])
        for c in range(K):
            if patch_counts[c] > 0:
                w.writerow([c, int(patch_counts[c]), float(mean_agent_frac[c]), float(hit_rate[c])])

    # Plot: top codes by mean overlap
    used = patch_counts > 0
    codes_used = np.where(used)[0]
    order = codes_used[np.argsort(-mean_agent_frac[codes_used])]
    topk = order[: min(30, len(order))]

    plt.figure(figsize=(10, 4))
    plt.bar([int(c) for c in topk], [float(mean_agent_frac[c]) for c in topk])
    plt.title("Top codes by mean agent overlap")
    plt.xlabel("code")
    plt.ylabel("mean_agent_frac (within patch)")
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "D_top_agent_overlap.png")
    plt.savefig(plot_path, dpi=160)
    plt.close()

    print(f"[D] saved: {csv_path}")
    print(f"[D] saved: {plot_path}")
    if len(topk) > 0:
        print("[D] top-10 codes by mean_agent_frac:")
        for c in topk[:10]:
            print(f"  code {int(c):4d} | mean_agent_frac={mean_agent_frac[c]:.4f} | hit_rate={hit_rate[c]:.3f}")

    return mean_agent_frac, hit_rate


# -------------------------
# E) Transition entropy
# -------------------------
def find_next_obs_in_batch(batch):
    """
    Tries to find next_obs in a dataloader batch.
    Common patterns:
      batch = (obs, action, next_obs, ...)
      batch = dict with keys
    Returns: next_obs tensor or None
    """
    if isinstance(batch, dict):
        for k in ["next_obs", "obs_next", "next", "next_observation"]:
            if k in batch:
                return batch[k]
        return None

    if isinstance(batch, (list, tuple)):
        # Heuristic: the first tensor is obs; next_obs often the 3rd element
        # We'll search for a tensor with same shape as obs
        obs = batch[0]
        if isinstance(obs, (list, tuple)):
            obs = obs[0]
        if torch.is_tensor(obs):
            for item in batch[1:]:
                if torch.is_tensor(item) and item.shape == obs.shape:
                    # could be next_obs or another obs-like field; pick the first match
                    return item
        # fallback: common index 2
        if len(batch) >= 3 and torch.is_tensor(batch[2]):
            return batch[2]
    return None


@torch.no_grad()
def analysis_E_transition_entropy(args, encoder, loader, H_lat, W_lat, K, out_dir):
    """
    Estimate per-code:
      - conditional entropy H(next_code | code) at same latent position
      - change rate P(next!=curr | code)
    """
    print("\n=== E) Code transition entropy ===")
    ensure_dir(out_dir)

    # We store sparse counts: dict[curr_code] -> dict[next_code] -> count
    trans_counts = [defaultdict(int) for _ in range(K)]
    curr_tot = np.zeros((K,), dtype=np.int64)
    change_cnt = np.zeros((K,), dtype=np.int64)

    target = int(args.n_samples)
    collected = 0
    pbar = tqdm(total=target, desc="E) transitions", unit="obs", ncols=110)

    for batch in loader:
        obs = batch[0]
        if isinstance(obs, (list, tuple)):
            obs = obs[0]
        next_obs = find_next_obs_in_batch(batch)

        if next_obs is None:
            pbar.close()
            raise RuntimeError(
                "Could not find next_obs in your dataloader batch.\n"
                "E) needs (obs, next_obs) transitions. Fix by making prepare_dataloader return transitions "
                "(e.g., use buffer split that includes next_obs or add extra_buffer_keys)."
            )

        obs = obs.to(args.device).float().clamp(0, 1)
        next_obs = next_obs.to(args.device).float().clamp(0, 1)

        B = obs.shape[0]
        idx = extract_indices(encoder.encode(obs))
        idx2 = extract_indices(encoder.encode(next_obs))
        if idx is None or idx2 is None:
            raise RuntimeError("encode() did not produce indices for obs/next_obs.")

        g1 = indices_to_grid(idx, H_lat, W_lat)    # (B,H,W)
        g2 = indices_to_grid(idx2, H_lat, W_lat)   # (B,H,W)

        g1 = g1.detach().cpu().numpy().astype(np.int64)
        g2 = g2.detach().cpu().numpy().astype(np.int64)

        for b in range(B):
            if collected >= target:
                break
            for i in range(H_lat):
                for j in range(W_lat):
                    c = int(g1[b, i, j])
                    n = int(g2[b, i, j])
                    if 0 <= c < K and 0 <= n < K:
                        trans_counts[c][n] += 1
                        curr_tot[c] += 1
                        if n != c:
                            change_cnt[c] += 1
            collected += 1
            pbar.update(1)

        if collected >= target:
            break

    pbar.close()

    # compute entropy per code
    entropy = np.zeros((K,), dtype=np.float64)
    change_rate = np.zeros((K,), dtype=np.float64)
    for c in range(K):
        tot = int(curr_tot[c])
        if tot <= 0:
            continue
        change_rate[c] = float(change_cnt[c]) / tot

        # H = -sum p log p
        H = 0.0
        for n, cnt in trans_counts[c].items():
            p = cnt / tot
            if p > 0:
                H -= p * math.log(p + 1e-12)
        entropy[c] = H  # nats

    # Save CSV
    csv_path = os.path.join(out_dir, "E_transition_entropy.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "curr_count", "transition_entropy_nats", "change_rate"])
        for c in range(K):
            if curr_tot[c] > 0:
                w.writerow([c, int(curr_tot[c]), float(entropy[c]), float(change_rate[c])])

    # Plot
    used = curr_tot > 0
    plt.figure(figsize=(6, 5))
    plt.scatter(np.log10(curr_tot[used] + 1), entropy[used])
    plt.title("Transition entropy vs usage")
    plt.xlabel("log10(curr_count+1)")
    plt.ylabel("H(next_code | code) [nats]")
    plt.tight_layout()
    plot1 = os.path.join(out_dir, "E_entropy_vs_count.png")
    plt.savefig(plot1, dpi=160)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(entropy[used], change_rate[used])
    plt.title("Change-rate vs transition entropy")
    plt.xlabel("H(next_code | code) [nats]")
    plt.ylabel("P(next != curr | code)")
    plt.tight_layout()
    plot2 = os.path.join(out_dir, "E_change_vs_entropy.png")
    plt.savefig(plot2, dpi=160)
    plt.close()

    print(f"[E] saved: {csv_path}")
    print(f"[E] saved: {plot1}")
    print(f"[E] saved: {plot2}")

    return entropy, change_rate, curr_tot


# -------------------------
# F) Ablations by label groups
# -------------------------
def load_spatial_labels_csv(path):
    """
    Loads:
      code,count,spatial_entropy,center_x,center_y,label,...
    Returns dict: label -> list[codes], and dict code->label
    """
    label_to_codes = defaultdict(list)
    code_to_label = {}
    if path is None or (not os.path.exists(path)):
        print(f"[F] spatial_csv not found: {path}")
        return label_to_codes, code_to_label

    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            c = int(row["code"])
            lab = row.get("label", "").strip()
            if lab == "":
                continue
            label_to_codes[lab].append(c)
            code_to_label[c] = lab
    return label_to_codes, code_to_label


@torch.no_grad()
def analysis_F_ablations(args, encoder, loader, H_lat, W_lat, patch_scale, K, out_dir, baseline_code, label_to_codes):
    """
    Make montages comparing original recon vs ablated recon for each code-group label.
    """
    print("\n=== F) Ablations by code groups ===")
    ensure_dir(out_dir)

    # pick some observations
    n_show = int(getattr(args, "abl_n_show", 20))
    obs_list = []
    got = 0
    for batch in loader:
        obs = batch[0]
        if isinstance(obs, (list, tuple)):
            obs = obs[0]
        obs = obs.to(args.device).float().clamp(0, 1)
        for b in range(obs.shape[0]):
            obs_list.append(obs[b:b+1])
            got += 1
            if got >= n_show:
                break
        if got >= n_show:
            break

    if len(obs_list) == 0:
        print("[F] no observations to show.")
        return

    obs_show = torch.cat(obs_list, dim=0)  # (N,3,H,W)
    N = obs_show.shape[0]

    # encode all once
    idx = extract_indices(encoder.encode(obs_show))
    if idx is None:
        raise RuntimeError("encode() didn't return indices for ablation.")
    grid = indices_to_grid(idx, H_lat, W_lat)  # (N,H,W)

    # original reconstruction (decode indices)
    orig_recon = decode_from_indices_flat(encoder, grid_to_flat(grid)).clamp(0, 1)

    # helper to save montage
    def save_montage(imgs_chw, title, fname):
        imgs_u8 = []
        for k in range(imgs_chw.shape[0]):
            u8 = to_uint8_hwc(imgs_chw[k])
            u8 = resize_nearest_u8(u8, patch_scale)
            imgs_u8.append(u8)
        sheet = contact_sheet(imgs_u8, ncols=5, pad=8, pad_val=255)
        if sheet is not None:
            save_uint8(sheet, os.path.join(out_dir, fname))

    # Save original recon montage
    save_montage(orig_recon.detach().cpu(), "orig_recon", "F_orig_recon_montage.png")

    # For each label group, ablate
    for lab, codes in label_to_codes.items():
        if len(codes) == 0:
            continue

        codes_set = set(int(c) for c in codes if 0 <= int(c) < K)
        if len(codes_set) == 0:
            continue

        g2 = grid.clone()
        # replace codes in this label with baseline_code
        mask = torch.zeros_like(g2, dtype=torch.bool)
        for c in codes_set:
            mask |= (g2 == c)
        g2[mask] = int(baseline_code)

        recon2 = decode_from_indices_flat(encoder, grid_to_flat(g2)).clamp(0, 1)
        save_montage(recon2.detach().cpu(), f"abl_{lab}", f"F_ablation_{lab}.png")

        # also save side-by-side (orig vs ablated) montage
        side_imgs = []
        for k in range(N):
            a = to_uint8_hwc(orig_recon[k].detach().cpu())
            b = to_uint8_hwc(recon2[k].detach().cpu())
            a = resize_nearest_u8(a, patch_scale)
            b = resize_nearest_u8(b, patch_scale)
            side = np.concatenate([a, b], axis=1)
            side_imgs.append(side)
        sheet = contact_sheet(side_imgs, ncols=3, pad=10, pad_val=255)
        if sheet is not None:
            save_uint8(sheet, os.path.join(out_dir, f"F_side_by_side_{lab}.png"))

        print(f"[F] saved ablation montages for label='{lab}' (codes={len(codes_set)})")

    print(f"[F] outputs in: {out_dir}")


# -------------------------
# main
# -------------------------
@torch.no_grad()
def main_run(args):
    out_dir = args.out_dir
    ensure_dir(out_dir)

    # dataloader
    loader = prepare_dataloader(
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
    sample = next(iter(loader))
    obs0 = sample[0]
    if isinstance(obs0, (list, tuple)):
        obs0 = obs0[0]
    obs_shape = tuple(obs0.shape[1:])  # (C,H,W)
    print(f"[info] obs={obs_shape}")

    # model
    encoder = construct_ae_model(obs_shape, args)[0].to(args.device).eval()
    print(f"[info] encoder={type(encoder).__name__}")

    # codebook size K
    K = getattr(args, "codebook_size", None)
    if hasattr(encoder, "n_embeddings"):
        K = int(encoder.n_embeddings)
    if K is None:
        raise RuntimeError("Could not infer codebook size K. Set --codebook_size.")

    # infer latent size + patch size
    obs1 = obs0[:1].to(args.device).float().clamp(0, 1)
    idx1 = extract_indices(encoder.encode(obs1))
    if idx1 is None:
        raise RuntimeError("encode() did not yield indices; this script expects VQ indices.")
    grid1 = indices_to_grid(idx1)
    H_lat, W_lat = int(grid1.shape[1]), int(grid1.shape[2])
    C, H, W = obs_shape
    patch_h, patch_w = H // H_lat, W // W_lat
    print(f"[info] latent=(h={H_lat}, w={W_lat}) patch=(h={patch_h}, w={patch_w}) codes={K}")

    # baseline code for ablation = most frequent code (we'll compute using quick pass)
    # (We compute it from the same run's encoding statistics on first few batches.)
    counts = np.zeros((K,), dtype=np.int64)
    quick = 0
    for batch in loader:
        obs = batch[0]
        if isinstance(obs, (list, tuple)):
            obs = obs[0]
        obs = obs.to(args.device).float().clamp(0, 1)
        idx = extract_indices(encoder.encode(obs))
        g = indices_to_grid(idx, H_lat, W_lat).detach().cpu().numpy().astype(np.int64)
        for c in g.reshape(-1):
            if 0 <= c < K:
                counts[c] += 1
        quick += obs.shape[0]
        if quick >= min(500, args.n_samples):
            break
    baseline_code = int(np.argmax(counts))
    print(f"[info] baseline_code (most frequent) = {baseline_code}")

    # load spatial label groups
    label_to_codes, _ = load_spatial_labels_csv(args.spatial_csv)
    if len(label_to_codes) == 0:
        print("[warn] No labels loaded from spatial_csv. F) will be skipped unless you provide the CSV.")
    else:
        print("[info] loaded label groups:")
        for lab, codes in label_to_codes.items():
            print(f"  - {lab}: {len(codes)} codes")

    # Recreate loader fresh for full analyses (since we advanced it)
    loader = prepare_dataloader(
        args.env_name, "test",
        batch_size=args.batch_size,
        preprocess=args.preprocess,
        randomize=True,
        n=args.n_samples,
        n_preload=0,
        preload=False,
        extra_buffer_keys=getattr(args, "extra_buffer_keys", None),
    )

    # D
    mean_agent_frac, hit_rate = analysis_D_agent_overlap(
        args, encoder, loader, H_lat, W_lat, patch_h, patch_w, K,
        out_dir=ensure_dir(os.path.join(out_dir, "D_agent_overlap"))
    )

    # Recreate loader for E (so counts/targets match)
    loader = prepare_dataloader(
        args.env_name, "test",
        batch_size=args.batch_size,
        preprocess=args.preprocess,
        randomize=True,
        n=args.n_samples,
        n_preload=0,
        preload=False,
        extra_buffer_keys=getattr(args, "extra_buffer_keys", None),
    )

    # E
    try:
        entropy, change_rate, curr_tot = analysis_E_transition_entropy(
            args, encoder, loader, H_lat, W_lat, K,
            out_dir=ensure_dir(os.path.join(out_dir, "E_transitions"))
        )
    except RuntimeError as e:
        print("\n[skip] E) transition analysis skipped:")
        print(str(e))
        entropy = change_rate = curr_tot = None

    # Recreate loader for F
    loader = prepare_dataloader(
        args.env_name, "test",
        batch_size=args.batch_size,
        preprocess=args.preprocess,
        randomize=True,
        n=args.n_samples,
        n_preload=0,
        preload=False,
        extra_buffer_keys=getattr(args, "extra_buffer_keys", None),
    )

    # F
    if len(label_to_codes) > 0:
        analysis_F_ablations(
            args, encoder, loader, H_lat, W_lat,
            patch_scale=int(args.abl_scale),
            K=K,
            out_dir=ensure_dir(os.path.join(out_dir, "F_ablations")),
            baseline_code=baseline_code,
            label_to_codes=label_to_codes
        )
    else:
        print("[skip] F) ablations skipped (no spatial_csv labels).")

    print("\n[done] D/E/F complete.")
    print(f"[done] outputs root: {out_dir}")


def main():
    parser = make_argparser()

    # Required/explicit paths
    _try_add_arg(parser, "--spatial_csv", type=str,
                 default="analysis/codebook_spatial_maps/codebook_spatial_stats.csv",
                 help="CSV produced by codebook_spatial_maps.py (with labels)")
    _try_add_arg(parser, "--out_dir", type=str,
                 default="analysis/codebook_semantics/analysis_def",
                 help="Output directory")

    # DEF knobs
    _try_add_arg(parser, "--n_samples", type=int, default=5000, help="Number of obs/transitions to analyze")
    _try_add_arg(parser, "--abl_n_show", type=int, default=20, help="How many examples to show in ablation montages")
    _try_add_arg(parser, "--abl_scale", type=int, default=6, help="Upscale for montage images (nearest neighbor)")

    args = parser.parse_args()
    args = process_args(args)

    # Disable cloud logging
    args.wandb = False
    args.comet_ml = False
    args = init_experiment("codebook_analyses_def", args)

    main_run(args)


if __name__ == "__main__":
    main()
