#!/usr/bin/env python3
"""
Make montage figures from a folder of PNGs.

- Reads all .png images in an input folder (sorted by filename).
- Arranges 10 sub-images per figure (default grid: 2 rows x 5 cols).
- Puts the image filename under each subplot (robust: uses ax.text()).
- Saves each figure to an output folder.

Usage:
  python montage_pngs.py --input_dir ./imgs --output_dir ./montages
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def chunk_list(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def make_montages(input_dir: Path, output_dir: Path, per_fig: int, rows: int, cols: int, dpi: int):
    if rows * cols != per_fig:
        raise ValueError(f"--rows * --cols must equal --per_fig (got {rows}*{cols} != {per_fig}).")

    png_paths = sorted(input_dir.glob("*.png"))
    if not png_paths:
        raise FileNotFoundError(f"No .png files found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_figs = math.ceil(len(png_paths) / per_fig)

    for fig_idx, batch in enumerate(chunk_list(png_paths, per_fig), start=1):
        fig_w = cols * 4
        fig_h = rows * 4
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=dpi)

        axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]

        for ax_i, ax in enumerate(axes_flat):
            if ax_i < len(batch):
                img_path = batch[ax_i]
                img = mpimg.imread(str(img_path))
                ax.imshow(img)

                # Hide ticks/spines but keep the axes object so text renders reliably
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

                # Filename UNDER the subplot (in axes coords)
                ax.text(
                    0.5, -0.10, img_path.name,
                    transform=ax.transAxes,
                    ha="center", va="top",
                    fontsize=9
                )
            else:
                ax.set_visible(False)

        fig.suptitle(f"Montage {fig_idx}/{total_figs}", fontsize=14)

        # Leave extra room at the bottom for the under-subplot filenames
        fig.tight_layout(rect=[0, 0.06, 1, 0.96])

        out_path = output_dir / f"montage_{fig_idx:03d}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--per_fig", type=int, default=10)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    make_montages(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        per_fig=args.per_fig,
        rows=args.rows,
        cols=args.cols,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
