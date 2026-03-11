#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openslide
import tifffile


def read_mask_preview(mask_path: Path, max_dim: int = 2048) -> np.ndarray:
    try:
        arr = tifffile.memmap(str(mask_path))
    except ValueError:
        arr = tifffile.imread(str(mask_path))

    if arr.ndim > 2:
        arr = arr[..., 0]

    h, w = arr.shape
    step = max(1, int(np.ceil(max(h, w) / max_dim)))
    preview = np.asarray(arr[::step, ::step])

    if preview.max() > 1:
        preview = (preview > 0).astype(np.uint8)

    return preview.astype(np.uint8)


def read_input_preview(input_path: Path, max_dim: int = 2048) -> np.ndarray:
    # Try WSI formats first (svs, mrxs, ndpi, etc.)
    try:
        with openslide.OpenSlide(str(input_path)) as slide:
            w, h = slide.dimensions
            scale = max(w, h) / float(max_dim)
            if scale < 1:
                scale = 1
            thumb_size = (max(1, int(w / scale)), max(1, int(h / scale)))
            thumb = slide.get_thumbnail(thumb_size).convert("RGB")
            return np.array(thumb)
    except Exception:
        pass

    # Fallback for normal TIFF/image files
    arr = tifffile.imread(str(input_path))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim > 3:
        arr = arr[..., :3]

    h, w = arr.shape[:2]
    step = max(1, int(np.ceil(max(h, w) / max_dim)))
    return np.asarray(arr[::step, ::step, :3]).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Visualize input WSI and output mask together.")
    parser.add_argument("input_file", type=Path, help="Path to input WSI/image (e.g., data/slide.svs)")
    parser.add_argument("mask_tiff", type=Path, help="Path to output mask TIFF (e.g., outputs/mask.tiff)")
    parser.add_argument("--max-dim", type=int, default=2048, help="Max preview dimension")
    parser.add_argument("--save", type=Path, default=None, help="Optional output PNG path")
    args = parser.parse_args()

    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    if not args.mask_tiff.exists():
        raise FileNotFoundError(f"Mask file not found: {args.mask_tiff}")

    img = read_input_preview(args.input_file, max_dim=args.max_dim)
    mask = read_mask_preview(args.mask_tiff, max_dim=args.max_dim)

    # Match mask size to input preview for overlay
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlay = img.copy()
    overlay[mask_resized > 0] = (0.6 * overlay[mask_resized > 0] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img)
    axes[0].set_title("Input preview")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Mask preview")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (mask in red)")
    axes[2].axis("off")

    plt.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"Saved preview to: {args.save}")

    plt.show()


if __name__ == "__main__":
    main()