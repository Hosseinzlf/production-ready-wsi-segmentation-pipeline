"""
WSI Reader — handles all I/O against the slide file.

Design decisions:
- Uses openslide for reading: the standard cross-format library for SVS/TIFF/MRXS/etc.
- Finds the pyramid level closest to the target MPP instead of rescaling, which
  avoids unnecessary interpolation and keeps memory predictable.
- Tissue masking via Otsu on a low-res thumbnail: skips ~60-80% of patches
  in typical slides (background is white), dramatically reducing inference time.
- Patches are yielded one at a time — the caller controls batching, so this
  module has no opinion on GPU batch size.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np
import openslide
from loguru import logger


@dataclass
class PatchInfo:
    """Metadata for a single extracted patch."""

    x_level: int  # top-left x in level-space pixels
    y_level: int  # top-left y in level-space pixels
    x0: int  # top-left x in level-0 pixels (required by openslide.read_region)
    y0: int  # top-left y in level-0 pixels
    row: int  # grid row index
    col: int  # grid column index


class WSIReader:
    """
    Reads patches from a Whole Slide Image at the pyramid level
    that best matches the requested MPP.
    """

    def __init__(
        self,
        wsi_path: Path,
        target_mpp: float,
        patch_size: int,
        overlap: int,
    ) -> None:
        self.wsi_path = wsi_path
        self.target_mpp = target_mpp
        self.patch_size = patch_size
        self.overlap = overlap

        self.slide = openslide.OpenSlide(str(wsi_path))
        self.level, self.mpp = self._find_best_level()
        self.level_dims: Tuple[int, int] = self.slide.level_dimensions[self.level]
        self.downsample: float = self.slide.level_downsamples[self.level]

        logger.info(
            f"Opened: {wsi_path.name} | "
            f"Using level {self.level} | "
            f"MPP={self.mpp:.4f} (target={target_mpp}) | "
            f"Dims={self.level_dims}"
        )

    # ------------------------------------------------------------------
    # Level selection
    # ------------------------------------------------------------------

    def _find_best_level(self) -> Tuple[int, float]:
        """Return (level_index, actual_mpp) for the level closest to target_mpp."""
        mpp_x = self.slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        mpp_y = self.slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)

        if mpp_x is None or mpp_y is None:
            logger.warning(
                "MPP metadata not found — assuming level 0 equals target MPP. "
                "Results may be at wrong magnification."
            )
            return 0, self.target_mpp

        base_mpp = (float(mpp_x) + float(mpp_y)) / 2.0
        logger.debug(f"Slide base MPP (level 0): {base_mpp:.4f}")

        best_level, best_mpp, best_diff = 0, base_mpp, float("inf")

        for level in range(self.slide.level_count):
            level_mpp = base_mpp * self.slide.level_downsamples[level]
            diff = abs(level_mpp - self.target_mpp)
            logger.debug(f"  Level {level}: MPP={level_mpp:.4f}, diff={diff:.4f}")
            if diff < best_diff:
                best_diff = diff
                best_level = level
                best_mpp = level_mpp

        return best_level, best_mpp

    # ------------------------------------------------------------------
    # Tissue detection
    # ------------------------------------------------------------------

    def get_tissue_mask(self, thumbnail_size: int = 1024) -> np.ndarray:
        """
        Return a boolean tissue mask at thumbnail resolution.

        Uses Otsu thresholding on a grayscale thumbnail: background (glass/white)
        gets thresholded out, leaving only tissue regions as True.
        """
        thumbnail = self.slide.get_thumbnail((thumbnail_size, thumbnail_size))
        gray = np.array(thumbnail.convert("L"))

        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Clean up small noise and fill small holes
        kernel = np.ones((7, 7), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        tissue_fraction = binary.mean() / 255.0
        logger.info(
            f"Tissue mask computed | "
            f"Thumbnail size: {thumbnail.size} | "
            f"Tissue coverage: {tissue_fraction:.1%}"
        )

        return binary.astype(bool)

    # ------------------------------------------------------------------
    # Patch iteration
    # ------------------------------------------------------------------

    def iter_patches(
        self,
        tissue_mask: Optional[np.ndarray] = None,
        tissue_threshold: float = 0.05,
    ) -> Iterator[Tuple[PatchInfo, np.ndarray]]:
        """
        Yield (PatchInfo, patch_array) for all patches at the target level.

        Args:
            tissue_mask: Boolean mask at thumbnail resolution. If provided,
                         patches with < tissue_threshold tissue fraction are skipped.
            tissue_threshold: Minimum tissue fraction in a patch to process it.

        Yields:
            (PatchInfo, np.ndarray of shape [patch_size, patch_size, 3], dtype uint8)
        """
        level_w, level_h = self.level_dims
        stride = self.patch_size - self.overlap

        n_cols = max(1, (level_w + stride - 1) // stride)
        n_rows = max(1, (level_h + stride - 1) // stride)

        thumb_h, thumb_w = (
            (tissue_mask.shape[0], tissue_mask.shape[1])
            if tissue_mask is not None
            else (1, 1)
        )

        skipped = 0

        for row in range(n_rows):
            for col in range(n_cols):
                x_level = col * stride
                y_level = row * stride

                # --- Tissue check ---
                if tissue_mask is not None:
                    tx0 = int(x_level / level_w * thumb_w)
                    ty0 = int(y_level / level_h * thumb_h)
                    tx1 = min(
                        int((x_level + self.patch_size) / level_w * thumb_w),
                        thumb_w - 1,
                    )
                    ty1 = min(
                        int((y_level + self.patch_size) / level_h * thumb_h),
                        thumb_h - 1,
                    )
                    region = tissue_mask[ty0 : ty1 + 1, tx0 : tx1 + 1]
                    if region.size == 0 or region.mean() < tissue_threshold:
                        skipped += 1
                        continue

                # Convert level coords → level-0 coords for openslide.read_region
                x0 = int(x_level * self.downsample)
                y0 = int(y_level * self.downsample)

                patch_pil = self.slide.read_region(
                    (x0, y0), self.level, (self.patch_size, self.patch_size)
                )
                patch_rgb = np.array(patch_pil.convert("RGB"))

                info = PatchInfo(
                    x_level=x_level,
                    y_level=y_level,
                    x0=x0,
                    y0=y0,
                    row=row,
                    col=col,
                )

                yield info, patch_rgb

        if tissue_mask is not None:
            total = n_rows * n_cols
            logger.info(
                f"Patch iteration complete | "
                f"Skipped {skipped}/{total} background patches "
                f"({skipped / total:.1%})"
            )

    def close(self) -> None:
        self.slide.close()

    def __enter__(self) -> "WSIReader":
        return self

    def __exit__(self, *args) -> None:
        self.close()
