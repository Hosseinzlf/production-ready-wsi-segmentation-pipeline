"""
Mask writer — incrementally builds the output mask without loading it into RAM.

Design decisions:
- Uses numpy.memmap to accumulate patch predictions on disk.
  This keeps RAM usage constant regardless of WSI size: only one batch of
  patches ever lives in memory at a time.
- Writes the final result as a tiled, compressed GeoTIFF via tifffile.
  Tiled TIFFs allow efficient random-access reads for downstream processing
  (e.g., QuPath, ASAP, or other WSI viewers).
- bigtiff=True is set by default to support masks > 4 GB (large WSIs at 10x).
- MPP is stored in the TIFF resolution tag so spatial coordinates are preserved.
- Overlap cropping: when patches are extracted with overlap, only the centre
  of each patch is written to avoid seam artifacts at patch boundaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tifffile
from loguru import logger


class MaskWriter:
    """
    Accumulates patch masks into a memory-mapped array and saves as tiled TIFF.

    Usage:
        with MaskWriter(output_path, mask_shape, mpp) as writer:
            writer.write_patch(mask, x=100, y=200, crop=32)
        # TIFF is written on __exit__
    """

    def __init__(
        self,
        output_path: Path,
        mask_shape: Tuple[int, int],  # (height, width)
        mpp: float,
        compression: str = "lzw",
        tile_size: int = 512,
        bigtiff: bool = True,
    ) -> None:
        self.output_path = output_path
        self.mask_shape = mask_shape
        self.mpp = mpp
        self.compression = compression
        self.tile_size = tile_size
        self.bigtiff = bigtiff

        self._memmap_path = output_path.with_suffix(".tmp.mmap")
        self._mask: Optional[np.memmap] = np.memmap(
            str(self._memmap_path),
            dtype=np.uint8,
            mode="w+",
            shape=mask_shape,
        )

        size_mb = (mask_shape[0] * mask_shape[1]) / (1024**2)
        logger.info(
            f"Mask writer ready | "
            f"shape={mask_shape} | "
            f"size≈{size_mb:.1f} MB on disk | "
            f"tmp={self._memmap_path}"
        )

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write_patch(
        self,
        patch_mask: np.ndarray,
        x: int,
        y: int,
        crop: int = 0,
    ) -> None:
        """
        Write a patch binary mask to the correct position in the full mask.

        Args:
            patch_mask: 2D array (H, W) with dtype uint8, values {0, 1}.
            x: Top-left x coordinate in level-space pixels.
            y: Top-left y coordinate in level-space pixels.
            crop: Pixels to crop from each side of the patch (overlap handling).
                  Typically overlap // 2.
        """
        if self._mask is None:
            raise RuntimeError("MaskWriter has already been closed.")

        if crop > 0 and patch_mask.shape[0] > 2 * crop and patch_mask.shape[1] > 2 * crop:
            patch_mask = patch_mask[crop:-crop, crop:-crop]
            x += crop
            y += crop

        ph, pw = patch_mask.shape
        h, w = self.mask_shape

        # Destination bounds (clamped to mask)
        dst_x0 = max(0, x)
        dst_y0 = max(0, y)
        dst_x1 = min(x + pw, w)
        dst_y1 = min(y + ph, h)

        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            return  # Patch entirely out of bounds

        # Source bounds (handles negative x/y at borders)
        src_x0 = dst_x0 - x
        src_y0 = dst_y0 - y
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)

        self._mask[dst_y0:dst_y1, dst_x0:dst_x1] = patch_mask[src_y0:src_y1, src_x0:src_x1]

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Flush the memmap and write the final tiled TIFF."""
        if self._mask is None:
            logger.warning("save() called but mask is already closed.")
            return

        logger.info("Flushing memmap and writing TIFF...")
        self._mask.flush()

        # Resolution: pixels per centimetre (tifffile convention)
        # mpp is micrometres/pixel → 1e4 µm/cm → pixels/cm = 1e4 / mpp
        resolution = (1e4 / self.mpp, 1e4 / self.mpp)

        tifffile.imwrite(
            str(self.output_path),
            self._mask,
            bigtiff=self.bigtiff,
            compression=self.compression,
            tile=(self.tile_size, self.tile_size),
            resolutionunit=tifffile.RESUNIT.CENTIMETER,
            resolution=resolution,
            photometric="minisblack",
        )

        logger.success(f"Mask saved → {self.output_path}")
        self._cleanup()

    def _cleanup(self) -> None:
        """Release memmap and delete the temporary file."""
        if self._mask is not None:
            del self._mask
            self._mask = None
        if self._memmap_path.exists():
            self._memmap_path.unlink()
            logger.debug(f"Temp file removed: {self._memmap_path}")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "MaskWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self.save()
        else:
            logger.error(f"Pipeline failed — cleaning up temp files. Error: {exc_val}")
            self._cleanup()
        return False  # do not suppress exceptions
