"""Tests for wsi_pipeline.writer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from wsi_pipeline.writer import MaskWriter


@pytest.fixture
def writer(tmp_path):
    """A MaskWriter backed by a real memmap in a temp directory."""
    output = tmp_path / "mask.tiff"
    with patch("tifffile.imwrite"):
        w = MaskWriter(output_path=output, mask_shape=(1024, 1024), mpp=0.88)
        yield w
        # Cleanup without actually calling tifffile
        w._cleanup()


class TestWritePatch:
    def test_patch_written_to_correct_position(self, writer):
        """Patch at (x=0, y=0) should fill top-left of the mask."""
        patch = np.ones((512, 512), dtype=np.uint8)
        writer.write_patch(patch, x=0, y=0, crop=0)

        assert writer._mask[0:512, 0:512].sum() == 512 * 512
        assert writer._mask[512:, :].sum() == 0
        assert writer._mask[:, 512:].sum() == 0

    def test_patch_written_at_offset(self, writer):
        """Patch at (x=100, y=200) should land at the right position."""
        patch = np.ones((512, 512), dtype=np.uint8)
        writer.write_patch(patch, x=100, y=200, crop=0)

        assert writer._mask[200:712, 100:612].sum() == 512 * 512
        assert writer._mask[0:200, :].sum() == 0
        assert writer._mask[:, 0:100].sum() == 0

    def test_out_of_bounds_patch_clamped(self, writer):
        """Patch extending beyond mask boundary should be clamped, not raise."""
        patch = np.ones((512, 512), dtype=np.uint8)
        writer.write_patch(patch, x=900, y=900, crop=0)

        # Only 124×124 = 15376 pixels should be written
        assert writer._mask[900:, 900:].sum() == 124 * 124

    def test_fully_out_of_bounds_patch_ignored(self, writer):
        """Patch fully outside mask should be silently ignored."""
        patch = np.ones((512, 512), dtype=np.uint8)
        writer.write_patch(patch, x=2000, y=2000, crop=0)
        assert writer._mask.sum() == 0

    def test_crop_reduces_written_area(self, writer):
        """With crop=32, a 512×512 patch writes 448×448 at (32, 32)."""
        patch = np.ones((512, 512), dtype=np.uint8)
        writer.write_patch(patch, x=0, y=0, crop=32)

        # Cropped patch is 448×448, placed at (32, 32)
        assert writer._mask[32:480, 32:480].sum() == 448 * 448
        assert writer._mask[0:32, :].sum() == 0
        assert writer._mask[:, 0:32].sum() == 0

    def test_write_after_close_raises(self, tmp_path):
        """Writing after cleanup should raise RuntimeError."""
        output = tmp_path / "mask.tiff"
        with patch("tifffile.imwrite"):
            writer = MaskWriter(output_path=output, mask_shape=(512, 512), mpp=0.88)
            writer._cleanup()

        with pytest.raises(RuntimeError):
            writer.write_patch(np.ones((512, 512), dtype=np.uint8), x=0, y=0)


class TestContextManager:
    def test_save_called_on_clean_exit(self, tmp_path):
        """save() should be called when exiting the context manager normally."""
        output = tmp_path / "mask.tiff"
        with patch("tifffile.imwrite") as mock_imwrite:
            with MaskWriter(output_path=output, mask_shape=(512, 512), mpp=0.88):
                pass
        mock_imwrite.assert_called_once()

    def test_save_not_called_on_exception(self, tmp_path):
        """save() should NOT be called if the block raises an exception."""
        output = tmp_path / "mask.tiff"
        with patch("tifffile.imwrite") as mock_imwrite:
            with pytest.raises(ValueError):
                with MaskWriter(output_path=output, mask_shape=(512, 512), mpp=0.88):
                    raise ValueError("simulated failure")
        mock_imwrite.assert_not_called()
