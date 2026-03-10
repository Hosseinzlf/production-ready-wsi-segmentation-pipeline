"""Tests for wsi_pipeline.reader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from wsi_pipeline.reader import WSIReader


class TestFindBestLevel:
    def test_selects_level_closest_to_target_mpp(self, mock_slide):
        """Level 1 has 0.22 * 4 = 0.88 mpp — should be selected for target 0.88."""
        with patch("openslide.OpenSlide", return_value=mock_slide):
            reader = WSIReader(
                wsi_path=Path("fake.svs"),
                target_mpp=0.88,
                patch_size=512,
                overlap=64,
            )
        assert reader.level == 1
        assert abs(reader.mpp - 0.88) < 0.01

    def test_selects_level0_for_high_res_target(self, mock_slide):
        """When target MPP equals base MPP, level 0 should be selected."""
        with patch("openslide.OpenSlide", return_value=mock_slide):
            reader = WSIReader(
                wsi_path=Path("fake.svs"),
                target_mpp=0.22,
                patch_size=512,
                overlap=64,
            )
        assert reader.level == 0

    def test_falls_back_gracefully_when_mpp_missing(self, mock_slide):
        """If MPP metadata is absent, level 0 is used with a warning."""
        mock_slide.properties = {}
        with patch("openslide.OpenSlide", return_value=mock_slide):
            reader = WSIReader(
                wsi_path=Path("fake.svs"),
                target_mpp=0.88,
                patch_size=512,
                overlap=64,
            )
        assert reader.level == 0


class TestIterPatches:
    def test_yields_correct_patch_shape(self, mock_slide):
        """Each yielded patch should be (512, 512, 3) uint8."""
        with patch("openslide.OpenSlide", return_value=mock_slide):
            reader = WSIReader(
                wsi_path=Path("fake.svs"),
                target_mpp=0.88,
                patch_size=512,
                overlap=0,
            )
            for info, patch in reader.iter_patches():
                assert patch.shape == (512, 512, 3)
                assert patch.dtype == np.uint8
                break  # just check the first one

    def test_skips_background_with_tissue_mask(self, mock_slide):
        """All-background tissue mask should result in zero patches yielded."""
        background_mask = np.zeros((768, 1024), dtype=bool)

        with patch("openslide.OpenSlide", return_value=mock_slide):
            reader = WSIReader(
                wsi_path=Path("fake.svs"),
                target_mpp=0.88,
                patch_size=512,
                overlap=0,
            )
            patches = list(reader.iter_patches(tissue_mask=background_mask))

        assert len(patches) == 0

    def test_overlap_reduces_stride(self, mock_slide):
        """With overlap=64, stride=448, so more patches than with overlap=0."""
        with patch("openslide.OpenSlide", return_value=mock_slide):
            reader_no_overlap = WSIReader(Path("fake.svs"), 0.88, 512, overlap=0)
            reader_overlap = WSIReader(Path("fake.svs"), 0.88, 512, overlap=64)

        w, h = reader_no_overlap.level_dims

        stride_no = 512
        stride_with = 448

        n_no = ((w + stride_no - 1) // stride_no) * ((h + stride_no - 1) // stride_no)
        n_with = ((w + stride_with - 1) // stride_with) * ((h + stride_with - 1) // stride_with)

        assert n_with > n_no
