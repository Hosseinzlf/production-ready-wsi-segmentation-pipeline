"""Tests for pipeline preflight checks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from wsi_pipeline.config import PipelineConfig
from wsi_pipeline.pipeline import SegmentationPipeline


def _build_cfg(model_path: Path) -> PipelineConfig:
    cfg = PipelineConfig.default()
    cfg.model.path = model_path
    cfg.model.target_mpp = 0.88
    return cfg


def test_preflight_fails_when_model_missing(tmp_path):
    cfg = _build_cfg(tmp_path / "missing_model.pt")
    pipeline = SegmentationPipeline(cfg)

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        pipeline._run_preflight_checks(
            wsi_path=tmp_path / "slide.svs",
            output_path=tmp_path / "out" / "mask.tiff",
        )


def test_preflight_fails_when_wsi_missing(tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")

    cfg = _build_cfg(model)
    pipeline = SegmentationPipeline(cfg)

    with pytest.raises(FileNotFoundError, match="WSI not found"):
        pipeline._run_preflight_checks(
            wsi_path=tmp_path / "missing_slide.svs",
            output_path=tmp_path / "out" / "mask.tiff",
        )


def test_preflight_fails_when_disk_space_is_insufficient(tmp_path, mock_slide):
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    wsi = tmp_path / "slide.svs"
    wsi.write_bytes(b"fake-wsi")

    cfg = _build_cfg(model)
    pipeline = SegmentationPipeline(cfg)

    class FakeDiskUsage:
        total = 10 * 1024 * 1024
        used = 9 * 1024 * 1024
        free = 1 * 1024 * 1024

    with patch("wsi_pipeline.pipeline.openslide.OpenSlide", return_value=mock_slide):
        with patch("wsi_pipeline.pipeline.shutil.disk_usage", return_value=FakeDiskUsage()):
            with pytest.raises(OSError, match="Insufficient free disk space"):
                pipeline._run_preflight_checks(
                    wsi_path=wsi,
                    output_path=tmp_path / "out" / "mask.tiff",
                )
