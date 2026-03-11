"""
CLI entry point for the WSI segmentation pipeline.

Usage:
    python main.py data/slide.svs outputs/mask.tiff
    python main.py data/slide.svs outputs/mask.tiff --config config/config.yaml
    python main.py data/slide.svs outputs/mask.tiff --model models/custom.pt --batch-size 16
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from wsi_pipeline.config import PipelineConfig
from wsi_pipeline.pipeline import SegmentationPipeline
from wsi_pipeline.utils import setup_logging

app = typer.Typer(
    name="wsi-segmentation",
    help="Memory-efficient lesion segmentation pipeline for Whole Slide Images.",
    add_completion=False,
)


@app.command()
def run(
    wsi_path: Path = typer.Argument(..., help="Path to input WSI file (.svs or .tiff)"),
    output_path: Path = typer.Argument(..., help="Path for output binary mask (.tiff)"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to YAML config file (uses defaults if omitted)"
    ),
    model_path: Optional[Path] = typer.Option(
        None, "--model", "-m", help="Override model path from config"
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", "-b", help="Override batch size from config"
    ),
    device: Optional[str] = typer.Option(
        None, "--device", "-d", help="Override device (auto/cpu/cuda/cuda:0)"
    ),
    no_tissue_mask: bool = typer.Option(
        False, "--no-tissue-mask", help="Disable tissue detection (process all patches)"
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    # Load config
    cfg = PipelineConfig.from_yaml(config) if config else PipelineConfig.default()

    # Apply CLI overrides
    if model_path:
        cfg.model.path = model_path
    if batch_size:
        cfg.model.batch_size = batch_size
    if device:
        cfg.model.device = device
    if no_tissue_mask:
        cfg.inference.use_tissue_mask = False

    setup_logging(level=log_level, log_file=cfg.logging.file)

    pipeline = SegmentationPipeline(config=cfg)
    pipeline.run(wsi_path=wsi_path, output_path=output_path)


if __name__ == "__main__":
    app()
