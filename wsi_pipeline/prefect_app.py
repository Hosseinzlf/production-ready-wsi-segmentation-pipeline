"""
Prefect integration with minimal impact on the core pipeline.

This module adds:
- A Prefect flow that wraps the existing SegmentationPipeline.
- A lightweight "server" command that serves the flow as a deployment.

The existing CLI and pipeline code remain unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from prefect import flow, get_run_logger, task

from wsi_pipeline.config import PipelineConfig
from wsi_pipeline.pipeline import SegmentationPipeline
from wsi_pipeline.utils import setup_logging

app = typer.Typer(
    name="wsi-prefect",
    help="Prefect server/client entrypoints for WSI segmentation.",
    add_completion=False,
)


def _build_config(
    config: Optional[Path],
    model_path: Optional[Path],
    batch_size: Optional[int],
    device: Optional[str],
    no_tissue_mask: bool,
    log_level: str,
) -> PipelineConfig:
    """Load and override pipeline config from CLI-like parameters."""
    cfg = PipelineConfig.from_yaml(config) if config else PipelineConfig.default()

    if model_path:
        cfg.model.path = model_path
    if batch_size:
        cfg.model.batch_size = batch_size
    if device:
        cfg.model.device = device
    if no_tissue_mask:
        cfg.inference.use_tissue_mask = False

    # Keep existing logging behavior and let Prefect capture stdout/stderr.
    setup_logging(level=log_level, log_file=cfg.logging.file)
    return cfg


@task(
    name="Run Segmentation Pipeline",
    retries=2,
    retry_delay_seconds=10,
)
def _run_pipeline_task(
    wsi_path: Path,
    output_path: Path,
    config: Optional[Path],
    model_path: Optional[Path],
    batch_size: Optional[int],
    device: Optional[str],
    no_tissue_mask: bool,
    log_level: str,
) -> str:
    cfg = _build_config(
        config=config,
        model_path=model_path,
        batch_size=batch_size,
        device=device,
        no_tissue_mask=no_tissue_mask,
        log_level=log_level,
    )
    pipeline = SegmentationPipeline(config=cfg)
    result = pipeline.run(wsi_path=wsi_path, output_path=output_path)
    return str(result)


@flow(name="wsi-segmentation-flow")
def segment_wsi_flow(
    wsi_path: str,
    output_path: str,
    config: str = "config/config.yaml",
    model_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    no_tissue_mask: bool = False,
    log_level: str = "INFO",
) -> str:
    """
    Prefect flow entrypoint for one WSI segmentation request.
    """
    logger = get_run_logger()
    logger.info("Received request | input=%s | output=%s", wsi_path, output_path)

    config_path = Path(config) if config else None
    result = _run_pipeline_task.submit(
        wsi_path=Path(wsi_path),
        output_path=Path(output_path),
        config=config_path,
        model_path=Path(model_path) if model_path else None,
        batch_size=batch_size,
        device=device,
        no_tissue_mask=no_tissue_mask,
        log_level=log_level,
    ).result()

    logger.info("Request completed | output=%s", result)
    return result


@app.command()
def serve(
    deployment_name: str = typer.Option(
        "wsi-segmentation",
        "--deployment-name",
        help="Deployment name shown in Prefect UI.",
    ),
    concurrency_limit: int = typer.Option(
        1,
        "--concurrency-limit",
        min=1,
        help="Maximum number of flow runs processed in parallel by this process.",
    ),
) -> None:
    """
    Start a long-running Prefect deployment process.

    Before running this command, start Prefect server:
        prefect server start
    """
    segment_wsi_flow.serve(
        name=deployment_name,
        tags=["wsi", "segmentation"],
        limit=concurrency_limit,
    )


@app.command("run-local")
def run_local(
    wsi_path: Path = typer.Argument(..., help="Path to input WSI file"),
    output_path: Path = typer.Argument(..., help="Path for output binary mask"),
    config: Optional[Path] = typer.Option(
        Path("config/config.yaml"), "--config", "-c", help="YAML config file"
    ),
    model_path: Optional[Path] = typer.Option(None, "--model", "-m", help="Model path"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Batch size"),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Compute device"),
    no_tissue_mask: bool = typer.Option(False, "--no-tissue-mask", help="Process all patches"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """
    Run the Prefect flow once in-process (useful for smoke tests).
    """
    result = segment_wsi_flow(
        wsi_path=str(wsi_path),
        output_path=str(output_path),
        config=str(config) if config else "",
        model_path=str(model_path) if model_path else None,
        batch_size=batch_size,
        device=device,
        no_tissue_mask=no_tissue_mask,
        log_level=log_level,
    )
    typer.echo(f"Flow completed: {result}")


if __name__ == "__main__":
    app()
