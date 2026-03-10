"""
Pipeline orchestrator — wires Reader → Model → Writer together.

The pipeline is intentionally thin: it delegates all domain logic to the
three specialised modules and only handles orchestration concerns:
batching, progress tracking, error propagation, and timing.
"""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from wsi_pipeline.config import PipelineConfig
from wsi_pipeline.model import SegmentationModel
from wsi_pipeline.reader import PatchInfo, WSIReader
from wsi_pipeline.writer import MaskWriter


class SegmentationPipeline:
    """
    End-to-end WSI segmentation pipeline.

    Usage:
        cfg = PipelineConfig.from_yaml(Path("config/config.yaml"))
        pipeline = SegmentationPipeline(cfg)
        pipeline.run(Path("slide.svs"), Path("outputs/slide_mask.tiff"))
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(self, wsi_path: Path, output_path: Path) -> Path:
        """
        Run the full pipeline for a single WSI.

        Args:
            wsi_path: Path to the input WSI file (.svs, .tiff, etc.)
            output_path: Path for the output binary mask (.tiff).

        Returns:
            The resolved output_path.
        """
        if not wsi_path.exists():
            raise FileNotFoundError(f"WSI not found: {wsi_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        start_time = time.monotonic()
        logger.info(f"{'='*60}")
        logger.info(f"Pipeline start: {wsi_path.name}")

        model = SegmentationModel(
            model_path=self.config.model.path,
            threshold=self.config.model.threshold,
            device=self.config.model.device,
        )

        with WSIReader(
            wsi_path=wsi_path,
            target_mpp=self.config.model.target_mpp,
            patch_size=self.config.model.patch_size,
            overlap=self.config.inference.overlap,
        ) as reader:

            tissue_mask = None
            if self.config.inference.use_tissue_mask:
                logger.info("Computing tissue mask to skip background patches...")
                tissue_mask = reader.get_tissue_mask(
                    thumbnail_size=self.config.inference.tissue_thumbnail_size
                )

            level_w, level_h = reader.level_dims
            stride = self.config.model.patch_size - self.config.inference.overlap
            n_cols = max(1, (level_w + stride - 1) // stride)
            n_rows = max(1, (level_h + stride - 1) // stride)
            total_patches = n_rows * n_cols

            logger.info(
                f"Grid: {n_rows} rows × {n_cols} cols = {total_patches} patches max"
            )

            with MaskWriter(
                output_path=output_path,
                mask_shape=(level_h, level_w),
                mpp=reader.mpp,
                compression=self.config.output.compression,
                tile_size=self.config.output.tile_size,
                bigtiff=self.config.output.bigtiff,
            ) as writer:

                self._run_inference_loop(
                    reader=reader,
                    model=model,
                    writer=writer,
                    tissue_mask=tissue_mask,
                    total_patches=total_patches,
                )

        elapsed = time.monotonic() - start_time
        logger.info(f"Pipeline complete in {elapsed:.1f}s → {output_path}")
        logger.info(f"{'='*60}")

        return output_path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_inference_loop(
        self,
        reader: WSIReader,
        model: SegmentationModel,
        writer: MaskWriter,
        tissue_mask,
        total_patches: int,
    ) -> None:
        """Batch patches, run inference, write results."""
        batch_size = self.config.model.batch_size
        crop = self.config.inference.overlap // 2
        tissue_threshold = self.config.inference.tissue_threshold

        batch_patches: list = []
        batch_infos: list[PatchInfo] = []

        patch_iter = reader.iter_patches(
            tissue_mask=tissue_mask,
            tissue_threshold=tissue_threshold,
        )

        with tqdm(
            total=total_patches,
            desc=f"Segmenting {reader.wsi_path.name}",
            unit="patch",
            dynamic_ncols=True,
        ) as pbar:

            for info, patch in patch_iter:
                batch_patches.append(patch)
                batch_infos.append(info)

                if len(batch_patches) >= batch_size:
                    self._flush_batch(model, writer, batch_patches, batch_infos, crop)
                    pbar.update(len(batch_patches))
                    batch_patches.clear()
                    batch_infos.clear()

            # Process remaining patches
            if batch_patches:
                self._flush_batch(model, writer, batch_patches, batch_infos, crop)
                pbar.update(len(batch_patches))

    def _flush_batch(
        self,
        model: SegmentationModel,
        writer: MaskWriter,
        patches: list,
        infos: list[PatchInfo],
        crop: int,
    ) -> None:
        """Run model on one batch and write results to mask."""
        masks = model.predict_batch(patches)
        for info, mask in zip(infos, masks):
            writer.write_patch(
                patch_mask=mask,
                x=info.x_level,
                y=info.y_level,
                crop=crop,
            )
