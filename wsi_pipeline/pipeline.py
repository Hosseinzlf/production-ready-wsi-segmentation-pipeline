"""
Pipeline orchestrator — wires Reader → Model → Writer together.

The pipeline is intentionally thin: it delegates all domain logic to the
three specialised modules and only handles orchestration concerns:
batching, progress tracking, error propagation, and timing.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Tuple

import openslide
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
        self._run_preflight_checks(wsi_path=wsi_path, output_path=output_path)
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

    def _run_preflight_checks(self, wsi_path: Path, output_path: Path) -> None:
        """
        Validate critical runtime requirements before running expensive work.

        Checks:
        - model path exists and is readable
        - WSI exists and can be opened by OpenSlide
        - output directory is writable
        - free disk space is likely sufficient for temp/output files
        """
        self._verify_model_readable()
        level_dims = self._verify_wsi_readable_and_get_level_dims(wsi_path)
        output_dir = self._ensure_output_dir_writable(output_path)
        self._check_disk_capacity(output_dir, level_dims)

    def _verify_model_readable(self) -> None:
        """Ensure the model exists and can be read."""
        model_path = self.config.model.path
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not model_path.is_file():
            raise FileNotFoundError(f"Model path is not a file: {model_path}")

        try:
            with open(model_path, "rb") as f:
                f.read(1)
        except OSError as exc:
            raise PermissionError(f"Model file is not readable: {model_path}") from exc

    def _verify_wsi_readable_and_get_level_dims(self, wsi_path: Path) -> Tuple[int, int]:
        """Ensure WSI exists/is readable and return chosen level dimensions."""
        if not wsi_path.exists():
            raise FileNotFoundError(f"WSI not found: {wsi_path}")
        if not wsi_path.is_file():
            raise FileNotFoundError(f"WSI path is not a file: {wsi_path}")

        try:
            with open(wsi_path, "rb") as f:
                f.read(1)
        except OSError as exc:
            raise PermissionError(f"WSI file is not readable: {wsi_path}") from exc

        try:
            slide = openslide.OpenSlide(str(wsi_path))
        except openslide.OpenSlideError as exc:
            raise ValueError(
                f"WSI could not be opened by OpenSlide: {wsi_path}"
            ) from exc

        try:
            best_level = self._select_best_level_for_target_mpp(slide)
            level_dims = slide.level_dimensions[best_level]
            return int(level_dims[0]), int(level_dims[1])
        finally:
            slide.close()

    def _select_best_level_for_target_mpp(self, slide: openslide.OpenSlide) -> int:
        """Select the pyramid level closest to configured target MPP."""
        mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        mpp_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
        if mpp_x is None or mpp_y is None:
            logger.warning(
                "MPP metadata missing during preflight; "
                "disk estimate assumes level 0 dimensions."
            )
            return 0

        target_mpp = self.config.model.target_mpp
        base_mpp = (float(mpp_x) + float(mpp_y)) / 2.0
        best_level = 0
        best_diff = float("inf")

        for level in range(slide.level_count):
            level_mpp = base_mpp * slide.level_downsamples[level]
            diff = abs(level_mpp - target_mpp)
            if diff < best_diff:
                best_diff = diff
                best_level = level

        return best_level

    def _ensure_output_dir_writable(self, output_path: Path) -> Path:
        """Ensure output parent directory exists and accepts writes."""
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".wsi-write-test-",
                dir=str(output_dir),
                delete=True,
            ) as tmp:
                tmp.write(b"ok")
                tmp.flush()
        except OSError as exc:
            raise PermissionError(
                f"Output directory is not writable: {output_dir}"
            ) from exc

        return output_dir

    def _check_disk_capacity(self, output_dir: Path, level_dims: Tuple[int, int]) -> None:
        """
        Estimate disk needs and fail early when free space is likely insufficient.

        We store a full uint8 memmap (1 byte/pixel) plus final TIFF output.
        """
        width, height = level_dims
        raw_mask_bytes = width * height  # uint8 -> 1 byte/pixel
        tmp_memmap_bytes = raw_mask_bytes
        estimated_output_bytes = raw_mask_bytes
        safety_buffer_bytes = 256 * 1024 * 1024  # 256 MB
        required_bytes = tmp_memmap_bytes + estimated_output_bytes + safety_buffer_bytes

        free_bytes = shutil.disk_usage(output_dir).free
        logger.info(
            f"Preflight disk estimate | required≈{self._format_bytes(required_bytes)} | "
            f"free={self._format_bytes(free_bytes)} | output_fs={output_dir}"
        )

        if free_bytes < required_bytes:
            raise OSError(
                "Insufficient free disk space for this run. "
                f"Estimated required: {self._format_bytes(required_bytes)}, "
                f"available: {self._format_bytes(free_bytes)}. "
                "Choose a different output location or free disk space."
            )

        warning_threshold = int(required_bytes * 1.25)
        if free_bytes < warning_threshold:
            logger.warning(
                "Disk space is tight for this run "
                f"(required≈{self._format_bytes(required_bytes)}, "
                f"free={self._format_bytes(free_bytes)})."
            )

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        """Human-readable bytes for logs/errors."""
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(num_bytes)
        for unit in units:
            if size < 1024.0 or unit == units[-1]:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

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
