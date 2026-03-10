"""
Model inference wrapper.

Design decisions:
- Loads the model once and keeps it on the chosen device for the lifetime of the pipeline.
- Preprocessing (normalize to [0,1], NHWC → NCHW) is centralised here so the
  rest of the code never thinks about tensor formats.
- @torch.no_grad() is applied at the predict_batch level — not per-patch — to
  avoid Python overhead from repeated context manager entry.
- Device selection defaults to "auto": uses CUDA if available, otherwise CPU.
  This makes the code portable without any config change.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
from loguru import logger


class SegmentationModel:
    """Wraps a TorchScript JIT segmentation model for batched inference."""

    def __init__(
        self,
        model_path: Path,
        threshold: float = 0.5,
        device: str = "auto",
    ) -> None:
        self.threshold = threshold
        self.device = self._resolve_device(device)
        self.model = self._load_model(model_path)

        logger.info(
            f"Model loaded | device={self.device} | threshold={threshold} | path={model_path}"
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            selected = "cuda" if torch.cuda.is_available() else "cpu"
            logger.debug(f"Auto device selection: {selected}")
            return torch.device(selected)
        return torch.device(device)

    def _load_model(self, model_path: Path) -> torch.jit.ScriptModule:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Make sure the model is downloaded and the path in config.yaml is correct."
            )
        model = torch.jit.load(str(model_path), map_location=self.device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def preprocess(self, patches: List[np.ndarray]) -> torch.Tensor:
        """
        Convert a list of uint8 RGB patches to a normalised float tensor.

        Args:
            patches: List of arrays with shape (H, W, 3), dtype uint8, values [0, 255].

        Returns:
            Tensor of shape (N, 3, H, W), dtype float32, values [0.0, 1.0].
        """
        batch = np.stack(patches).astype(np.float32) / 255.0  # (N, H, W, 3)
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2)  # NHWC → NCHW
        return tensor.to(self.device)

    @torch.no_grad()
    def predict_batch(self, patches: List[np.ndarray]) -> List[np.ndarray]:
        """
        Run inference on a batch of patches.

        Args:
            patches: List of uint8 RGB arrays, each (H, W, 3).

        Returns:
            List of binary masks, each (H, W), dtype uint8, values {0, 1}.
        """
        tensor = self.preprocess(patches)
        logits = self.model(tensor)  # expected: (N, 1, H, W) or (N, H, W)

        # Normalise output shape to (N, H, W)
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        elif logits.dim() != 3:
            raise ValueError(
                f"Unexpected model output shape: {logits.shape}. "
                "Expected (N, H, W) or (N, 1, H, W)."
            )

        probs = torch.sigmoid(logits)
        binary = (probs > self.threshold).cpu().numpy().astype(np.uint8)  # (N, H, W)

        return [binary[i] for i in range(binary.shape[0])]
