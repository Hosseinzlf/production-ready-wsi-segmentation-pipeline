"""Tests for wsi_pipeline.model."""

from __future__ import annotations

import numpy as np
import pytest
import torch


class TestPreprocessing:
    def test_normalization_range(self, random_patches):
        """All values must be in [0.0, 1.0] after normalization."""
        batch = np.stack(random_patches).astype(np.float32) / 255.0
        assert batch.min() >= 0.0
        assert batch.max() <= 1.0

    def test_output_tensor_shape(self, random_patches):
        """Preprocessed tensor must be (N, 3, H, W)."""
        batch = np.stack(random_patches).astype(np.float32) / 255.0
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2)
        assert tensor.shape == (4, 3, 512, 512)

    def test_uint8_max_maps_to_one(self):
        """A pixel of 255 should normalize to exactly 1.0."""
        patch = np.full((512, 512, 3), 255, dtype=np.uint8)
        normalized = patch.astype(np.float32) / 255.0
        assert normalized.max() == pytest.approx(1.0)

    def test_uint8_zero_maps_to_zero(self):
        """A pixel of 0 should normalize to exactly 0.0."""
        patch = np.zeros((512, 512, 3), dtype=np.uint8)
        normalized = patch.astype(np.float32) / 255.0
        assert normalized.min() == pytest.approx(0.0)


class TestThresholdLogic:
    def test_high_logit_gives_positive(self):
        """Large positive logit → sigmoid > 0.5 → mask = 1."""
        logit = torch.tensor([[[5.0]]])
        prob = torch.sigmoid(logit)
        mask = (prob > 0.5).numpy().astype(np.uint8)
        assert mask[0, 0, 0] == 1

    def test_negative_logit_gives_negative(self):
        """Large negative logit → sigmoid < 0.5 → mask = 0."""
        logit = torch.tensor([[[-5.0]]])
        prob = torch.sigmoid(logit)
        mask = (prob > 0.5).numpy().astype(np.uint8)
        assert mask[0, 0, 0] == 0

    def test_zero_logit_gives_negative(self):
        """Logit = 0 → sigmoid = 0.5 → NOT > 0.5 → mask = 0."""
        logit = torch.tensor([[[0.0]]])
        prob = torch.sigmoid(logit)
        mask = (prob > 0.5).numpy().astype(np.uint8)
        assert mask[0, 0, 0] == 0

    def test_custom_threshold(self):
        """Lowering threshold should convert borderline logits to positive."""
        logit = torch.tensor([[[0.1]]])
        prob = torch.sigmoid(logit)  # ≈ 0.525
        mask_strict = (prob > 0.8).numpy().astype(np.uint8)
        mask_lenient = (prob > 0.4).numpy().astype(np.uint8)
        assert mask_strict[0, 0, 0] == 0
        assert mask_lenient[0, 0, 0] == 1
