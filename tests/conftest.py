"""Shared pytest fixtures."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def mock_slide():
    """A mock openslide.OpenSlide with realistic metadata."""
    slide = MagicMock()
    slide.level_count = 3
    # level 0: full res, level 1: 4× down (~0.88 mpp), level 2: 16× down
    slide.level_dimensions = [(80000, 60000), (20000, 15000), (5000, 3750)]
    slide.level_downsamples = [1.0, 4.0, 16.0]
    slide.properties = {
        "openslide.mpp-x": "0.22",
        "openslide.mpp-y": "0.22",
    }

    def read_region(location, level, size):
        """Return a white RGBA image for any region."""
        from PIL import Image
        return Image.new("RGBA", size, (255, 255, 255, 255))

    slide.read_region.side_effect = read_region

    thumbnail = MagicMock()
    thumbnail.size = (1024, 768)
    # Return a mostly-tissue image (dark pixels)
    import numpy as np
    thumb_array = np.full((768, 1024, 3), 100, dtype=np.uint8)
    from PIL import Image
    slide.get_thumbnail.return_value = Image.fromarray(thumb_array)

    return slide


@pytest.fixture
def random_patches():
    """A batch of 4 random uint8 patches."""
    return [np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8) for _ in range(4)]
