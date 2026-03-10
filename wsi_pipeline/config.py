"""
Configuration management using Pydantic for validation.
All pipeline parameters live here — no hardcoded values elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    path: Path = Path("models/model.pt")
    patch_size: int = 512
    target_mpp: float = 0.88
    threshold: float = 0.5
    batch_size: int = 8
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", etc.


class InferenceConfig(BaseModel):
    overlap: int = 64  # pixel overlap between adjacent patches
    use_tissue_mask: bool = True  # skip background patches (big speedup)
    tissue_thumbnail_size: int = 1024  # resolution of the tissue detection thumbnail
    tissue_threshold: float = 0.05  # min tissue fraction to process a patch


class OutputConfig(BaseModel):
    compression: Literal["lzw", "deflate", "none"] = "lzw"
    tile_size: int = 512
    bigtiff: bool = True  # required for masks > 4 GB


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: Optional[Path] = None


class PipelineConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def default(cls) -> "PipelineConfig":
        return cls()
