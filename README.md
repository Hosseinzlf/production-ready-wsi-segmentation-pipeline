# WSI Lesion Segmentation Pipeline

A **memory-efficient**, production-ready pipeline for generating binary lesion masks from Whole Slide Images (WSI) using a pre-trained TorchScript segmentation model.

---

## Quick Start

```bash
# 1. Clone and install
git clone <your-repo-url>
cd wsi-segmentation-pipeline
uv sync                    # or: pip install -e .

# 2. Place your model
cp /path/to/model.pt models/model.pt

# 3. Run
python main.py slide.svs outputs/mask.tiff
python main.py slide.svs outputs/mask.tiff --config config/config.yaml
```

---

## Setup

### Option A — uv (recommended)

```bash
pip install uv
uv sync
uv run python main.py --help
```

### Option B — pip + venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Option C — Docker

```bash
docker build -t wsi-pipeline .

docker run --rm \
  -v /path/to/wsis:/data/wsi \
  -v /path/to/outputs:/data/outputs \
  -v /path/to/models:/app/models \
  wsi-pipeline \
  /data/wsi/slide.svs /data/outputs/mask.tiff
```

> **Note (Linux):** OpenSlide requires `libopenslide0`. Install via:  
> `apt-get install libopenslide0` or `brew install openslide` (macOS).

---

## Usage

```
Usage: main.py [OPTIONS] WSI_PATH OUTPUT_PATH

Arguments:
  WSI_PATH      Path to input WSI file (.svs or .tiff)
  OUTPUT_PATH   Path for output binary mask (.tiff)

Options:
  -c, --config PATH         Path to YAML config file
  -m, --model PATH          Override model path
  -b, --batch-size INT      Override GPU batch size
  -d, --device TEXT         Override device (auto/cpu/cuda/cuda:0)
  --no-tissue-mask          Disable tissue detection
  --log-level TEXT          Logging level [default: INFO]
```

### Examples

```bash
# Default config
python main.py data/slide.svs outputs/mask.tiff

# Custom config
python main.py data/slide.svs outputs/mask.tiff --config config/config.yaml

# GPU with large batch
python main.py data/slide.svs outputs/mask.tiff --device cuda --batch-size 32

# Disable tissue masking (process every patch including background)
python main.py data/slide.svs outputs/mask.tiff --no-tissue-mask
```

---

## Configuration

All parameters live in `config/config.yaml`. No code changes needed to tune the pipeline.

```yaml
model:
  path: "models/model.pt"
  patch_size: 512
  target_mpp: 0.88          # 10x magnification
  threshold: 0.5
  batch_size: 8
  device: "auto"            # auto-selects GPU if available

inference:
  overlap: 64               # pixels of overlap between patches
  use_tissue_mask: true     # skip white/background patches
  tissue_thumbnail_size: 1024
  tissue_threshold: 0.05

output:
  compression: "lzw"
  tile_size: 512
  bigtiff: true             # required for masks > 4 GB
```

---

## Architecture

### Module Overview

```
wsi_pipeline/
├── config.py     — Pydantic config models, YAML loading
├── reader.py     — WSI I/O, MPP detection, tissue masking, patch iteration
├── model.py      — TorchScript model loading, preprocessing, batched inference
├── writer.py     — Memory-mapped mask accumulation, tiled TIFF output
└── pipeline.py   — Orchestration: batching, progress, error handling
```

### Key Design Decisions

#### 1. Memory management via `numpy.memmap`
The output mask is never held in RAM. It lives on disk as a memory-mapped file, and patches are written directly to their position. RAM usage is constant at approximately `batch_size × 512 × 512 × 3 bytes` regardless of WSI size.

**Trade-off:** Requires temporary disk space equal to the mask size. For a 20000×15000 mask that's ~300 MB — acceptable for clinical workstations.

#### 2. Pyramid level selection over rescaling
Instead of reading at an arbitrary zoom and rescaling, we find the pyramid level whose native MPP is closest to the model's expected MPP (0.88). This avoids lossy interpolation and is faster.

**Trade-off:** The actual MPP may differ slightly from 0.88 if no level matches exactly. For the current model specification this is negligible, but a future improvement would be to rescale the closest level.

#### 3. Tissue masking (Otsu on thumbnail)
We detect tissue regions on a low-resolution thumbnail before processing. Background patches (white glass) are skipped entirely, saving 60–80% of inference time on typical slides.

**Trade-off:** Adds ~1 second of overhead per slide. The Otsu threshold can fail on heavily stained or artefact-heavy slides — configurable via `tissue_threshold`.

#### 4. Patch overlap for border artifact suppression
Adjacent patches overlap by `overlap` pixels. Only the centre of each prediction is written to the output. This prevents the "grid" artefact that appears when a model sees artificial borders.

**Trade-off:** Increases total patches processed by ~15-20%. Configurable.

#### 5. Tiled GeoTIFF output
The output is a tiled, compressed, georeferenced TIFF with MPP stored in the resolution tag. This makes it directly compatible with WSI viewers (QuPath, ASAP) and downstream processing tools.

---

## Testing

```bash
pytest tests/ -v
pytest tests/ -v --cov=wsi_pipeline --cov-report=term-missing
```

Tests are unit-level: all OpenSlide and model I/O is mocked, so no WSI files or GPU are required to run them.

---

## Scaling Vision

### Scaling to Many Slides (Batch Processing)

The current pipeline processes one slide at a time. For production, a simple extension would be to wrap `pipeline.run()` in a queue consumer (e.g., Celery, AWS SQS) that pulls WSI paths from a job queue. Each worker is stateless and processes one slide, making horizontal scaling trivial.

### GPU Utilisation

The current bottleneck on GPU machines is I/O: reading patches from disk is slower than inference. Solutions:
- Prefetch patches in a background thread using `torch.utils.data.DataLoader` with `num_workers > 0`
- Cache the current pyramid level tile in a memory-mapped read buffer

### Very Large Slides (>200,000 × 200,000 px)

For slides exceeding a few GB at inference resolution, the memmap approach still holds: the temporary mask file scales linearly with slide area. The `bigtiff=True` flag ensures the output TIFF can exceed 4 GB. For even larger outputs, a Zarr store (via `zarr` + `ome-zarr`) would be preferable as it supports chunked, cloud-native I/O.

---

## Output Format

The output is a single-channel, binary TIFF:
- **Values:** 0 = background, 1 = lesion
- **Format:** Tiled BigTIFF, LZW compressed
- **Resolution:** MPP stored in TIFF resolution tag (compatible with QuPath / ASAP)
- **Spatial alignment:** 1:1 pixel correspondence with the WSI at the inference level
