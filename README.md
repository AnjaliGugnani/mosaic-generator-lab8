---
title: Image Mosaic Generator (Lab 8)
emoji: 🧩
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# Image Mosaic Generator: profiled, optimized and refactored (Lab 8)

Rebuilds any photo out of colour tiles. This version of my Lab 1 project is about **780× faster** for a 512×512 image with a 32×32 grid (4.06 s → 5.2 ms) and is organised as a reusable Python package.

**Live demo:** https://mosaic-generator-lab8.onrender.com/

**Author:** Anjali Puri

## Installation

```bash
git clone <your-repo-url> && cd <repo>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                                       # opens http://127.0.0.1:7860
```

Python 3.10 or newer is required (3.12 recommended). `matplotlib`, `pandas` and `line_profiler` are only needed for the notebook and benchmarks.

## Usage

### Web app
Upload an image or click an example, choose the grid size and number of tiles, and press **Generate Mosaic**. The app shows the segmented preview (each cell's average colour), the mosaic, MSE and optionally SSIM, plus the processing time broken down by pipeline stage.

New compared with Lab 1:
- a processing-time panel with a per-stage breakdown and cache statistics;
- a choice to match cells against the measured colours of the tile photos instead of the nominal palette (visibly closer mosaics, about 20–35% lower MSE);
- SSIM is optional (off by default), because it costs more than the mosaic itself on large images;
- uploads larger than 1024 px are downscaled once before processing, to keep the free-tier demo responsive (the Python API's limit is 2048 px, `config.MAX_IMAGE_DIM`);
- grid sizes from 8 to 128 and tile sets from 3 to 13.

### Python API

```python
from mosaic_generator import MosaicBuilder, TileManager

tile_manager = TileManager(tile_directory="tiles/")       # decodes tiles once
builder = MosaicBuilder(tile_manager, grid_size=(32, 32), tile_count=9)

mosaic = builder.create_mosaic(input_image)                # numpy array, PIL image or path
similarity = builder.compute_similarity(input_image, mosaic)   # {'mse': ..., 'ssim': ...}

result = builder.build(input_image)                        # mosaic + segmented + indices + timings
print(result.timings, result.tile_usage)
```

Options: `match_on="palette"` (Lab 1 behaviour, default) or `"tile_mean"`; `matcher="auto" | "bruteforce" | "kdtree"`. Any folder of images works as a tile set. Files named after the Lab 1 palette colours keep the Lab 1 order, and other files are appended.

### Reproducing the benchmarks

```bash
pip install line_profiler jupyter
jupyter notebook profiling_analysis.ipynb   # Kernel → Restart & Run All (~5 min)
python benchmarks/run_benchmarks.py         # same timings from the command line
python -m pytest tests/                     # or: python -m unittest tests.test_mosaic
```

## Performance benchmarks

Baseline = my Lab 1 code (`baseline/lab1_baseline.py`, unchanged apart from removing the UI). Both versions use the same original Lab 1 tile photos and 12 tiles. Each time is the mean of repeated runs (3 for the baseline, 20 for Lab 8) on an Apple Silicon Mac (arm64) with Python 3.12, NumPy 2.2 and Pillow 11.3. "Warm" means the tile cache for that cell size is filled, which is the normal state in the running app; "cold" clears the cache before every run.

| Image | Grid | Baseline (s) | Lab 8 warm (ms) | Lab 8 cold (ms) | Speedup warm | Speedup cold |
|---|---|---|---|---|---|---|
| 256×256 | 16×16 | 4.12 | 1.20 | 7.63 | 3,441× | 540× |
| **512×512** | **32×32** | **4.06** | **5.20** | **11.75** | **781×** | **345×** |
| 1024×1024 | 64×64 | 4.69 | 21.00 | 27.34 | 223× | 171× |

All 9 image/grid combinations are in the notebook; the slowest is still 161× faster cold. **Correctness:** in all 9 configurations, 100% of cells receive the same tile as in Lab 1, the segmented image is bit-identical, and the mosaic pixel MSE versus Lab 1 is at most 0.27 (resampling noise only).

Where the speedup comes from (512×512, 32×32; a separate measurement, so the totals differ slightly from the table above):

| Step | Time | Cumulative speedup |
|---|---|---|
| Lab 1 baseline | 4,041 ms | 1× |
| + decode tiles once at startup (in-memory thumbnails) | 15.6 ms | 260× |
| + LRU cache of tiles resized per cell size | 9.1 ms | 446× |
| + cached k-d tree matcher, `reduceat` cell means, broadcast preview | 5.1 ms | 786× |

Tile loading now happens once per process: about 2.3 s for the 75 MB original photos (4.8 s without `Image.draft`), or about 40 ms for the small tiles the app ships with.

## Project structure

```
mosaic_generator/
├── __init__.py         exports MosaicBuilder, TileManager, errors
├── config.py           all constants (grid limits, palette, cache sizes, thresholds)
├── image_processor.py  loading, RGB conversion, one-time resize, crop, vectorized grid ops
├── tile_manager.py     TileManager: load once, thumbnails, features, LRU resize cache
├── mosaic_builder.py   MosaicBuilder + matching strategies (brute force / k-d tree)
├── metrics.py          MSE, PSNR, SSIM, tile agreement
└── utils.py            error classes, validation, timing helpers
app.py                  Gradio web app
baseline/               Lab 1 code and original tiles (benchmark baseline only)
benchmarks/             shared harness, CLI benchmark, results JSON, figures
examples/               sample test images: astronaut at 256/512/1024 px (benchmarks) + sunflower, coffee, chelsea_cat, rocket
tiles/                  small tile set used by the app (192 KB)
tests/test_mosaic.py    15 unit tests
profiling_analysis.ipynb
```

## Deploying to Render (live demo)

1. Push the project to a GitHub repository. Including `baseline/` is fine; no single file is near GitHub's 100 MB limit.
2. In the Render dashboard choose **New → Blueprint**, connect the repository, and Render reads `render.yaml` (free plan, Python 3.12, `pip install -r requirements.txt`, `python app.py`).
   Manual alternative: **New → Web Service**, runtime Python 3, the same build and start commands, instance type **Free**, and the environment variable `PYTHON_VERSION=3.12.8`.
3. When the deploy log shows Gradio running, open the `https://<name>.onrender.com` URL and click an example image to check it works.

`app.py` listens on `0.0.0.0:$PORT` when Render sets `PORT`, and loads the 192 KB tile set once at startup. Free services sleep after 15 minutes without traffic; the first visit afterwards takes about a minute while the app wakes up, then requests are fast again.

## Deploying to Hugging Face Spaces (alternative)

1. Create a new Space with the **Gradio** SDK.
2. Upload `app.py`, `requirements.txt`, `README.md` (the YAML header above configures the Space), `mosaic_generator/`, `tiles/` and `examples/`. Leave out `baseline/`: its 75 MB of tiles are only needed for the benchmarks.
3. Wait for the build to finish, open the app, and click an example to check it works. 

## Sample test images

The `examples/` folder contains the test images used in the benchmarks (`astronaut_256.png`, `astronaut_512.png`, `astronaut_1024.png`, the same scene at three sizes) and extra example images for the web app (`sunflower.jpg`, `coffee.jpg`, `chelsea_cat.jpg`, `rocket.jpg`). The web app shows one image per scene: the sunflower (the image I used most in Lab 1), the 512 px astronaut, the cat, the coffee and the rocket.

Credits: the astronaut, coffee, cat and rocket images are public-domain samples from scikit-image. `sunflower.jpg` is "Sunflower sky backdrop" by Fir0002 (Flagstaffotos), from Wikimedia Commons, used for non-commercial educational purposes under the licence given on its Commons file page; it was downscaled to 1024 px for this project.

## Notes

- I kept Gradio from Lab 1, so there was no framework change.
- Example images are public-domain samples from scikit-image (`astronaut`, `coffee`, `chelsea`, `rocket`).
