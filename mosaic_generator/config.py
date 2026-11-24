"""Configuration constants for the mosaic generator.

Every tunable value lives here so that no module hard-codes sizes, paths or
thresholds. Import the module and read the constants, e.g.::

    from mosaic_generator import config
    config.DEFAULT_GRID_SIZE  # -> 32
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
#: Project root (the folder that contains ``mosaic_generator/`` and ``app.py``).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

#: Default tile directory used by the web app (small, deployment-friendly tiles).
DEFAULT_TILE_DIR: Path = PROJECT_ROOT / "tiles"

#: Directory with example input images shown in the web app.
EXAMPLES_DIR: Path = PROJECT_ROOT / "examples"

#: File extensions accepted as tile images.
SUPPORTED_TILE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# --------------------------------------------------------------------------
# Grid / image parameters
# --------------------------------------------------------------------------
#: Default number of cells across and down (a 32x32 grid -> 1,024 cells).
DEFAULT_GRID_SIZE: int = 32
#: Smallest and largest grid size accepted by the builder.
MIN_GRID_SIZE: int = 4
MAX_GRID_SIZE: int = 256

#: Inputs whose longer side exceeds this are downscaled once, up front.
#: Set to ``None`` to disable the limit.
MAX_IMAGE_DIM: int | None = 2048

#: Tighter limit used by the web app: the free hosting tier has a small share of
#: one CPU and little memory, so large uploads are downscaled to this first.
APP_MAX_IMAGE_DIM: int = 1024

#: Example images shown in the web app (one per scene).
APP_EXAMPLE_FILES: tuple[str, ...] = (
    "sunflower.jpg", "astronaut_512.png", "chelsea_cat.jpg", "coffee.jpg", "rocket.jpg",
)

# --------------------------------------------------------------------------
# Tile parameters
# --------------------------------------------------------------------------
#: The Lab 1 colour palette, in Lab 1 order. Each entry is
#: ``(file stem, nominal RGB colour)``. "tile_count = k" selects the first k
#: entries, exactly as the Lab 1 "Tile Set" slider did.
TILE_PALETTE: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("red", (255, 0, 0)),
    ("green", (0, 255, 0)),
    ("blue", (0, 0, 255)),
    ("yellow", (255, 255, 0)),
    ("orange", (255, 165, 0)),
    ("purple", (128, 0, 128)),
    ("white", (255, 255, 255)),
    ("gray", (128, 128, 128)),
    ("black", (0, 0, 0)),
    ("brown", (139, 69, 19)),
    ("beige", (245, 245, 220)),
    ("pink", (255, 182, 193)),
    ("skyblue", (135, 206, 235)),
)

#: Default and allowed number of tiles (Lab 1 default was 9).
DEFAULT_TILE_COUNT: int = 9
MIN_TILE_COUNT: int = 1

#: Tiles are decoded once at startup and stored as square thumbnails of this
#: size. Every cell size is then produced from these in-memory thumbnails.
TILE_THUMBNAIL_SIZE: int = 256

#: Number of (cell size, tile count) resized tile stacks kept in the LRU cache.
TILE_RESIZE_CACHE_SIZE: int = 32

# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
#: ``"palette"`` matches cells against the Lab 1 nominal palette colours
#: (reproduces Lab 1 output). ``"tile_mean"`` matches against the measured
#: average colour of each tile image (usually a closer mosaic).
MATCH_MODES: tuple[str, ...] = ("palette", "tile_mean")
DEFAULT_MATCH_MODE: str = "palette"

#: Matcher selection: ``"bruteforce"``, ``"kdtree"`` or ``"auto"``.
MATCHER_METHODS: tuple[str, ...] = ("auto", "bruteforce", "kdtree")
DEFAULT_MATCHER: str = "auto"
#: With ``"auto"``, a k-d tree is used once there are more tiles than this.
#: Measured (profiling_analysis.ipynb, section 2.5): brute force wins for very small tile
#: sets; the crossover is around 9-13 tiles depending on the machine. Output is identical.
KDTREE_THRESHOLD: int = 8

# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
#: Pixel value range of uint8 images (used by SSIM / PSNR).
PIXEL_DATA_RANGE: float = 255.0
