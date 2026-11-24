"""mosaic_generator: fast, modular photo-mosaic generation (Lab 8).

Quick start::

    from mosaic_generator import MosaicBuilder, TileManager

    tile_manager = TileManager(tile_directory="tiles/")   # loads tiles once
    builder = MosaicBuilder(tile_manager, grid_size=(32, 32))
    mosaic = builder.create_mosaic(input_image)
    similarity = builder.compute_similarity(input_image, mosaic)
"""
from .mosaic_builder import (
    BruteForceMatcher,
    ColorMatcher,
    KDTreeMatcher,
    MosaicBuilder,
    MosaicResult,
    make_matcher,
)
from .tile_manager import TileManager
from .utils import InvalidImageError, InvalidParameterError, MosaicError, TileLoadError

__version__ = "2.0.0"

__all__ = [
    "MosaicBuilder",
    "TileManager",
    "MosaicResult",
    "ColorMatcher",
    "BruteForceMatcher",
    "KDTreeMatcher",
    "make_matcher",
    "MosaicError",
    "InvalidImageError",
    "InvalidParameterError",
    "TileLoadError",
    "__version__",
]
