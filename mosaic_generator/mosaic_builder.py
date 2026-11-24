"""Main mosaic construction logic.

Pipeline (every step is a whole-array operation; there is no loop over cells):

1. ``prepare_image``     convert to RGB, downscale once if huge, crop to the grid
2. ``cell_mean_colors``  one reshape + mean gives every cell's average colour
3. ``ColorMatcher``      nearest tile colour for all cells at once
                         (broadcasted distances, or a k-d tree for big tile sets)
4. ``get_resized_tiles`` cached tile stack for this cell size (LRU)
5. fancy indexing        ``tiles[indices]`` picks a tile for every cell at once
6. ``assemble_from_cells`` one transpose + reshape stitches the mosaic
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from . import config
from .image_processor import assemble_from_cells, cell_mean_colors, expand_cells, prepare_image
from .metrics import compute_metrics
from .tile_manager import TileManager
from .utils import InvalidParameterError, normalize_grid_size, timed, validate_int


# --------------------------------------------------------------------------
# Matching strategies (Strategy pattern)
# --------------------------------------------------------------------------
class ColorMatcher:
    """Interface for nearest-colour matchers.

    A matcher is built once from the tile feature colours and can then match
    any number of cell colours. Subclasses implement :meth:`match`.

    Args:
        features (np.ndarray): Tile colours, shape ``(K, 3)``.
    """

    name = "base"

    def __init__(self, features: np.ndarray) -> None:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != 3 or len(features) == 0:
            raise InvalidParameterError(f"features must have shape (K, 3) with K >= 1, got {features.shape}.")
        self.features = features

    def match(self, colors: np.ndarray) -> np.ndarray:
        """Return the index of the closest feature colour for every input colour.

        Args:
            colors (np.ndarray): Colours of shape ``(..., 3)``.

        Returns:
            np.ndarray: Integer indices of shape ``colors.shape[:-1]``.
        """
        raise NotImplementedError


class BruteForceMatcher(ColorMatcher):
    """Exact nearest colour by broadcasting all cell-to-tile squared distances.

    Computes an ``(N, K)`` distance matrix in one vectorized expression. This
    is the same arithmetic as Lab 1, so the chosen tiles are identical. It is
    the fastest choice for very small tile sets (below roughly 9-13 tiles).

    Example:
        >>> BruteForceMatcher(np.array([[0, 0, 0], [255, 255, 255]])).match(np.array([[10, 10, 10]]))
        array([0])
    """

    name = "bruteforce"

    def match(self, colors: np.ndarray) -> np.ndarray:
        """See :meth:`ColorMatcher.match`."""
        colors = np.asarray(colors, dtype=np.float64)
        diff = colors[..., None, :] - self.features  # (..., K, 3)
        distances = np.sum(diff * diff, axis=-1)     # (..., K)
        return np.argmin(distances, axis=-1)


class KDTreeMatcher(ColorMatcher):
    """Nearest colour via a ``scipy.spatial.cKDTree`` built once over the tile colours.

    Each query costs about O(log K) instead of O(K). Measured on this project
    it overtakes brute force at roughly 9-13 tiles (machine-dependent) and is
    over 100x faster for thousands of tiles. Exact ties may be broken differently from
    :class:`BruteForceMatcher` (none occurred in the correctness tests).

    Example:
        >>> KDTreeMatcher(np.array([[0, 0, 0], [255, 255, 255]])).match(np.array([[250, 250, 250]]))
        array([1])
    """

    name = "kdtree"

    def __init__(self, features: np.ndarray) -> None:
        super().__init__(features)
        self.tree = cKDTree(self.features)

    def match(self, colors: np.ndarray) -> np.ndarray:
        """See :meth:`ColorMatcher.match`."""
        colors = np.asarray(colors, dtype=np.float64)
        _, idx = self.tree.query(colors.reshape(-1, 3), k=1)
        return idx.reshape(colors.shape[:-1])


def make_matcher(features: np.ndarray, method: str = config.DEFAULT_MATCHER) -> ColorMatcher:
    """Factory that builds the right :class:`ColorMatcher` for a tile set.

    Args:
        features (np.ndarray): Tile colours, shape ``(K, 3)``.
        method (str): ``"bruteforce"``, ``"kdtree"`` or ``"auto"`` (k-d tree
            only when K exceeds :data:`config.KDTREE_THRESHOLD`).

    Returns:
        ColorMatcher: A ready-to-use matcher.

    Raises:
        InvalidParameterError: If ``method`` is unknown.

    Example:
        >>> make_matcher(np.zeros((5, 3))).name
        'bruteforce'
        >>> make_matcher(np.zeros((13, 3))).name
        'kdtree'
    """
    if method == "auto":
        method = "kdtree" if len(features) > config.KDTREE_THRESHOLD else "bruteforce"
    if method == "bruteforce":
        return BruteForceMatcher(features)
    if method == "kdtree":
        return KDTreeMatcher(features)
    raise InvalidParameterError(f"Unknown matcher {method!r}; choose from {config.MATCHER_METHODS}.")


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------
@dataclass
class MosaicResult:
    """Everything produced by one call to :meth:`MosaicBuilder.build`.

    Attributes:
        mosaic (np.ndarray): The mosaic, ``uint8`` ``(H, W, 3)``.
        processed (np.ndarray): The preprocessed input the mosaic was built
            from (same shape as ``mosaic``); compare metrics against this.
        tile_indices (np.ndarray): Chosen tile per cell, ``(rows, cols)``.
        mean_colors (np.ndarray): Average colour per cell, ``(rows, cols, 3)``.
        segmented (np.ndarray | None): Each cell filled with its average colour.
        grid_size (tuple[int, int]): ``(rows, cols)``.
        cell_size (tuple[int, int]): ``(cell_h, cell_w)`` in pixels.
        tile_names (list[str]): Names of the tiles that were available.
        timings (dict[str, float]): Seconds spent in each pipeline stage.
    """

    mosaic: np.ndarray
    processed: np.ndarray
    tile_indices: np.ndarray
    mean_colors: np.ndarray
    segmented: np.ndarray | None
    grid_size: tuple[int, int]
    cell_size: tuple[int, int]
    tile_names: list[str]
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def total_time(self) -> float:
        """float: Total seconds across all timed stages."""
        return float(sum(self.timings.values()))

    @property
    def tile_usage(self) -> dict[str, int]:
        """dict[str, int]: How many cells use each tile (unused tiles omitted)."""
        counts = np.bincount(self.tile_indices.ravel(), minlength=len(self.tile_names))
        return {name: int(n) for name, n in zip(self.tile_names, counts) if n}


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------
class MosaicBuilder:
    """Build photo mosaics from a shared :class:`TileManager`.

    The builder is cheap to create: all expensive state (decoded tiles,
    features, resized-tile cache) lives in the ``TileManager``, which should be
    created once per process and shared. Matchers are cached per
    ``(match_on, tile_count, method)``.

    Args:
        tile_manager (TileManager): Loaded tiles to build from.
        grid_size (int | tuple[int, int]): Cells down and across.
        tile_count (int | None): Use the first ``tile_count`` tiles
            (Lab 1 "Tile Set"); ``None`` uses every loaded tile.
        match_on (str): ``"palette"`` (Lab 1 behaviour) or ``"tile_mean"``.
        matcher (str): ``"auto"``, ``"bruteforce"`` or ``"kdtree"``.
        max_image_dim (int | None): Inputs larger than this are downscaled once.

    Raises:
        InvalidParameterError: If any parameter is out of range.

    Example:
        >>> builder = MosaicBuilder(TileManager("tiles/"), grid_size=(32, 32))  # doctest: +SKIP
        >>> mosaic = builder.create_mosaic(input_image)                         # doctest: +SKIP
        >>> builder.compute_similarity(input_image, mosaic)                     # doctest: +SKIP
        {'mse': 2210.4, 'ssim': 0.41}
    """

    def __init__(
        self,
        tile_manager: TileManager,
        grid_size: int | tuple[int, int] = config.DEFAULT_GRID_SIZE,
        tile_count: int | None = None,
        match_on: str = config.DEFAULT_MATCH_MODE,
        matcher: str = config.DEFAULT_MATCHER,
        max_image_dim: int | None = config.MAX_IMAGE_DIM,
    ) -> None:
        if not isinstance(tile_manager, TileManager):
            raise InvalidParameterError("tile_manager must be a TileManager instance.")
        if match_on not in config.MATCH_MODES:
            raise InvalidParameterError(f"match_on must be one of {config.MATCH_MODES}, got {match_on!r}.")
        if matcher not in config.MATCHER_METHODS:
            raise InvalidParameterError(f"matcher must be one of {config.MATCHER_METHODS}, got {matcher!r}.")
        self.tile_manager = tile_manager
        self.grid_size = normalize_grid_size(grid_size, config.MIN_GRID_SIZE, config.MAX_GRID_SIZE)
        self.tile_count = (
            tile_manager.count if tile_count is None
            else validate_int("tile_count", tile_count, config.MIN_TILE_COUNT, tile_manager.count)
        )
        self.match_on = match_on
        self.matcher_method = matcher
        self.max_image_dim = max_image_dim
        self._matchers: dict[tuple[str, int, str], ColorMatcher] = {}

    def _get_matcher(self) -> ColorMatcher:
        """Return the cached matcher for the current settings, building it if needed.

        Returns:
            ColorMatcher: Matcher over the selected tiles' feature colours.
        """
        key = (self.match_on, self.tile_count, self.matcher_method)
        if key not in self._matchers:
            features = self.tile_manager.get_features(self.match_on, self.tile_count)
            self._matchers[key] = make_matcher(features, self.matcher_method)
        return self._matchers[key]

    def build(self, image, make_segmented: bool = True) -> MosaicResult:
        """Build a mosaic and return it with intermediate results and timings.

        Args:
            image (np.ndarray | PIL.Image.Image | str | Path): Input image.
            make_segmented (bool): Also build the average-colour preview.

        Returns:
            MosaicResult: Mosaic, preprocessed input, tile indices, timings, ...

        Raises:
            InvalidImageError: If the image is missing, malformed or smaller
                than the grid.
        """
        timings: dict[str, float] = {}
        rows, cols = self.grid_size

        with timed(timings, "preprocess"):
            processed = prepare_image(image, self.grid_size, self.max_image_dim)
            cell_h, cell_w = processed.shape[0] // rows, processed.shape[1] // cols

        with timed(timings, "cell_colors"):
            mean_colors = cell_mean_colors(processed, self.grid_size)

        with timed(timings, "matching"):
            indices = self._get_matcher().match(mean_colors)

        with timed(timings, "tile_lookup"):
            tiles = self.tile_manager.get_resized_tiles(cell_h, cell_w, self.tile_count)

        with timed(timings, "assembly"):
            mosaic = assemble_from_cells(tiles[indices])  # (rows, cols, ch, cw, 3) -> image

        segmented = None
        if make_segmented:
            with timed(timings, "segmented"):
                segmented = expand_cells(mean_colors, cell_h, cell_w)

        return MosaicResult(
            mosaic=mosaic,
            processed=processed,
            tile_indices=indices,
            mean_colors=mean_colors,
            segmented=segmented,
            grid_size=(rows, cols),
            cell_size=(cell_h, cell_w),
            tile_names=self.tile_manager.names[: self.tile_count],
            timings=timings,
        )

    def create_mosaic(self, image) -> np.ndarray:
        """Build a mosaic and return only the image.

        Args:
            image (np.ndarray | PIL.Image.Image | str | Path): Input image.

        Returns:
            np.ndarray: ``uint8`` mosaic of shape ``(H', W', 3)``, where the
            input was cropped (and, if huge, downscaled) to fit the grid.

        Example:
            >>> mosaic = builder.create_mosaic(np.asarray(Image.open("photo.jpg")))  # doctest: +SKIP
        """
        return self.build(image, make_segmented=False).mosaic

    def compute_similarity(
        self, original, mosaic: np.ndarray, metrics: tuple[str, ...] = ("mse", "ssim")
    ) -> dict[str, float]:
        """Compare a mosaic with its source image.

        The original is preprocessed exactly as in :meth:`build` (RGB, size
        limit, crop) so both images have the same shape.

        Args:
            original (np.ndarray | PIL.Image.Image | str | Path): The input image.
            mosaic (np.ndarray): Mosaic produced from it by this builder.
            metrics (tuple[str, ...]): Any of ``"mse"``, ``"psnr"``, ``"ssim"``.

        Returns:
            dict[str, float]: Metric name -> value.

        Example:
            >>> builder.compute_similarity(img, builder.create_mosaic(img))  # doctest: +SKIP
            {'mse': 2210.4, 'ssim': 0.41}
        """
        reference = prepare_image(original, self.grid_size, self.max_image_dim)
        return compute_metrics(reference, np.asarray(mosaic), metrics)

    @staticmethod
    def to_pil(array: np.ndarray) -> Image.Image:
        """Convert a result array to a Pillow image (for saving or display).

        Args:
            array (np.ndarray): ``uint8`` image array.

        Returns:
            PIL.Image.Image: The same image as a Pillow object.
        """
        return Image.fromarray(np.ascontiguousarray(array))

    def __repr__(self) -> str:
        return (
            f"MosaicBuilder(grid_size={self.grid_size}, tile_count={self.tile_count}, "
            f"match_on={self.match_on!r}, matcher={self.matcher_method!r})"
        )
