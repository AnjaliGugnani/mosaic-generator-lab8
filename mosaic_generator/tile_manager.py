"""Tile loading, caching and feature extraction.

The :class:`TileManager` decodes every tile image exactly once, when it is
created, and keeps three things in memory:

* square thumbnails of every tile (``uint8``, shape ``(N, T, T, 3)``),
* pre-computed features: the measured average colour of each tile and its
  nominal Lab 1 palette colour, and
* an LRU cache of tile stacks already resized to a given cell size.

Lab 1 re-opened and re-resized every multi-megapixel tile photo on every
request; that was the single largest cost in the profile.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from . import config
from .utils import InvalidParameterError, TileLoadError, logger, validate_int


class TileManager:
    """Load tiles once and serve cached, pre-resized tiles and colour features.

    Tiles named after a Lab 1 palette colour (``red.png``, ``skyblue.jpg``, ...)
    are ordered as in Lab 1, so ``tile_count=k`` selects the same first k
    tiles the Lab 1 "Tile Set" slider did. Any other images in the directory
    are appended in alphabetical order, with their measured average colour
    used as their nominal colour.

    Args:
        tile_directory (str | Path | None): Folder containing tile images.
            Defaults to :data:`config.DEFAULT_TILE_DIR`.
        thumbnail_size (int): Side length of the square in-memory thumbnails.
        cache_size (int): Number of resized tile stacks kept in the LRU cache.
        use_draft (bool): Let Pillow decode JPEGs directly at reduced
            resolution (much faster for large photos). Disable only for
            benchmarking.

    Attributes:
        names (list[str]): Tile names (file stems) in selection order.
        paths (list[Path]): Tile file paths in the same order.
        thumbnails (np.ndarray): ``uint8`` array ``(N, T, T, 3)``.
        mean_colors (np.ndarray): Measured average RGB of each tile, ``(N, 3)`` float64.
        palette_colors (np.ndarray): Nominal Lab 1 RGB of each tile, ``(N, 3)`` float64.
        skipped (dict[str, str]): Files that could not be read, with the reason.
        load_time (float): Seconds spent loading tiles at construction.

    Raises:
        TileLoadError: If the directory is missing or contains no readable tiles.

    Example:
        >>> tm = TileManager("tiles/")                 # doctest: +SKIP
        >>> tm.count, tm.get_resized_tiles(16, 16).shape  # doctest: +SKIP
        (13, (13, 16, 16, 3))
    """

    def __init__(
        self,
        tile_directory: str | Path | None = None,
        thumbnail_size: int = config.TILE_THUMBNAIL_SIZE,
        cache_size: int = config.TILE_RESIZE_CACHE_SIZE,
        use_draft: bool = True,
    ) -> None:
        self.tile_directory = Path(tile_directory) if tile_directory else config.DEFAULT_TILE_DIR
        self.thumbnail_size = validate_int("thumbnail_size", thumbnail_size, 8, 2048)
        self.cache_size = validate_int("cache_size", cache_size, 1, 10_000)
        self.use_draft = use_draft
        self.skipped: dict[str, str] = {}
        self._resize_cache: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

        start = time.perf_counter()
        self._load_all()
        self.load_time = time.perf_counter() - start
        logger.info(
            "Loaded %d tiles from %s in %.3fs (skipped %d)",
            self.count, self.tile_directory, self.load_time, len(self.skipped),
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _discover_files(self) -> list[tuple[str, Path, tuple[int, int, int] | None]]:
        """Find tile files and order them: Lab 1 palette order first, then the rest.

        Returns:
            list[tuple[str, Path, tuple[int, int, int] | None]]: ``(name, path,
            nominal colour or None)`` for every candidate tile file.

        Raises:
            TileLoadError: If the tile directory does not exist.
        """
        if not self.tile_directory.is_dir():
            raise TileLoadError(
                f"Tile directory not found: {self.tile_directory}. "
                "Pass tile_directory=... pointing at a folder of tile images."
            )
        files = sorted(
            p for p in self.tile_directory.iterdir()
            if p.is_file() and p.suffix.lower() in config.SUPPORTED_TILE_EXTENSIONS
        )
        by_stem: dict[str, Path] = {}
        for path in files:
            by_stem.setdefault(path.stem.lower(), path)

        ordered: list[tuple[str, Path, tuple[int, int, int] | None]] = []
        palette_stems, missing = set(), []
        for stem, color in config.TILE_PALETTE:
            palette_stems.add(stem)
            if stem in by_stem:
                ordered.append((stem, by_stem[stem], color))
            else:
                missing.append(stem)
        if missing:
            logger.info("Palette tiles not found in %s: %s", self.tile_directory, ", ".join(missing))
        for stem, path in by_stem.items():
            if stem not in palette_stems:
                ordered.append((stem, path, None))
        return ordered

    def _load_thumbnail(self, path: Path) -> np.ndarray:
        """Decode one tile file into a square RGB thumbnail.

        For JPEGs, ``Image.draft`` asks the decoder for a 1/2, 1/4 or 1/8 scale
        image directly from the DCT data, avoiding a full-resolution decode of
        photos that are up to 8000 pixels wide.

        Args:
            path (Path): Tile image file.

        Returns:
            np.ndarray: ``uint8`` array of shape ``(T, T, 3)``.

        Raises:
            OSError | UnidentifiedImageError: If the file cannot be decoded.
        """
        size = (self.thumbnail_size, self.thumbnail_size)
        with Image.open(path) as img:
            if self.use_draft:
                img.draft("RGB", size)
            rgb = img.convert("RGB")
            return np.asarray(rgb.resize(size, Image.Resampling.BICUBIC))

    def _load_all(self) -> None:
        """Load every tile once, skipping unreadable files, and compute features.

        Raises:
            TileLoadError: If no tile could be loaded.
        """
        names, paths, thumbs, nominal = [], [], [], []
        for name, path, color in self._discover_files():  # one pass at startup
            try:
                thumb = self._load_thumbnail(path)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                self.skipped[path.name] = str(exc)
                logger.warning("Skipping unreadable tile %s: %s", path.name, exc)
                continue
            names.append(name)
            paths.append(path)
            thumbs.append(thumb)
            nominal.append(color)

        if not thumbs:
            detail = f" Unreadable files: {sorted(self.skipped)}." if self.skipped else ""
            raise TileLoadError(f"No usable tile images found in {self.tile_directory}.{detail}")

        self.names: list[str] = names
        self.paths: list[Path] = paths
        self.thumbnails: np.ndarray = np.stack(thumbs)
        # Feature extraction, vectorized over all tiles at once.
        self.mean_colors: np.ndarray = self.thumbnails.mean(axis=(1, 2))
        self.palette_colors: np.ndarray = np.array(
            [c if c is not None else self.mean_colors[i] for i, c in enumerate(nominal)],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def count(self) -> int:
        """int: Number of tiles successfully loaded."""
        return len(self.names)

    def _check_count(self, tile_count: int | None) -> int:
        """Validate a requested tile count, defaulting to all tiles.

        Args:
            tile_count (int | None): Number of leading tiles to use.

        Returns:
            int: The validated count.
        """
        if tile_count is None:
            return self.count
        return validate_int("tile_count", tile_count, config.MIN_TILE_COUNT, self.count)

    def get_features(self, mode: str = config.DEFAULT_MATCH_MODE, tile_count: int | None = None) -> np.ndarray:
        """Return the pre-computed representative colours used for matching.

        Args:
            mode (str): ``"palette"`` for the nominal Lab 1 colours or
                ``"tile_mean"`` for the measured average colour of each tile.
            tile_count (int | None): Use only the first ``tile_count`` tiles.

        Returns:
            np.ndarray: Float64 array of shape ``(tile_count, 3)``.

        Raises:
            InvalidParameterError: If ``mode`` or ``tile_count`` is invalid.

        Example:
            >>> tm.get_features("palette", 3)  # doctest: +SKIP
            array([[255., 0., 0.], [0., 255., 0.], [0., 0., 255.]])
        """
        k = self._check_count(tile_count)
        if mode == "palette":
            return self.palette_colors[:k]
        if mode == "tile_mean":
            return self.mean_colors[:k]
        raise InvalidParameterError(f"Unknown match mode {mode!r}; choose from {config.MATCH_MODES}.")

    def resize_tiles(self, cell_h: int, cell_w: int, tile_count: int | None = None) -> np.ndarray:
        """Resize the in-memory thumbnails to a cell size (uncached).

        The loop runs over tiles (at most a few dozen), never over pixels or
        grid cells. Prefer :meth:`get_resized_tiles`, which caches the result.

        Args:
            cell_h (int): Target height in pixels.
            cell_w (int): Target width in pixels.
            tile_count (int | None): Use only the first ``tile_count`` tiles.

        Returns:
            np.ndarray: ``uint8`` array of shape ``(tile_count, cell_h, cell_w, 3)``.
        """
        k = self._check_count(tile_count)
        cell_h = validate_int("cell height", cell_h, 1, 10_000)
        cell_w = validate_int("cell width", cell_w, 1, 10_000)
        return np.stack([
            np.asarray(Image.fromarray(t).resize((cell_w, cell_h), Image.Resampling.BICUBIC))
            for t in self.thumbnails[:k]
        ])

    def get_resized_tiles(self, cell_h: int, cell_w: int, tile_count: int | None = None) -> np.ndarray:
        """Return tiles resized to ``(cell_h, cell_w)``, using an LRU cache.

        The first request for a given cell size resizes the thumbnails; every
        later request for the same size is a dictionary lookup.

        Args:
            cell_h (int): Target height in pixels.
            cell_w (int): Target width in pixels.
            tile_count (int | None): Use only the first ``tile_count`` tiles.

        Returns:
            np.ndarray: Read-only ``uint8`` array ``(tile_count, cell_h, cell_w, 3)``.

        Example:
            >>> tm.get_resized_tiles(16, 16, 9).shape  # doctest: +SKIP
            (9, 16, 16, 3)
        """
        k = self._check_count(tile_count)
        key = (int(cell_h), int(cell_w), k)
        cached = self._resize_cache.get(key)
        if cached is not None:
            self._resize_cache.move_to_end(key)
            self._cache_hits += 1
            return cached
        self._cache_misses += 1
        tiles = self.resize_tiles(cell_h, cell_w, k)
        tiles.setflags(write=False)  # cached arrays are shared; protect them
        self._resize_cache[key] = tiles
        if len(self._resize_cache) > self.cache_size:
            self._resize_cache.popitem(last=False)
        return tiles

    def cache_info(self) -> dict[str, int]:
        """Report resize-cache statistics.

        Returns:
            dict[str, int]: ``{"hits", "misses", "size", "max_size"}``.
        """
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._resize_cache),
            "max_size": self.cache_size,
        }

    def clear_cache(self) -> None:
        """Empty the resized-tile cache (thumbnails and features are kept).

        Returns:
            None
        """
        self._resize_cache.clear()
        self._cache_hits = self._cache_misses = 0

    def __repr__(self) -> str:
        return (
            f"TileManager(tile_directory={str(self.tile_directory)!r}, count={self.count}, "
            f"thumbnail_size={self.thumbnail_size})"
        )
