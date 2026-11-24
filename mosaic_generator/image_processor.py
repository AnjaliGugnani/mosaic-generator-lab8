"""Image loading, preprocessing and grid division.

All grid operations are vectorized: an image is split into cells with a single
``reshape``/``transpose`` (a zero-copy view), and per-cell statistics are
reductions over the cell axes. There are no Python loops over cells or pixels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from . import config
from .utils import InvalidImageError, normalize_grid_size


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file from disk as an RGB ``uint8`` array.

    Args:
        path (str | Path): Path to an image file readable by Pillow.

    Returns:
        np.ndarray: Array of shape ``(H, W, 3)`` and dtype ``uint8``.

    Raises:
        InvalidImageError: If the file does not exist or cannot be decoded.

    Example:
        >>> img = load_image("examples/astronaut_512.png")  # doctest: +SKIP
        >>> img.shape                                        # doctest: +SKIP
        (512, 512, 3)
    """
    path = Path(path)
    if not path.is_file():
        raise InvalidImageError(f"Image file not found: {path}")
    try:
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError(f"Could not read image {path}: {exc}") from exc


def to_rgb_array(image: np.ndarray | Image.Image | str | Path) -> np.ndarray:
    """Convert any supported image input into an RGB ``uint8`` array.

    Grayscale and RGBA inputs are converted to RGB (alpha is dropped, as in
    Lab 1), and float images in ``[0, 1]`` are scaled to ``[0, 255]``. This is
    what makes the reshape trick safe: it requires exactly 3 channels.

    Args:
        image (np.ndarray | PIL.Image.Image | str | Path): A NumPy array of shape
            ``(H, W)``, ``(H, W, 1)``, ``(H, W, 3)`` or ``(H, W, 4)``, a Pillow
            image, or a path to an image file.

    Returns:
        np.ndarray: Contiguous array of shape ``(H, W, 3)`` and dtype ``uint8``.

    Raises:
        InvalidImageError: If the input is ``None``, empty, or has an
            unsupported shape.

    Example:
        >>> to_rgb_array(np.zeros((4, 6), dtype=np.uint8)).shape
        (4, 6, 3)
    """
    if image is None:
        raise InvalidImageError("No image was provided. Please upload an image.")
    if isinstance(image, (str, Path)):
        return load_image(image)
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))

    arr = np.asarray(image)
    if arr.size == 0:
        raise InvalidImageError("The image is empty.")
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif not (arr.ndim == 3 and arr.shape[2] == 3):
        raise InvalidImageError(
            f"Unsupported image shape {arr.shape}; expected (H, W), (H, W, 3) or (H, W, 4)."
        )

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.max(initial=0) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def limit_size(image: np.ndarray, max_dim: int | None = config.MAX_IMAGE_DIM) -> np.ndarray:
    """Downscale an image once, up front, so its longer side is at most ``max_dim``.

    Uses Pillow's ``BOX`` filter, which is the fastest high-quality filter
    for downscaling. Images that are already small enough are returned as is.

    Args:
        image (np.ndarray): RGB ``uint8`` array of shape ``(H, W, 3)``.
        max_dim (int | None): Maximum allowed length of the longer side, or
            ``None`` to disable the limit.

    Returns:
        np.ndarray: The (possibly downscaled) image.

    Example:
        >>> limit_size(np.zeros((4000, 2000, 3), np.uint8), 2048).shape
        (2048, 1024, 3)
    """
    if max_dim is None:
        return image
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return image
    scale = max_dim / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = Image.fromarray(image).resize(new_size, Image.Resampling.BOX)
    return np.asarray(resized)


def crop_to_grid(image: np.ndarray, grid_size: int | tuple[int, int]) -> np.ndarray:
    """Crop the bottom/right edges so the image divides evenly into the grid.

    This matches Lab 1's ``resize_image``: ``H - H % rows`` by ``W - W % cols``.
    The result is a view, so no pixels are copied.

    Args:
        image (np.ndarray): RGB array of shape ``(H, W, 3)``.
        grid_size (int | tuple[int, int]): Cells down and across.

    Returns:
        np.ndarray: View of shape ``(rows * cell_h, cols * cell_w, 3)``.

    Raises:
        InvalidImageError: If the image is smaller than the grid
            (cells would be less than 1 pixel).

    Example:
        >>> crop_to_grid(np.zeros((100, 70, 3), np.uint8), 32).shape
        (96, 64, 3)
    """
    rows, cols = normalize_grid_size(grid_size, 1, 10**6)
    height, width = image.shape[:2]
    if height < rows or width < cols:
        raise InvalidImageError(
            f"Image is {width}x{height} pixels, which is too small for a "
            f"{cols}x{rows} grid. Use a smaller grid or a larger image."
        )
    return image[: height - height % rows, : width - width % cols]


def split_into_cells(image: np.ndarray, grid_size: int | tuple[int, int]) -> np.ndarray:
    """Divide an image into grid cells with a single reshape (no loops, no copy).

    Precondition: the image is RGB and its sides are exact multiples of the
    grid size (call :func:`crop_to_grid` first).

    Args:
        image (np.ndarray): Array of shape ``(rows * cell_h, cols * cell_w, 3)``.
        grid_size (int | tuple[int, int]): Cells down and across.

    Returns:
        np.ndarray: View of shape ``(rows, cols, cell_h, cell_w, 3)``, where
        ``cells[i, j]`` is the cell in grid row ``i`` and column ``j``.

    Raises:
        InvalidImageError: If the image does not divide evenly into the grid.

    Example:
        >>> split_into_cells(np.zeros((64, 64, 3), np.uint8), 4).shape
        (4, 4, 16, 16, 3)
    """
    rows, cols = normalize_grid_size(grid_size, 1, 10**6)
    height, width = image.shape[:2]
    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError(f"Expected an RGB image (H, W, 3), got shape {image.shape}.")
    if height % rows or width % cols:
        raise InvalidImageError(
            f"Image size {width}x{height} is not a multiple of the {cols}x{rows} grid; "
            "call crop_to_grid() first."
        )
    cell_h, cell_w = height // rows, width // cols
    return image.reshape(rows, cell_h, cols, cell_w, 3).transpose(0, 2, 1, 3, 4)


def cell_mean_colors(image: np.ndarray, grid_size: int | tuple[int, int]) -> np.ndarray:
    """Compute the average RGB colour of every grid cell with vectorized reductions.

    Sums each cell with two ``np.add.reduceat`` passes (rows, then columns) in
    exact ``uint32`` integer arithmetic and divides once. This is about 3x
    faster than Lab 1's ``grid.mean(axis=(1, 3))`` on a 1024x1024 image and
    gives bit-identical float64 results, because integer sums are exact.

    Args:
        image (np.ndarray): Array of shape ``(rows * cell_h, cols * cell_w, 3)``.
        grid_size (int | tuple[int, int]): Cells down and across.

    Returns:
        np.ndarray: Float64 array of shape ``(rows, cols, 3)``.

    Example:
        >>> img = np.full((32, 32, 3), 10, np.uint8)
        >>> cell_mean_colors(img, 4)[0, 0]
        array([10., 10., 10.])
    """
    rows, cols = normalize_grid_size(grid_size, 1, 10**6)
    height, width = image.shape[:2]
    if height % rows or width % cols:
        raise InvalidImageError("Image does not divide evenly into the grid; crop it first.")
    cell_h, cell_w = height // rows, width // cols
    row_sums = np.add.reduceat(image, np.arange(0, height, cell_h), axis=0, dtype=np.uint32)
    cell_sums = np.add.reduceat(row_sums, np.arange(0, width, cell_w), axis=1, dtype=np.uint32)
    return cell_sums / float(cell_h * cell_w)


def expand_cells(cell_values: np.ndarray, cell_h: int, cell_w: int) -> np.ndarray:
    """Paint each cell's value over its whole cell to build a full-size image.

    Used for the "segmented" preview (each cell filled with its average
    colour). Uses ``np.broadcast_to`` plus one reshape instead of two
    ``np.repeat`` calls, so only the final image is allocated.

    Args:
        cell_values (np.ndarray): Array of shape ``(rows, cols, 3)``.
        cell_h (int): Cell height in pixels.
        cell_w (int): Cell width in pixels.

    Returns:
        np.ndarray: ``uint8`` array of shape ``(rows * cell_h, cols * cell_w, 3)``.
        Floats are truncated toward zero, matching Lab 1's ``astype(np.uint8)``.

    Example:
        >>> expand_cells(np.ones((2, 2, 3)), 3, 3).shape
        (6, 6, 3)
    """
    rows, cols = cell_values.shape[:2]
    values = cell_values.astype(np.uint8)
    painted = np.broadcast_to(values[:, None, :, None, :], (rows, cell_h, cols, cell_w, 3))
    return painted.reshape(rows * cell_h, cols * cell_w, 3)


def assemble_from_cells(cells: np.ndarray) -> np.ndarray:
    """Stitch a ``(rows, cols, cell_h, cell_w, 3)`` cell array back into one image.

    This is the inverse of :func:`split_into_cells`: one transpose and one
    reshape.

    Args:
        cells (np.ndarray): Array of shape ``(rows, cols, cell_h, cell_w, 3)``.

    Returns:
        np.ndarray: Array of shape ``(rows * cell_h, cols * cell_w, 3)``.

    Example:
        >>> img = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        >>> bool((assemble_from_cells(split_into_cells(img, 2)) == img).all())
        True
    """
    rows, cols, cell_h, cell_w, channels = cells.shape
    return cells.transpose(0, 2, 1, 3, 4).reshape(rows * cell_h, cols * cell_w, channels)


def prepare_image(
    image: np.ndarray | Image.Image | str | Path,
    grid_size: int | tuple[int, int],
    max_dim: int | None = config.MAX_IMAGE_DIM,
) -> np.ndarray:
    """Run the full preprocessing pipeline: to RGB, limit size once, crop to grid.

    Args:
        image (np.ndarray | PIL.Image.Image | str | Path): Input image.
        grid_size (int | tuple[int, int]): Cells down and across.
        max_dim (int | None): Longest allowed side before downscaling.

    Returns:
        np.ndarray: RGB ``uint8`` image whose sides are multiples of the grid.

    Raises:
        InvalidImageError: If the image is missing, malformed or too small.

    Example:
        >>> prepare_image(np.zeros((100, 70, 3), np.uint8), 32).shape
        (96, 64, 3)
    """
    rgb = to_rgb_array(image)
    rgb = limit_size(rgb, max_dim)
    return crop_to_grid(rgb, grid_size)
