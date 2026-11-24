"""Similarity metrics between images, and agreement between tile assignments."""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

from . import config
from .utils import InvalidImageError, InvalidParameterError


def _check_same_shape(a: np.ndarray, b: np.ndarray) -> None:
    """Raise if two images do not have the same shape.

    Args:
        a (np.ndarray): First image.
        b (np.ndarray): Second image.

    Raises:
        InvalidImageError: If the shapes differ.
    """
    if a.shape != b.shape:
        raise InvalidImageError(f"Images must have the same shape, got {a.shape} and {b.shape}.")


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two images (lower is more similar).

    Args:
        a (np.ndarray): First image, any numeric dtype.
        b (np.ndarray): Second image with the same shape.

    Returns:
        float: Mean of the squared per-pixel, per-channel differences.

    Example:
        >>> mse(np.zeros((2, 2, 3)), np.ones((2, 2, 3)))
        1.0
    """
    _check_same_shape(a, b)
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diff * diff))


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = config.PIXEL_DATA_RANGE) -> float:
    """Peak signal-to-noise ratio in decibels (higher is more similar).

    Args:
        a (np.ndarray): First image.
        b (np.ndarray): Second image with the same shape.
        data_range (float): Maximum possible pixel value (255 for uint8).

    Returns:
        float: PSNR in dB, or ``inf`` for identical images.

    Example:
        >>> round(psnr(np.zeros((2, 2)), np.full((2, 2), 255.0)), 1)
        0.0
    """
    error = mse(a, b)
    if error == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range**2 / error))


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = config.PIXEL_DATA_RANGE) -> float:
    """Structural similarity index (scikit-image), in ``[-1, 1]``; 1 means identical.

    Args:
        a (np.ndarray): First image, ``(H, W)`` or ``(H, W, 3)``.
        b (np.ndarray): Second image with the same shape.
        data_range (float): Maximum possible pixel value (255 for uint8).

    Returns:
        float: Mean SSIM over the image (channels averaged for colour images).

    Raises:
        InvalidImageError: If the shapes differ or the image is under 7x7
            pixels (SSIM's default window size).

    Example:
        >>> img = np.random.default_rng(0).integers(0, 255, (16, 16, 3)).astype(np.uint8)
        >>> ssim(img, img)
        1.0
    """
    _check_same_shape(a, b)
    if min(a.shape[:2]) < 7:
        raise InvalidImageError("SSIM needs images of at least 7x7 pixels.")
    channel_axis = 2 if a.ndim == 3 else None
    return float(structural_similarity(a, b, channel_axis=channel_axis, data_range=data_range))


#: Metric name -> function, used by :func:`compute_metrics`.
METRICS = {"mse": mse, "psnr": psnr, "ssim": ssim}


def compute_metrics(a: np.ndarray, b: np.ndarray, names: tuple[str, ...] = ("mse", "ssim")) -> dict[str, float]:
    """Compute several similarity metrics at once.

    Args:
        a (np.ndarray): Reference image.
        b (np.ndarray): Image to compare (same shape).
        names (tuple[str, ...]): Any of ``"mse"``, ``"psnr"``, ``"ssim"``.

    Returns:
        dict[str, float]: Metric name -> value.

    Raises:
        InvalidParameterError: If an unknown metric name is requested.

    Example:
        >>> compute_metrics(np.zeros((8, 8, 3)), np.zeros((8, 8, 3)), ("mse",))
        {'mse': 0.0}
    """
    unknown = set(names) - METRICS.keys()
    if unknown:
        raise InvalidParameterError(f"Unknown metric(s) {sorted(unknown)}; choose from {sorted(METRICS)}.")
    return {name: METRICS[name](a, b) for name in names}


def tile_agreement(indices_a: np.ndarray, indices_b: np.ndarray) -> float:
    """Percentage of grid cells that were assigned the same tile in two mosaics.

    Used as the correctness check between the baseline and optimized code.

    Args:
        indices_a (np.ndarray): Tile index per cell, shape ``(rows, cols)``.
        indices_b (np.ndarray): Tile index per cell, same shape.

    Returns:
        float: Agreement in percent, ``0`` to ``100``.

    Example:
        >>> tile_agreement(np.array([[0, 1], [2, 3]]), np.array([[0, 1], [2, 0]]))
        75.0
    """
    if np.shape(indices_a) != np.shape(indices_b):
        raise InvalidParameterError(
            f"Index grids must match, got {np.shape(indices_a)} and {np.shape(indices_b)}."
        )
    return float(np.mean(np.asarray(indices_a) == np.asarray(indices_b)) * 100.0)
