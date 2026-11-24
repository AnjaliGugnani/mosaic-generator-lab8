"""Helper functions shared by the other modules: errors, validation and timing."""
from __future__ import annotations

import logging
import statistics
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("mosaic_generator")


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
class MosaicError(ValueError):
    """Base class for every error raised deliberately by this package.

    It subclasses :class:`ValueError` so callers that already catch bad
    input as ``ValueError`` keep working.
    """


class InvalidImageError(MosaicError):
    """The input image is missing, empty, or has an unsupported shape/dtype."""


class InvalidParameterError(MosaicError):
    """A numeric parameter (grid size, tile count, ...) is out of range."""


class TileLoadError(MosaicError):
    """No usable tiles could be loaded from the tile directory."""


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_int(name: str, value: Any, minimum: int, maximum: int) -> int:
    """Check that ``value`` is an integer within ``[minimum, maximum]``.

    Args:
        name (str): Parameter name, used in the error message.
        value (Any): The value to check. Integral floats such as ``32.0``
            (which UI sliders often send) are accepted.
        minimum (int): Smallest allowed value (inclusive).
        maximum (int): Largest allowed value (inclusive).

    Returns:
        int: ``value`` converted to ``int``.

    Raises:
        InvalidParameterError: If ``value`` is not integral or out of range.

    Example:
        >>> validate_int("grid_size", 32.0, 4, 256)
        32
    """
    if isinstance(value, bool):
        raise InvalidParameterError(f"{name} must be an integer, got a boolean.")
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        raise InvalidParameterError(f"{name} must be an integer, got {value!r}.") from None
    if not as_float.is_integer():
        raise InvalidParameterError(f"{name} must be a whole number, got {value!r}.")
    as_int = int(as_float)
    if not minimum <= as_int <= maximum:
        raise InvalidParameterError(
            f"{name} must be between {minimum} and {maximum}, got {as_int}."
        )
    return as_int


def normalize_grid_size(grid_size: int | tuple[int, int], minimum: int, maximum: int) -> tuple[int, int]:
    """Turn a grid size given as ``n`` or ``(rows, cols)`` into a validated tuple.

    Args:
        grid_size (int | tuple[int, int]): Cells down and across. A single int
            means a square grid.
        minimum (int): Smallest allowed number of cells per side.
        maximum (int): Largest allowed number of cells per side.

    Returns:
        tuple[int, int]: ``(rows, cols)``.

    Raises:
        InvalidParameterError: If the value has the wrong form or is out of range.

    Example:
        >>> normalize_grid_size(32, 4, 256)
        (32, 32)
    """
    if isinstance(grid_size, (tuple, list)):
        if len(grid_size) != 2:
            raise InvalidParameterError(
                f"grid_size must be an int or a (rows, cols) pair, got {grid_size!r}."
            )
        rows, cols = grid_size
    else:
        rows = cols = grid_size
    return (
        validate_int("grid rows", rows, minimum, maximum),
        validate_int("grid cols", cols, minimum, maximum),
    )


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------
@contextmanager
def timed(timings: dict[str, float], key: str) -> Iterator[None]:
    """Context manager that adds the elapsed wall time of its block to ``timings[key]``.

    Args:
        timings (dict[str, float]): Dictionary that collects stage timings (seconds).
        key (str): Name of the stage being timed.

    Yields:
        None

    Example:
        >>> t = {}
        >>> with timed(t, "work"):
        ...     _ = sum(range(10))
        >>> "work" in t
        True
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = timings.get(key, 0.0) + time.perf_counter() - start


def benchmark(func: Callable[[], Any], repeats: int = 3, warmup: int = 0) -> dict[str, Any]:
    """Time ``func()`` several times and summarise the results.

    Args:
        func (Callable[[], Any]): Zero-argument callable to time.
        repeats (int): Number of timed runs (at least 1).
        warmup (int): Untimed runs executed first (e.g. to fill caches).

    Returns:
        dict[str, Any]: ``{"mean", "std", "min", "runs"}`` in seconds; ``runs``
        is the list of individual timings.

    Example:
        >>> stats = benchmark(lambda: sum(range(1000)), repeats=3)
        >>> sorted(stats)
        ['mean', 'min', 'runs', 'std']
    """
    if repeats < 1:
        raise InvalidParameterError("repeats must be at least 1.")
    for _ in range(warmup):
        func()
    runs = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        runs.append(time.perf_counter() - start)
    return {
        "mean": statistics.fmean(runs),
        "std": statistics.stdev(runs) if len(runs) > 1 else 0.0,
        "min": min(runs),
        "runs": runs,
    }


def format_seconds(seconds: float) -> str:
    """Format a duration for display, choosing ms or s automatically.

    Args:
        seconds (float): Duration in seconds.

    Returns:
        str: For example ``"12.3 ms"`` or ``"3.40 s"``.

    Example:
        >>> format_seconds(0.0123)
        '12.3 ms'
    """
    if seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.2f} s"
