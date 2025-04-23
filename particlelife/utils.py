"""Utility functions for Particle Life simulation."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def mulberry32(seed: int) -> float:
    """
    A seedable PRNG function (copied from JS implementation).

    # TODO this is a mess and probably not relevant to the python version
    Args:
        seed (int): Random seed

    Returns:
        float: Random value between 0 and 1
    """
    def rng() -> float:
        nonlocal seed
        # Implementation of mulberry32 algorithm
        seed = (seed + 0x6D2B79F5) & 0xFFFFFFFF
        t = seed
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t = ((t ^ (t + ((t ^ (t >> 7)) * (t | 61)))) ^ (t >> 14)) & 0xFFFFFFFF
        return t / 4294967296.0

    return rng


def ensure_dir(directory: str | Path) -> Path:
    """
    Ensure a directory exists.

    Args:
        directory (str or Path): Directory path

    Returns:
        Path: Path object for the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataurl_downloader(file_path: str | Path, file_name: str | None = None) -> bool:
    """
    Handle saving files (similar to JS dataURL_downloader function).

    Args:
        file_path (str or Path): Path to the file
        file_name (str, optional): Name for the file
    """
    try:
        path = Path(file_path)

        if file_name:
            # If a file name is provided, use it
            target_path = path.parent / file_name
        else:
            target_path = path

        # Make sure directory exists
        ensure_dir(target_path.parent)

        # Log the saved file
        logger.info(f"File saved: {target_path}")

        return True
    except Exception as e:
        logger.error(f"Error saving file: {e!s}")
        return False
