"""Compress data if it exceeds threshold."""

__all__ = ["compress_if_large"]

import gzip


def compress_if_large(
    data: bytes,
    threshold: int = 1_000_000,
    compression_level: int = 6,
) -> tuple[bytes, bool]:
    """
    Compress data with gzip if it exceeds size threshold.

    Returns:
        Tuple of (data, was_compressed)
    """
    if len(data) < threshold:
        return data, False

    compressed = gzip.compress(data, compresslevel=compression_level)

    # Only use compressed if it's actually smaller
    if len(compressed) < len(data):
        return compressed, True

    return data, False
