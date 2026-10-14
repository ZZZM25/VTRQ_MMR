from __future__ import annotations

import numpy as np


def _part1by1(v: np.ndarray) -> np.ndarray:
    x = v.astype(np.uint64, copy=False) & np.uint64(0x00000000FFFFFFFF)
    x = (x | (x << np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
    x = (x | (x << np.uint64(8))) & np.uint64(0x00FF00FF00FF00FF)
    x = (x | (x << np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = (x | (x << np.uint64(2))) & np.uint64(0x3333333333333333)
    x = (x | (x << np.uint64(1))) & np.uint64(0x5555555555555555)
    return x


def morton_keys(lons: np.ndarray, lats: np.ndarray, bounds) -> np.ndarray:
    min_lon, min_lat, max_lon, max_lat = [float(x) for x in bounds]
    lon_span = max(max_lon - min_lon, 1e-15)
    lat_span = max(max_lat - min_lat, 1e-15)
    scale = float((1 << 32) - 1)
    x = np.clip((lons - min_lon) / lon_span, 0.0, 1.0)
    y = np.clip((lats - min_lat) / lat_span, 0.0, 1.0)
    xi = np.floor(x * scale).astype(np.uint64)
    yi = np.floor(y * scale).astype(np.uint64)
    return _part1by1(xi) | (_part1by1(yi) << np.uint64(1))


def morton_key_scalar(lon: float, lat: float, bounds) -> int:
    arr = morton_keys(np.array([lon], dtype=np.float64), np.array([lat], dtype=np.float64), bounds)
    return int(arr[0])
