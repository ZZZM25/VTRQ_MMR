from __future__ import annotations

import json
import math
from typing import Iterable

_COMPACT_SEPARATORS = (",", ":")


def json_string_size(value: str) -> int:
    """Exact UTF-8 byte size of one JSON string using stdlib defaults."""
    # Fast path covers project keys, SHA-256 trajectory IDs, dimensions, hashes,
    # and other ordinary ASCII strings without JSON escapes.
    if value.isascii() and value.isprintable() and '"' not in value and "\\" not in value:
        return len(value) + 2
    return len(
        json.dumps(value, ensure_ascii=True, separators=_COMPACT_SEPARATORS).encode("utf-8")
    )


def json_number_size(value: int | float) -> int:
    """Exact compact-JSON byte size for the finite numeric values used here."""
    if isinstance(value, bool):
        return 4 if value else 5
    if isinstance(value, int):
        return len(str(value))
    x = float(value)
    if math.isfinite(x):
        # CPython's JSON encoder uses float.__repr__ for finite floats.
        return len(repr(x))
    return len(json.dumps(x, separators=_COMPACT_SEPARATORS))


def json_array_size(item_sizes: Iterable[int]) -> int:
    total = 2  # []
    count = 0
    for size in item_sizes:
        total += int(size)
        count += 1
    if count > 1:
        total += count - 1  # commas
    return total


def json_object_size(items: Iterable[tuple[str, int]]) -> int:
    total = 2  # {}
    count = 0
    for key, value_size in items:
        total += json_string_size(key) + 1 + int(value_size)  # key:value
        count += 1
    if count > 1:
        total += count - 1  # commas
    return total


def json_literal_size(value: object) -> int:
    if value is None:
        return 4
    if value is True:
        return 4
    if value is False:
        return 5
    if isinstance(value, str):
        return json_string_size(value)
    if isinstance(value, (int, float)):
        return json_number_size(value)
    raise TypeError(f"unsupported JSON literal for size accounting: {type(value)!r}")


def proof_bundle_size(
    *,
    dimension: str,
    low: float,
    high: float,
    root_size: int,
) -> int:
    return json_object_size(
        [
            ("version", json_number_size(1)),
            ("dimension", json_string_size(dimension)),
            ("low", json_number_size(float(low))),
            ("high", json_number_size(float(high))),
            ("root", int(root_size)),
        ]
    )


def empty_proof_size(dimension: str) -> int:
    return json_object_size(
        [
            ("type", json_string_size("empty")),
            ("dimension", json_string_size(dimension)),
        ]
    )


def composite_three_mbt_vo_size(
    *,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    start_time: int,
    end_time: int,
    trusted_global_root_hex: str,
    vo_lon_size: int,
    vo_lat_size: int,
    vo_time_size: int,
) -> int:
    query_size = json_object_size(
        [
            ("min_lon", json_number_size(float(min_lon))),
            ("max_lon", json_number_size(float(max_lon))),
            ("min_lat", json_number_size(float(min_lat))),
            ("max_lat", json_number_size(float(max_lat))),
            ("start_time", json_number_size(int(start_time))),
            ("end_time", json_number_size(int(end_time))),
        ]
    )
    return json_object_size(
        [
            ("version", json_number_size(2)),
            ("query", query_size),
            ("trusted_global_root", json_string_size(trusted_global_root_hex)),
            ("vo_lon", int(vo_lon_size)),
            ("vo_lat", int(vo_lat_size)),
            ("vo_time", int(vo_time_size)),
        ]
    )
