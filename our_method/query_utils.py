from __future__ import annotations


UINT32_MAX = (1 << 32) - 1


def normalize_rectangle(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> tuple[float, float, float, float]:
    if min_lon > max_lon:
        min_lon, max_lon = max_lon, min_lon

    if min_lat > max_lat:
        min_lat, max_lat = max_lat, min_lat

    return min_lon, min_lat, max_lon, max_lat


def rectangle_overlap(
    min_lon_a: float,
    min_lat_a: float,
    max_lon_a: float,
    max_lat_a: float,
    min_lon_b: float,
    min_lat_b: float,
    max_lon_b: float,
    max_lat_b: float,
) -> bool:
    return not (
        max_lon_a < min_lon_b
        or max_lon_b < min_lon_a
        or max_lat_a < min_lat_b
        or max_lat_b < min_lat_a
    )


def segment_intersects_rectangle(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> bool:
    min_lon, min_lat, max_lon, max_lat = normalize_rectangle(
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    )

    dx = x2 - x1
    dy = y2 - y1

    p = (-dx, dx, -dy, dy)
    q = (
        x1 - min_lon,
        max_lon - x1,
        y1 - min_lat,
        max_lat - y1,
    )

    u1 = 0.0
    u2 = 1.0

    for pi, qi in zip(p, q):
        if pi == 0.0:
            if qi < 0.0:
                return False
            continue

        t = qi / pi

        if pi < 0.0:
            if t > u2:
                return False
            if t > u1:
                u1 = t
        else:
            if t < u1:
                return False
            if t < u2:
                u2 = t

    return u1 <= u2


def edge_intersects_query(
    eid: int,
    road,
    query_min_lon: float,
    query_min_lat: float,
    query_max_lon: float,
    query_max_lat: float,
) -> bool:
    query_min_lon, query_min_lat, query_max_lon, query_max_lat = (
        normalize_rectangle(
            query_min_lon,
            query_min_lat,
            query_max_lon,
            query_max_lat,
        )
    )

    if not road.edge_exists(eid):
        return False

    u, v = road.get_edge_nodes(eid)

    x1 = road.node_lon[u]
    y1 = road.node_lat[u]
    x2 = road.node_lon[v]
    y2 = road.node_lat[v]

    edge_min_lon = min(x1, x2)
    edge_min_lat = min(y1, y2)
    edge_max_lon = max(x1, x2)
    edge_max_lat = max(y1, y2)

    if not rectangle_overlap(
        edge_min_lon,
        edge_min_lat,
        edge_max_lon,
        edge_max_lat,
        query_min_lon,
        query_min_lat,
        query_max_lon,
        query_max_lat,
    ):
        return False

    return segment_intersects_rectangle(
        x1,
        y1,
        x2,
        y2,
        query_min_lon,
        query_min_lat,
        query_max_lon,
        query_max_lat,
    )


def is_empty_time_range(
    min_start: int,
    max_end: int,
) -> bool:
    return (
        min_start == UINT32_MAX
        and max_end == 0
    )
