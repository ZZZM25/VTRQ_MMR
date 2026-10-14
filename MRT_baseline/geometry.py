from __future__ import annotations

from dataclasses import dataclass

from .models import QueryWindow


@dataclass(frozen=True, slots=True)
class TimedCoord:
    timestamp: float
    latitude: float
    longitude: float


def bbox_intersects(bbox: tuple[float, float, float, float], q: QueryWindow) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    return not (
        max_lon < q.min_lon or min_lon > q.max_lon or max_lat < q.min_lat or min_lat > q.max_lat
    )


def bbox_union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    )


def point_in_query(lon: float, lat: float, q: QueryWindow) -> bool:
    return q.min_lon <= lon <= q.max_lon and q.min_lat <= lat <= q.max_lat


def point_matches_query(lon: float, lat: float, ts: float, q: QueryWindow) -> bool:
    return point_in_query(lon, lat, q) and q.start_time <= ts <= q.end_time


def interpolate(a: TimedCoord, b: TimedCoord, t: float) -> TimedCoord:
    if b.timestamp == a.timestamp:
        return TimedCoord(float(t), a.latitude, a.longitude)
    r = (t - a.timestamp) / (b.timestamp - a.timestamp)
    return TimedCoord(
        float(t),
        a.latitude + r * (b.latitude - a.latitude),
        a.longitude + r * (b.longitude - a.longitude),
    )


def clip_trajectory_to_time(
    points: list[tuple[float, float, float]], start_time: float, end_time: float
) -> list[TimedCoord]:
    if not points or start_time > end_time:
        return []
    pts = [TimedCoord(float(t), float(lat), float(lon)) for t, lat, lon in points]
    if end_time < pts[0].timestamp or start_time > pts[-1].timestamp:
        return []
    if len(pts) == 1:
        p = pts[0]
        return [p] if start_time <= p.timestamp <= end_time else []

    out: list[TimedCoord] = []
    for a, b in zip(pts, pts[1:]):
        if b.timestamp < start_time or a.timestamp > end_time:
            continue
        seg_start = max(start_time, a.timestamp)
        seg_end = min(end_time, b.timestamp)
        if seg_start > seg_end:
            continue
        pa = a if seg_start == a.timestamp else interpolate(a, b, seg_start)
        pb = b if seg_end == b.timestamp else interpolate(a, b, seg_end)
        if not out or out[-1] != pa:
            out.append(pa)
        if not out or out[-1] != pb:
            out.append(pb)
    return out


def _point_in_rect(p: TimedCoord, q: QueryWindow) -> bool:
    return q.min_lon <= p.longitude <= q.max_lon and q.min_lat <= p.latitude <= q.max_lat


def _segment_intersects_rect(a: TimedCoord, b: TimedCoord, q: QueryWindow) -> bool:
    x0, y0 = a.longitude, a.latitude
    x1, y1 = b.longitude, b.latitude
    dx, dy = x1 - x0, y1 - y0
    p = [-dx, dx, -dy, dy]
    r = [x0 - q.min_lon, q.max_lon - x0, y0 - q.min_lat, q.max_lat - y0]
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, r):
        if pi == 0.0:
            if qi < 0.0:
                return False
            continue
        t = qi / pi
        if pi < 0.0:
            if t > u2:
                return False
            u1 = max(u1, t)
        else:
            if t < u1:
                return False
            u2 = min(u2, t)
    return u1 <= u2


def trajectory_intersects_query(points: list[tuple[float, float, float]], q: QueryWindow) -> bool:
    clipped = clip_trajectory_to_time(points, q.start_time, q.end_time)
    if not clipped:
        return False
    if any(_point_in_rect(p, q) for p in clipped):
        return True
    return any(_segment_intersects_rect(a, b, q) for a, b in zip(clipped, clipped[1:]))
