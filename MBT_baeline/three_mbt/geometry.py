from __future__ import annotations

import math
from dataclasses import dataclass

from .models import GPSPoint, QueryWindow

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True, slots=True)
class TimedCoord:
    timestamp: float
    latitude: float
    longitude: float


def interpolate_point(a: GPSPoint, b: GPSPoint, t: float) -> TimedCoord:
    if b.timestamp == a.timestamp:
        return TimedCoord(float(t), a.latitude, a.longitude)
    ratio = (t - a.timestamp) / (b.timestamp - a.timestamp)
    return TimedCoord(
        timestamp=float(t),
        latitude=a.latitude + ratio * (b.latitude - a.latitude),
        longitude=a.longitude + ratio * (b.longitude - a.longitude),
    )


def clip_trajectory_to_time(points: list[GPSPoint], start_time: float, end_time: float) -> list[TimedCoord]:
    if not points or start_time > end_time:
        return []
    pts = sorted(points, key=lambda p: (p.timestamp, p.point_id))
    if end_time < pts[0].timestamp or start_time > pts[-1].timestamp:
        return []

    out: list[TimedCoord] = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        if b.timestamp < start_time or a.timestamp > end_time:
            continue
        seg_start = max(start_time, a.timestamp)
        seg_end = min(end_time, b.timestamp)
        if seg_start > seg_end:
            continue
        pa = TimedCoord(a.timestamp, a.latitude, a.longitude) if seg_start == a.timestamp else interpolate_point(a, b, seg_start)
        pb = TimedCoord(b.timestamp, b.latitude, b.longitude) if seg_end == b.timestamp else interpolate_point(a, b, seg_end)
        if not out or out[-1] != pa:
            out.append(pa)
        if not out or out[-1] != pb:
            out.append(pb)

    if len(pts) == 1 and start_time <= pts[0].timestamp <= end_time:
        p = pts[0]
        return [TimedCoord(p.timestamp, p.latitude, p.longitude)]
    return out


def point_in_rect(p: TimedCoord, q: QueryWindow) -> bool:
    return q.min_lon <= p.longitude <= q.max_lon and q.min_lat <= p.latitude <= q.max_lat


def _segment_intersects_rect(a: TimedCoord, b: TimedCoord, q: QueryWindow) -> bool:
    # Liang-Barsky clipping in lon/lat plane.
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


def trajectory_intersects_query(points: list[GPSPoint], query: QueryWindow) -> bool:
    clipped = clip_trajectory_to_time(points, query.start_time, query.end_time)
    if not clipped:
        return False
    if any(point_in_rect(p, query) for p in clipped):
        return True
    for a, b in zip(clipped, clipped[1:]):
        if _segment_intersects_rect(a, b, query):
            return True
    return False


def bbox_from_center_km(latitude: float, longitude: float, side_km: float) -> tuple[float, float, float, float]:
    half_m = side_km * 1000.0 / 2.0
    lat_delta = math.degrees(half_m / EARTH_RADIUS_M)
    cos_lat = max(1e-12, math.cos(math.radians(latitude)))
    lon_delta = math.degrees(half_m / (EARTH_RADIUS_M * cos_lat))
    return longitude - lon_delta, longitude + lon_delta, latitude - lat_delta, latitude + lat_delta
