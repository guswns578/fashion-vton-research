"""Audit realized GarmentCode pattern geometry against flat garment targets."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


Point = tuple[float, float]


def _distance(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _relative_point(start: Point, end: Point, relative: Iterable[float]) -> Point:
    rx, ry = relative
    dx, dy = end[0] - start[0], end[1] - start[1]
    return start[0] + rx * dx - ry * dy, start[1] + rx * dy + ry * dx


def _cubic_point(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    u = 1.0 - t
    return (
        u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0],
        u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1],
    )


def edge_point(panel: dict, edge_index: int, t: float) -> Point:
    if not 0.0 <= t <= 1.0:
        raise ValueError("t must be in [0, 1]")
    edge = panel["edges"][edge_index]
    start = tuple(panel["vertices"][edge["endpoints"][0]])
    end = tuple(panel["vertices"][edge["endpoints"][1]])
    curvature = edge.get("curvature")
    if not curvature:
        return start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1])
    if isinstance(curvature, list) or curvature["type"] == "quadratic":
        relative = curvature if isinstance(curvature, list) else curvature["params"][0]
        control = _relative_point(start, end, relative)
        return (
            (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t**2 * end[0],
            (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t**2 * end[1],
        )
    if curvature["type"] == "cubic":
        controls = [_relative_point(start, end, point) for point in curvature["params"]]
        return _cubic_point(start, controls[0], controls[1], end, t)
    if curvature["type"] == "circle":
        radius, large_arc, right = curvature["params"]
        dx, dy = end[0] - start[0], end[1] - start[1]
        chord = math.hypot(dx, dy)
        if chord == 0 or chord > 2.0 * radius + 1e-9:
            raise ValueError("Invalid circular arc geometry")
        midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
        height = math.sqrt(max(radius**2 - (chord / 2.0) ** 2, 0.0))
        perpendicular = (-dy / chord, dx / chord)
        direction = -1.0 if right else 1.0
        chosen = None
        for sign in (-1.0, 1.0):
            center = (
                midpoint[0] + sign * height * perpendicular[0],
                midpoint[1] + sign * height * perpendicular[1],
            )
            start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
            end_angle = math.atan2(end[1] - center[1], end[0] - center[0])
            delta = (end_angle - start_angle) % (2.0 * math.pi) if direction > 0 else (
                start_angle - end_angle
            ) % (2.0 * math.pi)
            if bool(delta > math.pi) == bool(large_arc):
                chosen = center, start_angle, delta
                break
        if chosen is None:
            raise ValueError("Could not resolve circular arc direction")
        center, start_angle, delta = chosen
        angle = start_angle + direction * delta * t
        return center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)
    raise ValueError(f"Unsupported edge curvature: {curvature['type']}")


def edge_point_by_length_fraction(
    panel: dict,
    edge_index: int,
    fraction: float,
    samples: int = 512,
) -> Point:
    """Sample a curve by normalized arc length instead of Bezier parameter time."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    if samples < 2:
        raise ValueError("samples must be at least 2")
    edge = panel["edges"][edge_index]
    curvature = edge.get("curvature")
    if not curvature or (not isinstance(curvature, list) and curvature["type"] == "circle"):
        return edge_point(panel, edge_index, fraction)
    points = [edge_point(panel, edge_index, index / samples) for index in range(samples + 1)]
    cumulative = [0.0]
    for left, right in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + _distance(left, right))
    target = fraction * cumulative[-1]
    upper = next((index for index, value in enumerate(cumulative) if value >= target), samples)
    if upper == 0:
        return points[0]
    span = cumulative[upper] - cumulative[upper - 1]
    local = 0.0 if span <= 1e-12 else (target - cumulative[upper - 1]) / span
    return edge_point(panel, edge_index, (upper - 1 + local) / samples)


def edge_length(panel: dict, edge_index: int, samples: int = 512) -> float:
    edge = panel["edges"][edge_index]
    start = tuple(panel["vertices"][edge["endpoints"][0]])
    end = tuple(panel["vertices"][edge["endpoints"][1]])
    curvature = edge.get("curvature")
    if not curvature:
        return _distance(start, end)
    if isinstance(curvature, list) or curvature["type"] == "quadratic":
        relative = curvature if isinstance(curvature, list) else curvature["params"][0]
        control = _relative_point(start, end, relative)
        previous = start
        length = 0.0
        for index in range(1, samples + 1):
            t = index / samples
            point = (
                (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t**2 * end[0],
                (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t**2 * end[1],
            )
            length += _distance(previous, point)
            previous = point
        return length
    if curvature["type"] == "cubic":
        controls = [_relative_point(start, end, point) for point in curvature["params"]]
        previous = start
        length = 0.0
        for index in range(1, samples + 1):
            point = _cubic_point(start, controls[0], controls[1], end, index / samples)
            length += _distance(previous, point)
            previous = point
        return length
    if curvature["type"] == "circle":
        radius, large_arc, _ = curvature["params"]
        chord = min(_distance(start, end), 2.0 * radius)
        angle = 2.0 * math.asin(chord / (2.0 * radius))
        if large_arc:
            angle = 2.0 * math.pi - angle
        return radius * angle
    raise ValueError(f"Unsupported edge curvature: {curvature['type']}")


def _underarm_x(panel: dict) -> float:
    edge_index = next(
        index for index, edge in enumerate(panel["edges"]) if "armhole" in edge.get("label", "")
    )
    points = [panel["vertices"][index] for index in panel["edges"][edge_index]["endpoints"]]
    return float(min(points, key=lambda point: point[1])[0])


def _shoulder_outer_x(panel: dict) -> float:
    edge_index = next(
        index for index, edge in enumerate(panel["edges"]) if "armhole" in edge.get("label", "")
    )
    points = [panel["vertices"][index] for index in panel["edges"][edge_index]["endpoints"]]
    return float(max(points, key=lambda point: point[1])[0])


def _vertical_span(panel: dict) -> float:
    ys = [float(point[1]) for point in panel["vertices"]]
    return max(ys) - min(ys)


def realized_measurements(pattern: dict) -> dict[str, float]:
    panels = pattern["panels"]
    front = panels["left_ftorso"]
    back = panels["left_btorso"]
    sleeve_front = panels["left_sleeve_f"]
    sleeve_back = panels["left_sleeve_b"]

    chest_flat = _underarm_x(front) + _underarm_x(back)
    hem_flat = max(point[0] for point in front["vertices"] if abs(point[1]) < 1e-6) + max(
        point[0] for point in back["vertices"] if abs(point[1]) < 1e-6
    )
    stitched_edges: dict[str, set[int]] = {}
    sleeve_pair_lengths = []
    for stitch in pattern.get("stitches", []):
        for side in stitch[:2]:
            stitched_edges.setdefault(side["panel"], set()).add(side["edge"])
        if all(side["panel"].startswith("left_sleeve_") for side in stitch[:2]):
            sleeve_pair_lengths.append(
                sum(edge_length(panels[side["panel"]], side["edge"]) for side in stitch[:2]) / 2.0
            )
    cuff_circumference = 0.0
    for panel_name in ("left_sleeve_f", "left_sleeve_b"):
        panel = panels[panel_name]
        unstitched = set(range(len(panel["edges"]))) - stitched_edges.get(panel_name, set())
        if len(unstitched) != 1:
            raise ValueError(f"Expected one cuff edge on {panel_name}, got {sorted(unstitched)}")
        cuff_circumference += edge_length(panel, unstitched.pop())
    shoulder_extension = min(sleeve_pair_lengths) if len(sleeve_pair_lengths) >= 3 else 0.0
    return {
        "shoulder": 2.0 * _shoulder_outer_x(back) + 2.0 * shoulder_extension,
        "chest_flat": chest_flat,
        "body_length": _vertical_span(back),
        "sleeve_length": max(abs(point[0]) for point in sleeve_front["vertices"]),
        "cuff_flat": cuff_circumference / 2.0,
        "hem_flat": hem_flat,
    }


def audit(spec_path: Path, targets: dict[str, float]) -> dict:
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    pattern = raw["pattern"]
    realized = realized_measurements(pattern)
    comparisons = {}
    for name, value in realized.items():
        if name not in targets:
            continue
        error = value - targets[name]
        comparisons[name] = {
            "target_cm": targets[name],
            "realized_cm": value,
            "error_cm": error,
            "relative_error": error / targets[name] if targets[name] else None,
        }

    stitch_pairs = []
    for index, stitch in enumerate(pattern.get("stitches", [])):
        sides = []
        for side in stitch[:2]:
            length = edge_length(pattern["panels"][side["panel"]], side["edge"])
            sides.append({**side, "length_cm": length})
        mismatch = abs(sides[0]["length_cm"] - sides[1]["length_cm"])
        stitch_pairs.append(
            {
                "stitch": index,
                "sides": sides,
                "mismatch_cm": mismatch,
                "mismatch_ratio": mismatch / max(sides[0]["length_cm"], sides[1]["length_cm"], 1e-9),
            }
        )

    return {
        "schema_version": 1,
        "specification": str(spec_path.resolve()),
        "measurements": comparisons,
        "max_abs_measurement_error_cm": max(abs(item["error_cm"]) for item in comparisons.values()),
        "stitch_summary": {
            "count": len(stitch_pairs),
            "max_mismatch_cm": max(item["mismatch_cm"] for item in stitch_pairs),
            "max_mismatch_ratio": max(item["mismatch_ratio"] for item in stitch_pairs),
        },
        "stitches": stitch_pairs,
        "limitations": [
            "Shoulder uses the back torso shoulder endpoint and excludes optional sleeve shoulder extensions.",
            "Armhole and upper-arm product measurements need an explicit store measurement convention before auditing.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    targets_raw = json.loads(args.targets.read_text(encoding="utf-8"))
    targets = targets_raw.get("measurements_cm", targets_raw)
    report = audit(args.spec, {key: float(value) for key, value in targets.items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
