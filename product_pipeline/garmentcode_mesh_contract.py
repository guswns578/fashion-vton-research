"""Convert a GarmentCode sewing specification into a Blender cloth mesh contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay
from scipy.sparse.linalg import spsolve

from garmentcode_pattern_audit import edge_length, edge_point, edge_point_by_length_fraction
from measurement_geometry import load_pilot_geometry


def _point_on_segment(point, start, end, tolerance=1e-7) -> bool:
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    edge = end - start
    offset = point - start
    cross = abs(float(edge[0] * offset[1] - edge[1] * offset[0]))
    if cross > tolerance * max(float(np.linalg.norm(edge)), 1.0):
        return False
    dot = float(np.dot(offset, edge))
    return -tolerance <= dot <= float(np.dot(edge, edge)) + tolerance


def _point_segment_distance(point, start, end) -> float:
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    edge = end - start
    denominator = float(np.dot(edge, edge))
    if denominator <= 1e-18:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, edge) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * edge)))


def point_in_polygon(point, polygon) -> bool:
    if any(
        _point_on_segment(point, left, right)
        for left, right in zip(polygon, polygon[1:] + polygon[:1])
    ):
        return True
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x0, y0 = previous
        x1, y1 = current
        if (y0 > y) != (y1 > y):
            crossing = (x1 - x0) * (y - y0) / (y1 - y0) + x0
            if x < crossing:
                inside = not inside
        previous = current
    return inside


def _rotation_matrix_xyz(degrees) -> np.ndarray:
    x, y, z = np.radians(np.asarray(degrees, dtype=float))
    rx = np.asarray([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    ry = np.asarray([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.asarray([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _to_blender(
    point_2d,
    panel,
    vertical_shift_m: float,
    depth_override_m: float | None = None,
) -> list[float]:
    local = np.asarray([point_2d[0], point_2d[1], 0.0])
    world_cm = _rotation_matrix_xyz(panel["rotation"]) @ local + np.asarray(panel["translation"])
    depth_m = float(-world_cm[2] / 100.0) if depth_override_m is None else depth_override_m
    return [float(world_cm[0] / 100.0), depth_m, float(world_cm[1] / 100.0 + vertical_shift_m)]


def edge_segment_counts(pattern: dict, resolution_cm: float) -> dict[tuple[str, int], int]:
    counts = {}
    for panel_name, panel in pattern["panels"].items():
        for edge_index in range(len(panel["edges"])):
            counts[(panel_name, edge_index)] = max(
                2, math.ceil(edge_length(panel, edge_index) / resolution_cm)
            )
    for stitch in pattern.get("stitches", []):
        left, right = stitch[:2]
        left_key = (left["panel"], left["edge"])
        right_key = (right["panel"], right["edge"])
        common = max(counts[left_key], counts[right_key])
        counts[left_key] = counts[right_key] = common
    return counts


def preclose_vertices(
    vertices, sewing_edges, maximum_pair_distance_m: float | None = None
) -> list[list[float]]:
    """Move every stitched vertex group to one start point without welding topology."""
    closed = np.asarray(vertices, dtype=float).copy()
    active_edges = [
        pair
        for pair in sewing_edges
        if maximum_pair_distance_m is None
        or np.linalg.norm(closed[pair[0]] - closed[pair[1]]) <= maximum_pair_distance_m
    ]
    parent = {index: index for pair in active_edges for index in pair}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in active_edges:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
    groups: dict[int, list[int]] = {}
    for index in parent:
        groups.setdefault(find(index), []).append(index)
    for group in groups.values():
        mean = closed[group].mean(axis=0)
        closed[group] = mean
    return closed.tolist()


def harmonic_preclose_vertices(vertices, faces, components, sewing_edges) -> list[list[float]]:
    """Close all seams and distribute boundary motion smoothly through each panel."""
    start = np.asarray(vertices, dtype=float)
    parent = {index: index for pair in sewing_edges for index in pair}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in sewing_edges:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
    groups: dict[int, list[int]] = {}
    for index in parent:
        groups.setdefault(find(index), []).append(index)
    targets = {}
    for group in groups.values():
        target = start[group].mean(axis=0)
        for index in group:
            targets[index] = target

    result = start.copy()
    adjacency = [set() for _ in range(len(start))]
    for face in faces:
        for left, right in zip(face, face[1:] + face[:1]):
            adjacency[left].add(right)
            adjacency[right].add(left)
    for begin, end in components.values():
        constrained = sorted(index for index in range(begin, end) if index in targets)
        free = sorted(set(range(begin, end)).difference(constrained))
        if not constrained or not free:
            continue
        free_lookup = {index: row for row, index in enumerate(free)}
        rows, columns, values = [], [], []
        rhs = np.zeros((len(free), 3), dtype=float)
        for row, index in enumerate(free):
            neighbors = adjacency[index]
            rows.append(row)
            columns.append(row)
            values.append(float(len(neighbors)))
            for neighbor in neighbors:
                if neighbor in free_lookup:
                    rows.append(row)
                    columns.append(free_lookup[neighbor])
                    values.append(-1.0)
                elif neighbor in targets:
                    rhs[row] += targets[neighbor] - start[neighbor]
        matrix = sparse.csr_matrix((values, (rows, columns)), shape=(len(free), len(free)))
        displacement = np.column_stack([spsolve(matrix, rhs[:, axis]) for axis in range(3)])
        result[free] += displacement
        for index in constrained:
            result[index] = targets[index]
    return result.tolist()


def structural_edge_ratios(vertices, rest_vertices, faces) -> np.ndarray:
    edges = set()
    for face in faces:
        for left, right in zip(face, face[1:] + face[:1]):
            edges.add(tuple(sorted((left, right))))
    edge_array = np.asarray(sorted(edges), dtype=int)
    current = np.asarray(vertices, dtype=float)
    rest = np.asarray(rest_vertices, dtype=float)
    rest_lengths = np.linalg.norm(rest[edge_array[:, 1]] - rest[edge_array[:, 0]], axis=1)
    current_lengths = np.linalg.norm(current[edge_array[:, 1]] - current[edge_array[:, 0]], axis=1)
    return current_lengths / np.maximum(rest_lengths, 1e-12)


def relax_preclosed_vertices(
    vertices,
    rest_vertices,
    faces,
    sewing_edges,
    iterations: int = 1600,
    relaxation: float = 0.8,
    tether: float = 1e-5,
) -> list[list[float]]:
    """Project pre-drape edges toward pattern lengths while keeping seams closed."""
    if iterations < 1 or not 0.0 < relaxation <= 1.0 or not 0.0 <= tether < 1.0:
        raise ValueError("Invalid pre-drape relaxation settings")
    anchor = np.asarray(vertices, dtype=float)
    rest = np.asarray(rest_vertices, dtype=float)
    result = anchor.copy()
    edges = set()
    for face in faces:
        for left, right in zip(face, face[1:] + face[:1]):
            edges.add(tuple(sorted((left, right))))
    edge_array = np.asarray(sorted(edges), dtype=int)
    rest_lengths = np.linalg.norm(rest[edge_array[:, 1]] - rest[edge_array[:, 0]], axis=1)

    parent = {index: index for pair in sewing_edges for index in pair}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in sewing_edges:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
    groups: dict[int, list[int]] = {}
    for index in parent:
        groups.setdefault(find(index), []).append(index)

    for _ in range(iterations):
        difference = result[edge_array[:, 1]] - result[edge_array[:, 0]]
        lengths = np.linalg.norm(difference, axis=1)
        correction = (0.5 * relaxation * (lengths - rest_lengths) / np.maximum(lengths, 1e-12))[
            :, None
        ] * difference
        delta = np.zeros_like(result)
        degree = np.zeros(len(result), dtype=float)
        np.add.at(delta, edge_array[:, 0], correction)
        np.add.at(delta, edge_array[:, 1], -correction)
        np.add.at(degree, edge_array[:, 0], 1.0)
        np.add.at(degree, edge_array[:, 1], 1.0)
        result += delta / np.maximum(degree[:, None], 1.0)
        result += (anchor - result) * tether
        for group in groups.values():
            result[group] = result[group].mean(axis=0)
    return result.tolist()


def weld_mesh_topology(
    vertices,
    faces,
    components,
    attachment_vertices,
    sewing_edges,
    rest_vertices=None,
    center_attachments: bool = False,
):
    parent = {index: index for pair in sewing_edges for index in pair}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in sewing_edges:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
    groups: dict[tuple[str, int], list[int]] = {}
    for index in range(len(vertices)):
        key = ("seam", find(index)) if index in parent else ("single", index)
        groups.setdefault(key, []).append(index)
    attachment_set = set(attachment_vertices)
    remap = {}
    welded_vertices = []
    welded_rest = [] if rest_vertices is not None else None
    for group in groups.values():
        mean = np.mean([vertices[index] for index in group], axis=0)
        if center_attachments and attachment_set.intersection(group):
            mean[1] = 0.0
        new_index = len(welded_vertices)
        welded_vertices.append(mean.tolist())
        if welded_rest is not None:
            welded_rest.append(np.mean([rest_vertices[index] for index in group], axis=0).tolist())
        for index in group:
            remap[index] = new_index
    welded_faces = []
    seen_faces = set()
    for face in faces:
        welded = tuple(remap[index] for index in face)
        if len(set(welded)) != 3:
            continue
        canonical = tuple(sorted(welded))
        if canonical in seen_faces:
            continue
        seen_faces.add(canonical)
        welded_faces.append(list(welded))
    welded_components = {}
    for name, (start, end) in components.items():
        welded_components[name] = {
            "vertex_indices": sorted({remap[index] for index in range(start, end)})
        }
    return (
        welded_vertices,
        welded_rest,
        welded_faces,
        welded_components,
        sorted({remap[index] for index in attachment_vertices}),
    )


def _ellipse_circumference(radius_x: float, radius_y: float) -> float:
    h = ((radius_x - radius_y) / (radius_x + radius_y)) ** 2
    return math.pi * (radius_x + radius_y) * (1.0 + 3.0 * h / (10.0 + math.sqrt(4.0 - 3.0 * h)))


def _ellipse_theta(radius_x: float, radius_y: float, fraction: float) -> float:
    angles = np.linspace(0.0, math.pi, 2001)
    dx = radius_x * np.cos(angles)
    dy = radius_y * np.sin(angles)
    increments = np.sqrt(dx[:-1] ** 2 + dy[:-1] ** 2) * np.diff(angles)
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    return float(np.interp(np.clip(fraction, 0.0, 1.0), cumulative / cumulative[-1], angles))


def _panel_cross_section(panel: dict, x: float) -> tuple[float, float]:
    boundary = []
    for edge_index in range(len(panel["edges"])):
        boundary.extend(edge_point(panel, edge_index, step / 40.0) for step in range(40))
    intersections = []
    for left, right in zip(boundary, boundary[1:] + boundary[:1]):
        x0, y0 = left
        x1, y1 = right
        if abs(x1 - x0) < 1e-9:
            if abs(x - x0) < 1e-6:
                intersections.extend((y0, y1))
            continue
        t = (x - x0) / (x1 - x0)
        if -1e-7 <= t <= 1.0 + 1e-7:
            intersections.append(y0 + t * (y1 - y0))
    if len(intersections) < 2:
        nearest = sorted(boundary, key=lambda point: abs(point[0] - x))[:8]
        intersections = [point[1] for point in nearest]
    return float(min(intersections)), float(max(intersections))


def body_surface_start_vertices(
    pattern: dict,
    local_points_by_panel: dict[str, list],
    rest_vertices: list[list[float]],
    components: dict[str, list[int]],
    body,
    vertical_shift_m: float,
) -> list[list[float]]:
    """Create a low-strain cylindrical pre-drape while keeping panel topology separate."""
    start = [list(vertex) for vertex in rest_vertices]
    torso_widths = {}
    for side in ("right", "left"):
        for region in ("f", "b"):
            panel = pattern["panels"][f"{side}_{region}torso"]
            torso_widths[(side, region)] = max(abs(point[0]) for point in panel["vertices"])
    garment_circumference_m = (
        2.0
        * (
            np.mean([torso_widths[(side, "f")] for side in ("right", "left")])
            + np.mean([torso_widths[(side, "b")] for side in ("right", "left")])
        )
        / 100.0
    )
    scale = garment_circumference_m / _ellipse_circumference(
        body.torso_radius_x_m, body.torso_radius_y_m
    )
    radius_x = body.torso_radius_x_m * scale
    radius_y = body.torso_radius_y_m * scale

    for panel_name, points in local_points_by_panel.items():
        begin, end = components[panel_name]
        panel = pattern["panels"][panel_name]
        side_sign = -1.0 if panel_name.startswith("right_") else 1.0
        if "torso" in panel_name:
            front_width = torso_widths[("right" if side_sign < 0 else "left", "f")]
            back_width = torso_widths[("right" if side_sign < 0 else "left", "b")]
            total = front_width + back_width
            for index, point in enumerate(points, start=begin):
                distance = (
                    abs(point[0])
                    if "ftorso" in panel_name
                    else front_width + back_width - abs(point[0])
                )
                theta = _ellipse_theta(radius_x, radius_y, distance / total)
                start[index][0] = side_sign * radius_x * math.sin(theta)
                start[index][1] = -radius_y * math.cos(theta)
            continue

        counterpart_name = (
            panel_name.replace("sleeve_f", "sleeve_b")
            if "sleeve_f" in panel_name
            else panel_name.replace("sleeve_b", "sleeve_f")
        )
        counterpart = pattern["panels"][counterpart_name]
        is_front = "sleeve_f" in panel_name
        for index, point in enumerate(points, start=begin):
            x, y = point
            low, high = _panel_cross_section(panel, x)
            other_low, other_high = _panel_cross_section(counterpart, x)
            width = max(high - low, 1e-6)
            other_width = max(other_high - other_low, 1e-6)
            total_width = width + other_width
            u = float(np.clip((high - y) / width, 0.0, 1.0))
            own_angle = 2.0 * math.pi * width / total_width
            theta = -u * own_angle if is_front else u * (2.0 * math.pi - own_angle)
            center_y = 0.5 * (high + 0.5 * (low + other_low))
            axis = _to_blender((x, center_y), panel, vertical_shift_m, 0.0)
            radius = total_width / 100.0 / (2.0 * math.pi)
            start[index] = [axis[0], radius * math.sin(theta), axis[2] + radius * math.cos(theta)]
    return start


def _panel_mesh(panel: dict, panel_name: str, counts: dict, resolution_cm: float):
    boundary = []
    edge_local_ids = {}
    for edge_index in range(len(panel["edges"])):
        count = counts[(panel_name, edge_index)]
        ids = []
        for segment in range(count):
            ids.append(len(boundary))
            boundary.append(edge_point_by_length_fraction(panel, edge_index, segment / count))
        edge_local_ids[edge_index] = ids
    for edge_index in range(len(panel["edges"])):
        next_edge = (edge_index + 1) % len(panel["edges"])
        edge_local_ids[edge_index] = edge_local_ids[edge_index] + [edge_local_ids[next_edge][0]]

    min_x, min_y = np.min(boundary, axis=0)
    max_x, max_y = np.max(boundary, axis=0)
    points = list(boundary)
    for y in np.arange(min_y + resolution_cm, max_y, resolution_cm):
        for x in np.arange(min_x + resolution_cm, max_x, resolution_cm):
            point = (float(x), float(y))
            boundary_distance = min(
                _point_segment_distance(point, left, right)
                for left, right in zip(boundary, boundary[1:] + boundary[:1])
            )
            if point_in_polygon(point, boundary) and boundary_distance >= 0.2 * resolution_cm:
                points.append((float(x), float(y)))
    array = np.asarray(points, dtype=float)
    triangulation = Delaunay(array)
    faces = []
    for raw_face in triangulation.simplices:
        triangle = [int(value) for value in raw_face]
        vertices = array[triangle]
        first = vertices[1] - vertices[0]
        second = vertices[2] - vertices[0]
        signed_area = float(first[0] * second[1] - first[1] * second[0])
        if abs(signed_area) <= resolution_cm**2 * 1e-6:
            continue
        checks = [vertices.mean(axis=0)]
        checks.extend((vertices[index] + vertices[(index + 1) % 3]) * 0.5 for index in range(3))
        if not all(point_in_polygon(tuple(point), boundary) for point in checks):
            continue
        if signed_area < 0:
            triangle[1], triangle[2] = triangle[2], triangle[1]
        faces.append(triangle)

    face_edge_counts: dict[tuple[int, int], int] = {}
    for face in faces:
        for index in range(3):
            edge = tuple(sorted((face[index], face[(index + 1) % 3])))
            face_edge_counts[edge] = face_edge_counts.get(edge, 0) + 1
    face_edges = set(face_edge_counts)
    missing = []
    overused = []
    for edge_index, ids in edge_local_ids.items():
        for left, right in zip(ids, ids[1:]):
            edge = tuple(sorted((left, right)))
            if edge not in face_edges:
                missing.append([edge_index, left, right])
            elif face_edge_counts[edge] != 1:
                overused.append([edge_index, left, right, face_edge_counts[edge]])
    if missing:
        raise ValueError(
            f"Constrained boundary lost on {panel_name}: {len(missing)} segments; {missing[:5]}"
        )
    if overused:
        raise ValueError(f"Boundary is non-manifold on {panel_name}: {overused[:5]}")
    return points, faces, edge_local_ids


def build_contract(
    spec_path: Path,
    resolution_cm: float,
    shoulder_z_m: float,
    torso_panel_offset_m: float | None = None,
    arm_panel_offset_m: float | None = None,
    weld_seams: bool = True,
    preclose_seams: bool = False,
    weld_relaxed_preclose: bool = False,
    body=None,
) -> dict:
    if resolution_cm <= 0:
        raise ValueError("resolution_cm must be positive")
    if sum((weld_seams, preclose_seams, weld_relaxed_preclose)) > 1:
        raise ValueError("Seam initialization modes are mutually exclusive")
    pattern = json.loads(spec_path.read_text(encoding="utf-8"))["pattern"]
    counts = edge_segment_counts(pattern, resolution_cm)
    panel_data = {}
    torso_top_cm = max(
        panel["translation"][1] + max(point[1] for point in panel["vertices"])
        for name, panel in pattern["panels"].items()
        if "torso" in name
    )
    vertical_shift_m = shoulder_z_m - torso_top_cm / 100.0
    vertices = []
    faces = []
    components = {}
    global_edge_ids = {}
    local_points_by_panel = {}
    for panel_name in pattern["panel_order"]:
        panel = pattern["panels"][panel_name]
        local_points, local_faces, edge_ids = _panel_mesh(panel, panel_name, counts, resolution_cm)
        local_points_by_panel[panel_name] = local_points
        is_back_panel = "btorso" in panel_name or "sleeve_b" in panel_name
        if is_back_panel:
            local_faces = [[face[0], face[2], face[1]] for face in local_faces]
        offset = len(vertices)
        panel_offset = arm_panel_offset_m if "sleeve" in panel_name else torso_panel_offset_m
        depth_override = None
        if panel_offset is not None:
            depth_override = (
                -panel_offset
                if ("ftorso" in panel_name or "sleeve_f" in panel_name)
                else panel_offset
            )
        vertices.extend(
            _to_blender(point, panel, vertical_shift_m, depth_override) for point in local_points
        )
        world_faces = [[offset + index for index in face] for face in local_faces]
        faces.extend(world_faces)
        components[panel_name] = [offset, len(vertices)]
        for edge_index, ids in edge_ids.items():
            global_edge_ids[(panel_name, edge_index)] = [offset + index for index in ids]
        world_array = np.asarray(vertices[offset:], dtype=float)
        normals = []
        for face in local_faces:
            normal = np.cross(
                world_array[face[1]] - world_array[face[0]],
                world_array[face[2]] - world_array[face[0]],
            )
            norm = float(np.linalg.norm(normal))
            if norm > 0:
                normals.append(normal / norm)
        panel_data[panel_name] = {
            "vertices": len(local_points),
            "faces": len(local_faces),
            "boundary_vertices": sum(
                counts[(panel_name, index)] for index in range(len(panel["edges"]))
            ),
            "outward_side": "back_+Y" if is_back_panel else "front_-Y",
            "mean_face_normal": np.mean(normals, axis=0).tolist(),
        }

    # This untouched placement is a rigid transform of every 2D panel, so all
    # intrinsic pattern edge lengths remain available to Blender's rest shape.
    rest_vertices = [list(vertex) for vertex in vertices]

    attachment_vertices = []
    for panel_name, panel in pattern["panels"].items():
        if "torso" not in panel_name:
            continue
        armhole_edges = [
            index for index, edge in enumerate(panel["edges"]) if "armhole" in edge.get("label", "")
        ]
        if len(armhole_edges) != 1:
            raise ValueError(f"Expected one labeled armhole edge on {panel_name}")
        ids = global_edge_ids[(panel_name, armhole_edges[0])]
        attachment_vertices.append(max(ids, key=lambda index: vertices[index][2]))
    attachment_vertices = sorted(set(attachment_vertices))
    for index in attachment_vertices:
        vertices[index][1] = 0.0

    sewing_edges = set()
    stitch_contract = []
    for stitch_index, stitch in enumerate(pattern.get("stitches", [])):
        left, right = stitch[:2]
        left_ids = global_edge_ids[(left["panel"], left["edge"])]
        right_ids = global_edge_ids[(right["panel"], right["edge"])]
        if len(left_ids) != len(right_ids):
            raise ValueError(f"Stitch {stitch_index} vertex counts differ")
        expected_reverse = not (len(stitch) == 3 and stitch[-1] == "right_wrong")
        vertex_array = np.asarray(vertices, dtype=float)
        forward_mean = float(
            np.mean(
                [
                    np.linalg.norm(vertex_array[left_id] - vertex_array[right_id])
                    for left_id, right_id in zip(left_ids, right_ids)
                ]
            )
        )
        reverse_mean = float(
            np.mean(
                [
                    np.linalg.norm(vertex_array[left_id] - vertex_array[right_id])
                    for left_id, right_id in zip(left_ids, reversed(right_ids))
                ]
            )
        )
        reverse = reverse_mean < forward_mean
        if reverse:
            right_ids = list(reversed(right_ids))
        pairs = []
        initial_distances = []
        for left_id, right_id in zip(left_ids, right_ids):
            pair = tuple(sorted((left_id, right_id)))
            sewing_edges.add(pair)
            pairs.append(list(pair))
            initial_distances.append(
                float(np.linalg.norm(vertex_array[left_id] - vertex_array[right_id]))
            )
        stitch_contract.append(
            {
                "stitch": stitch_index,
                "sides": [left, right],
                "vertex_pairs": len(pairs),
                "reverse_second": reverse,
                "garmentcode_expected_reverse": expected_reverse,
                "orientation_changed": reverse != expected_reverse,
                "forward_mean_distance_m": forward_mean,
                "reverse_mean_distance_m": reverse_mean,
                "chosen_initial_distance_m": {
                    "mean": float(np.mean(initial_distances)),
                    "maximum": float(np.max(initial_distances)),
                },
                "pairs": pairs,
            }
        )

    source_sewing_edges = sorted(sewing_edges)
    uses_predrape = preclose_seams or weld_relaxed_preclose
    if uses_predrape:
        if body is None:
            raise ValueError("preclose_seams requires body dimensions")
        vertices = body_surface_start_vertices(
            pattern,
            local_points_by_panel,
            rest_vertices,
            components,
            body,
            vertical_shift_m,
        )
        vertices = harmonic_preclose_vertices(vertices, faces, components, source_sewing_edges)
        harmonic_ratios = structural_edge_ratios(vertices, rest_vertices, faces)
        vertices = relax_preclosed_vertices(
            vertices,
            rest_vertices,
            faces,
            source_sewing_edges,
        )
        relaxed_ratios = structural_edge_ratios(vertices, rest_vertices, faces)
        if weld_relaxed_preclose:
            rest_vertices = [list(vertex) for vertex in vertices]
            vertices, rest_vertices, faces, components, attachment_vertices = weld_mesh_topology(
                vertices,
                faces,
                components,
                attachment_vertices,
                source_sewing_edges,
                rest_vertices=rest_vertices,
            )
            sewing_edges = set()
            seam_mode = "welded_relaxed_predrape_with_rest_shape"
        else:
            components = {name: {"vertex_range": bounds} for name, bounds in components.items()}
            seam_mode = "preclosed_duplicate_vertices_with_rest_shape"
    elif weld_seams:
        vertices, _, faces, components, attachment_vertices = weld_mesh_topology(
            vertices,
            faces,
            components,
            attachment_vertices,
            source_sewing_edges,
            center_attachments=True,
        )
        sewing_edges = set()
        seam_mode = "welded_shared_vertices"
    else:
        components = {name: {"vertex_range": bounds} for name, bounds in components.items()}
        seam_mode = "open_blender_sewing_springs"

    return {
        "schema_version": 2,
        "source_specification": str(spec_path.resolve()),
        "generator": "GarmentCode MIT pattern specification -> SciPy Delaunay -> Blender cloth mesh",
        "resolution_cm": resolution_cm,
        "coordinate_contract": {
            "garmentcode_cm": "X horizontal, Y vertical, Z depth",
            "blender_m": "X horizontal, Y depth with garment front negative, Z vertical",
            "vertical_shift_m": vertical_shift_m,
            "torso_top_cm_before_shift": torso_top_cm,
            "target_shoulder_z_m": shoulder_z_m,
            "torso_panel_offset_m": torso_panel_offset_m,
            "arm_panel_offset_m": arm_panel_offset_m,
            "weld_seams": weld_seams,
            "preclose_seams": preclose_seams,
            "weld_relaxed_preclose": weld_relaxed_preclose,
        },
        "vertices": vertices,
        "rest_vertices": rest_vertices if uses_predrape else None,
        "faces": faces,
        "sewing_edges": [list(pair) for pair in sorted(sewing_edges)],
        "attachment_vertices": attachment_vertices,
        "components": components,
        "panels": panel_data,
        "stitches": stitch_contract,
        "seam_initialization": {
            "mode": seam_mode,
            "source_sewing_pairs": len(source_sewing_edges),
            "blender_sewing_springs": not (weld_seams or weld_relaxed_preclose),
            "blender_rest_shape_key": uses_predrape,
            "pre_drape_relaxation": {
                "iterations": 1600,
                "relaxation": 0.8,
                "tether": 1e-5,
                "harmonic_edge_ratio_p01_p99": (
                    [
                        float(np.percentile(harmonic_ratios, 1)),
                        float(np.percentile(harmonic_ratios, 99)),
                    ]
                    if uses_predrape
                    else None
                ),
                "relaxed_edge_ratio_p01_p99": (
                    [
                        float(np.percentile(relaxed_ratios, 1)),
                        float(np.percentile(relaxed_ratios, 99)),
                    ]
                    if uses_predrape
                    else None
                ),
                "harmonic_edges_over_1_5x": (
                    int((harmonic_ratios > 1.5).sum()) if uses_predrape else None
                ),
                "relaxed_edges_over_1_5x": (
                    int((relaxed_ratios > 1.5).sum()) if uses_predrape else None
                ),
            },
        },
        "summary": {
            "vertices": len(vertices),
            "faces": len(faces),
            "sewing_edges": len(sewing_edges),
            "source_sewing_pairs": len(source_sewing_edges),
            "panels": len(panel_data),
            "stitches": len(stitch_contract),
            "attachment_vertices": len(attachment_vertices),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--resolution-cm", type=float, default=2.0)
    parser.add_argument("--leave-seams-open", action="store_true")
    parser.add_argument("--preclose-with-rest-shape", action="store_true")
    parser.add_argument("--weld-relaxed-preclose", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    body, panel_pattern, _ = load_pilot_geometry(args.pilot)
    if sum((args.leave_seams_open, args.preclose_with_rest_shape, args.weld_relaxed_preclose)) > 1:
        parser.error("Seam initialization flags are mutually exclusive")
    contract = build_contract(
        args.spec.resolve(),
        args.resolution_cm,
        panel_pattern.shoulder_z_m,
        torso_panel_offset_m=body.torso_radius_y_m + 0.02,
        arm_panel_offset_m=body.arm_radius_m + 0.015,
        weld_seams=not (
            args.leave_seams_open or args.preclose_with_rest_shape or args.weld_relaxed_preclose
        ),
        preclose_seams=args.preclose_with_rest_shape,
        weld_relaxed_preclose=args.weld_relaxed_preclose,
        body=body,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(contract), encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), **contract["summary"]}, indent=2))


if __name__ == "__main__":
    main()
