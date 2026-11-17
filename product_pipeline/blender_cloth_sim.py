"""Synthetic stationary-drape pilot using Blender's cloth solver.

This validates the simulation/export path. It is not a calibrated fabric model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from measurement_geometry import (
    BodyProxy,
    PanelPattern,
    load_pilot_geometry,
    load_pilot_material,
    resolve_mass_gsm,
    vertex_mass_from_gsm,
)

PRESETS = {
    "soft": {
        "mass_gsm": 140.0,
        "tension_stiffness": 8.0,
        "compression_stiffness": 8.0,
        "shear_stiffness": 3.0,
        "bending_stiffness": 0.05,
    },
    "medium": {
        "mass_gsm": 220.0,
        "tension_stiffness": 20.0,
        "compression_stiffness": 20.0,
        "shear_stiffness": 8.0,
        "bending_stiffness": 0.35,
    },
    "stiff": {
        "mass_gsm": 320.0,
        "tension_stiffness": 45.0,
        "compression_stiffness": 45.0,
        "shear_stiffness": 18.0,
        "bending_stiffness": 2.0,
    },
}


def script_args() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def add_collision(obj: bpy.types.Object) -> None:
    collision = obj.modifiers.new(name=f"{obj.name}_collision", type="COLLISION")
    if hasattr(collision.settings, "thickness_outer"):
        collision.settings.thickness_outer = 0.008
    if hasattr(collision.settings, "cloth_friction"):
        collision.settings.cloth_friction = 20.0


def import_mesh_objects(path: Path) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    suffix = path.suffix.lower()
    if suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".ply":
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        raise ValueError(f"Unsupported body mesh format: {suffix}")
    objects = [obj for obj in bpy.data.objects if obj not in before and obj.type == "MESH"]
    if not objects:
        raise ValueError(f"No mesh objects were imported from {path}")
    return objects


def object_bounds(objects: list[bpy.types.Object]) -> tuple[np.ndarray, np.ndarray]:
    points = []
    for obj in objects:
        points.extend(tuple(obj.matrix_world @ vertex.co) for vertex in obj.data.vertices)
    array = np.asarray(points, dtype=np.float64)
    return array.min(axis=0), array.max(axis=0)


def prepare_external_body(
    path: Path,
    scale_unit: str,
    alignment: str,
    dimensions: BodyProxy,
    pattern: PanelPattern,
) -> tuple[list[bpy.types.Object], dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    objects = import_mesh_objects(path)
    unit_scale = {"m": 1.0, "cm": 0.01, "mm": 0.001}[scale_unit]
    for obj in objects:
        obj.scale = tuple(value * unit_scale for value in obj.scale)
    bpy.context.view_layer.update()
    before_min, before_max = object_bounds(objects)
    height_scale = 1.0
    translation = np.zeros(3, dtype=np.float64)
    if alignment == "full-scan-bbox":
        observed_height = float(before_max[2] - before_min[2])
        expected_height = dimensions.source_cm["height"] / 100.0
        if observed_height <= 0:
            raise ValueError("Imported body mesh has zero height.")
        height_scale = expected_height / observed_height
        for obj in objects:
            obj.scale = tuple(value * height_scale for value in obj.scale)
        bpy.context.view_layer.update()
        scaled_min, scaled_max = object_bounds(objects)
        center_x = float((scaled_min[0] + scaled_max[0]) * 0.5)
        center_y = float((scaled_min[1] + scaled_max[1]) * 0.5)
        shoulder_z = float(scaled_min[2] + 0.82 * (scaled_max[2] - scaled_min[2]))
        translation = np.asarray((-center_x, -center_y, pattern.shoulder_z_m - shoulder_z))
        for obj in objects:
            obj.location = tuple(np.asarray(obj.location, dtype=np.float64) + translation)
        bpy.context.view_layer.update()
    after_min, after_max = object_bounds(objects)
    for obj in objects:
        add_collision(obj)
    return objects, {
        "mode": "external_mesh",
        "path": str(path.resolve()),
        "scale_unit": scale_unit,
        "unit_scale": unit_scale,
        "alignment": alignment,
        "height_scale": height_scale,
        "translation_m": translation.tolist(),
        "bbox_before": {"min": before_min.tolist(), "max": before_max.tolist()},
        "bbox_after": {"min": after_min.tolist(), "max": after_max.tolist()},
        "mesh_objects": len(objects),
    }


def resolve_body_input(
    pilot_path: Path | None,
    cli_path: Path | None,
    cli_scale_unit: str | None,
    cli_alignment: str | None,
) -> tuple[Path | None, str, str]:
    config: dict[str, object] = {}
    if pilot_path:
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        config = dict(pilot.get("subject", {}).get("body_mesh", {}))
    path = cli_path
    if path is None and config.get("path"):
        configured = Path(str(config["path"]))
        path = configured if configured.is_absolute() else pilot_path.parent / configured
    scale_unit = cli_scale_unit or str(config.get("scale_unit") or "m")
    alignment = cli_alignment or str(config.get("alignment") or "full-scan-bbox")
    return path, scale_unit, alignment


def create_body(garment_kind: str, dimensions: BodyProxy) -> list[bpy.types.Object]:
    metric = garment_kind in {"sewn-long-sleeve", "garmentcode-sewn"}
    torso_location = (0, 0, dimensions.torso_center_z_m) if metric else (0, 0, 0.03)
    torso_scale = (
        (
            dimensions.torso_radius_x_m,
            dimensions.torso_radius_y_m,
            dimensions.torso_half_height_m,
        )
        if metric
        else (0.46, 0.29, 0.92)
    )
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, location=torso_location)
    body = bpy.context.object
    body.name = "body_torso"
    body.scale = torso_scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    add_collision(body)
    objects = [body]
    if metric:
        shoulder_half = dimensions.source_cm["shoulder_breadth"] / 200.0
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=64,
            ring_count=24,
            location=(0.0, 0.0, dimensions.shoulder_z_m - dimensions.arm_radius_m * 0.15),
        )
        shoulder = bpy.context.object
        shoulder.name = "body_shoulder_bridge"
        shoulder.scale = (
            shoulder_half + dimensions.arm_radius_m * 0.55,
            dimensions.torso_radius_y_m * 0.92,
            dimensions.arm_radius_m * 1.20,
        )
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        add_collision(shoulder)
        objects.append(shoulder)

        neck_radius = dimensions.source_cm["neck"] / 100.0 / (2 * math.pi)
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=48,
            ring_count=24,
            location=(0.0, 0.0, dimensions.shoulder_z_m + neck_radius * 0.75),
        )
        neck = bpy.context.object
        neck.name = "body_neck"
        neck.scale = (neck_radius, neck_radius, neck_radius * 1.45)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        add_collision(neck)
        objects.append(neck)
    if garment_kind in {"long-sleeve", "sewn-long-sleeve", "garmentcode-sewn"}:
        for side in (-1, 1):
            arm_location = (
                (
                    dimensions.arm_center_x_m * side,
                    0.0,
                    dimensions.arm_center_z_m,
                )
                if metric
                else (0.73 * side, 0.0, 0.54)
            )
            arm_angle = dimensions.arm_angle_deg if metric else 8.0
            if metric:
                wrist_radius = dimensions.source_cm["wrist"] / 100.0 / (2 * math.pi)
                outward_rotation = math.radians(side * (90.0 + arm_angle))
                bpy.ops.mesh.primitive_cone_add(
                    vertices=64,
                    radius1=dimensions.arm_radius_m,
                    radius2=wrist_radius,
                    depth=dimensions.arm_half_length_m * 2.0,
                    location=arm_location,
                    rotation=(0.0, outward_rotation, 0.0),
                )
                arm = bpy.context.object
                arm.name = f"body_arm_{'l' if side < 0 else 'r'}"
                add_collision(arm)
                objects.append(arm)

                shoulder_half = dimensions.source_cm["shoulder_breadth"] / 200.0
                arm_length = dimensions.arm_half_length_m * 2.0
                outer_x = side * (shoulder_half + math.cos(math.radians(arm_angle)) * arm_length)
                outer_z = dimensions.shoulder_z_m - math.sin(math.radians(arm_angle)) * arm_length
                bpy.ops.mesh.primitive_uv_sphere_add(
                    segments=48,
                    ring_count=24,
                    location=(outer_x, 0.0, outer_z),
                )
                hand = bpy.context.object
                hand.name = f"body_hand_{'l' if side < 0 else 'r'}"
                hand.scale = (wrist_radius * 1.8, wrist_radius * 1.65, wrist_radius * 2.2)
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
                add_collision(hand)
                objects.append(hand)
            else:
                bpy.ops.mesh.primitive_uv_sphere_add(
                    segments=48,
                    ring_count=24,
                    location=arm_location,
                    rotation=(0.0, math.radians(arm_angle * side), 0.0),
                )
                arm = bpy.context.object
                arm.name = f"body_arm_{'l' if side < 0 else 'r'}"
                arm.scale = (0.72, 0.125, 0.125)
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
                add_collision(arm)
                objects.append(arm)
    return objects


def merge_body_colliders(objects: list[bpy.types.Object]) -> list[bpy.types.Object]:
    """Boolean-union overlapping procedural parts into one external collision shell."""
    if len(objects) < 2:
        return objects
    for obj in objects:
        for modifier in list(obj.modifiers):
            if modifier.type == "COLLISION":
                obj.modifiers.remove(modifier)
    base = objects[0]
    bpy.context.view_layer.objects.active = base
    for other in objects[1:]:
        modifier = base.modifiers.new(name=f"union_{other.name}", type="BOOLEAN")
        modifier.operation = "UNION"
        modifier.solver = "EXACT"
        modifier.object = other
        bpy.context.view_layer.objects.active = base
        base.select_set(True)
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(other, do_unlink=True)
    base.name = "body_union"
    add_collision(base)
    bpy.context.view_layer.update()
    return [base]


def point_in_polygon(x: float, z: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x0, z0 = previous
        x1, z1 = current
        if (z0 > z) != (z1 > z):
            intersection = (x1 - x0) * (z - z0) / (z1 - z0) + x0
            if x < intersection:
                inside = not inside
        previous = current
    return inside


def create_sewn_long_sleeve(
    pattern: PanelPattern,
    columns: int,
    rows: int,
) -> tuple[bpy.types.Object, np.ndarray, dict[str, tuple[int, int]]]:
    outline = list(pattern.outline_xz_m)
    min_x = min(point[0] for point in outline)
    max_x = max(point[0] for point in outline)
    min_z = min(point[1] for point in outline)
    max_z = max(point[1] for point in outline)
    x_values = np.linspace(min_x - 0.01, max_x + 0.01, columns + 1)
    z_values = np.linspace(min_z - 0.01, max_z + 0.01, rows + 1)
    dx = float(x_values[1] - x_values[0])
    dz = float(z_values[1] - z_values[0])
    local_vertices: list[tuple[float, float]] = []
    index_by_grid: dict[tuple[int, int], int] = {}
    faces_local: list[tuple[int, int, int, int]] = []

    def vertex(ix: int, iz: int) -> int:
        key = (ix, iz)
        if key not in index_by_grid:
            index_by_grid[key] = len(local_vertices)
            local_vertices.append((float(x_values[ix]), float(z_values[iz])))
        return index_by_grid[key]

    for iz in range(len(z_values) - 1):
        for ix in range(len(x_values) - 1):
            center_x = float((x_values[ix] + x_values[ix + 1]) * 0.5)
            center_z = float((z_values[iz] + z_values[iz + 1]) * 0.5)
            if point_in_polygon(center_x, center_z, outline):
                faces_local.append(
                    (
                        vertex(ix, iz),
                        vertex(ix + 1, iz),
                        vertex(ix + 1, iz + 1),
                        vertex(ix, iz + 1),
                    )
                )

    edge_counts: dict[tuple[int, int], int] = {}
    for face in faces_local:
        for left, right in zip(face, face[1:] + face[:1]):
            edge = tuple(sorted((left, right)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    seam_vertices: set[int] = set()
    for (left, right), count in edge_counts.items():
        if count != 1:
            continue
        x = (local_vertices[left][0] + local_vertices[right][0]) * 0.5
        z = (local_vertices[left][1] + local_vertices[right][1]) * 0.5
        is_hem = z < pattern.hem_z_m + 1.5 * dz
        is_cuff = abs(x) > pattern.sleeve_end_x_m - 1.5 * dx
        is_neck = z > pattern.neck_line_z_m - 1.5 * dz and abs(x) < pattern.neck_half_m + 1.5 * dx
        if not (is_hem or is_cuff or is_neck):
            seam_vertices.update((left, right))

    panel_size = len(local_vertices)
    vertices = [(x, -pattern.panel_offset_y_m, z) for x, z in local_vertices]
    vertices.extend((x, pattern.panel_offset_y_m, z) for x, z in local_vertices)
    faces = list(faces_local)
    faces.extend(tuple(panel_size + index for index in reversed(face)) for face in faces_local)
    sewing_edges = [(index, panel_size + index) for index in sorted(seam_vertices)]

    mesh = bpy.data.meshes.new("sewn_long_sleeve_mesh")
    mesh.from_pydata(vertices, sewing_edges, faces)
    mesh.update()
    garment = bpy.data.objects.new("garment_sewn_long_sleeve", mesh)
    bpy.context.scene.collection.objects.link(garment)
    garment.vertex_groups.new(name="top_pin")
    return (
        garment,
        np.asarray(vertices, dtype=np.float64),
        {
            "front_panel": (0, panel_size),
            "back_panel": (panel_size, panel_size * 2),
        },
    )


def create_garmentcode_sewn(
    contract_path: Path,
) -> tuple[bpy.types.Object, np.ndarray, dict[str, object], dict]:
    if not contract_path.exists():
        raise FileNotFoundError(contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    vertices = contract["vertices"]
    faces = contract["faces"]
    sewing_edges = contract["sewing_edges"]
    use_sewing_springs = bool(
        contract.get("seam_initialization", {}).get("blender_sewing_springs", True)
    )
    if not vertices or not faces or (use_sewing_springs and not sewing_edges):
        raise ValueError("GarmentCode mesh contract is missing required topology.")
    mesh = bpy.data.meshes.new("garmentcode_sewn_mesh")
    mesh.from_pydata(vertices, sewing_edges, faces)
    mesh.update()
    garment = bpy.data.objects.new("garment_garmentcode_sewn", mesh)
    bpy.context.scene.collection.objects.link(garment)
    rest_vertices = contract.get("rest_vertices")
    rest_shape_key_name = None
    if rest_vertices is not None:
        if len(rest_vertices) != len(vertices):
            raise ValueError("GarmentCode rest vertices do not match the start topology.")
        garment.shape_key_add(name="Basis")
        rest_key = garment.shape_key_add(name="cloth_pattern_rest")
        for point, coordinates in zip(rest_key.data, rest_vertices):
            point.co = coordinates
        garment.active_shape_key_index = 0
        rest_shape_key_name = rest_key.name
    pin = garment.vertex_groups.new(name="top_pin")
    attachment_vertices = [int(index) for index in contract.get("attachment_vertices", [])]
    if attachment_vertices:
        pin.add(attachment_vertices, 1.0, "REPLACE")
    components = contract["components"]
    return (
        garment,
        np.asarray(vertices, dtype=np.float64),
        components,
        {
            "path": str(contract_path.resolve()),
            "source_specification": contract.get("source_specification"),
            "resolution_cm": contract.get("resolution_cm"),
            "summary": contract.get("summary"),
            "coordinate_contract": contract.get("coordinate_contract"),
            "attachment_vertices": attachment_vertices,
            "use_sewing_springs": use_sewing_springs,
            "rest_shape_key_name": rest_shape_key_name,
            "_sewing_edges": [[int(left), int(right)] for left, right in sewing_edges],
            "_contract_rest_vertices": rest_vertices,
            "_contract_faces": faces,
        },
    )


def _closest_signed_distance(
    point_world: Vector,
    obj: bpy.types.Object,
    inside_mode: str = "normal",
) -> tuple[float, Vector, Vector] | None:
    inverse = obj.matrix_world.inverted()
    point_local = inverse @ point_world
    hit, location_local, normal_local, _ = obj.closest_point_on_mesh(point_local, distance=10.0)
    if not hit:
        return None
    location_world = obj.matrix_world @ location_local
    normal_world = (obj.matrix_world.to_3x3().inverted().transposed() @ normal_local).normalized()
    normal_world *= float(obj.get("outward_normal_sign", 1.0))
    if inside_mode == "ray_parity":
        inside = _ray_parity_inside(point_world, obj)
        distance = float((point_world - location_world).length)
        signed_distance = -distance if inside else distance
    elif inside_mode == "normal":
        signed_distance = float((point_world - location_world).dot(normal_world))
    else:
        raise ValueError(f"Unknown inside mode: {inside_mode}")
    return signed_distance, location_world, normal_world


def _ray_parity_inside(point_world: Vector, obj: bpy.types.Object) -> bool:
    directions = (
        Vector((1.0, 0.371390676, 0.173205081)).normalized(),
        Vector((-0.238104, 1.0, 0.417203)).normalized(),
        Vector((0.312771, -0.193447, 1.0)).normalized(),
    )
    votes = sum(_ray_parity_vote(point_world, obj, direction) for direction in directions)
    return votes >= 2


def _ray_parity_vote(
    point_world: Vector,
    obj: bpy.types.Object,
    direction_world: Vector,
) -> bool:
    inverse = obj.matrix_world.inverted()
    origin = inverse @ point_world
    direction = (inverse.to_3x3() @ direction_world).normalized()
    epsilon = 1e-5
    hits = 0
    travelled = 0.0
    maximum_distance = 100.0
    for _ in range(128):
        hit, location, _, _ = obj.ray_cast(origin, direction, distance=maximum_distance - travelled)
        if not hit:
            break
        step = float((location - origin).length)
        travelled += step + epsilon
        if travelled >= maximum_distance:
            break
        hits += 1
        origin = location + direction * epsilon
    return bool(hits % 2)


def body_mesh_topology(obj: bpy.types.Object) -> dict[str, int | float | bool]:
    incidence: dict[tuple[int, int], int] = {}
    orientation_balance: dict[tuple[int, int], int] = {}
    signed_volume = 0.0
    vertices = obj.data.vertices
    for polygon in obj.data.polygons:
        indices = list(polygon.vertices)
        for left, right in zip(indices, indices[1:] + indices[:1]):
            edge = tuple(sorted((int(left), int(right))))
            incidence[edge] = incidence.get(edge, 0) + 1
            direction = 1 if (int(left), int(right)) == edge else -1
            orientation_balance[edge] = orientation_balance.get(edge, 0) + direction
        origin = Vector(vertices[indices[0]].co)
        for index in range(1, len(indices) - 1):
            left = Vector(vertices[indices[index]].co)
            right = Vector(vertices[indices[index + 1]].co)
            signed_volume += float(origin.dot(left.cross(right))) / 6.0
    boundary = sum(value == 1 for value in incidence.values())
    nonmanifold = sum(value > 2 for value in incidence.values())
    inconsistent_winding = sum(
        incidence[edge] == 2 and balance != 0 for edge, balance in orientation_balance.items()
    )
    return {
        "vertices": len(obj.data.vertices),
        "polygons": len(obj.data.polygons),
        "boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,
        "inconsistent_winding_edges": inconsistent_winding,
        "signed_volume_local": signed_volume,
        "outward_normal_sign": 1.0 if signed_volume > 0 else -1.0,
        "watertight": (
            boundary == 0
            and nonmanifold == 0
            and inconsistent_winding == 0
            and abs(signed_volume) > 1e-12
        ),
    }


def penetration_stats(
    garment: bpy.types.Object,
    body: list[bpy.types.Object],
    inside_mode: str = "normal",
) -> dict[str, float | int]:
    depths = []
    for vertex in garment.data.vertices:
        point = garment.matrix_world @ vertex.co
        signed = [
            result[0]
            for obj in body
            if (result := _closest_signed_distance(point, obj, inside_mode))
        ]
        depth = max((-value for value in signed if value < 0.0), default=0.0)
        depths.append(depth)
    array = np.asarray(depths, dtype=np.float64)
    positive = array[array > 0]
    return {
        "penetrating_vertices": int((array > 0).sum()),
        "penetrating_fraction": float((array > 0).mean()),
        "mean_depth_m": float(positive.mean()) if len(positive) else 0.0,
        "maximum_depth_m": float(array.max()) if len(array) else 0.0,
    }


def resolve_body_penetration(
    garment: bpy.types.Object,
    body: list[bpy.types.Object],
    clearance_m: float,
    iterations: int,
    inside_mode: str = "normal",
) -> dict:
    if clearance_m < 0 or iterations < 1:
        raise ValueError("Penetration clearance must be nonnegative and iterations positive.")
    bpy.context.view_layer.update()
    before = penetration_stats(garment, body, inside_mode)
    inverse_garment = garment.matrix_world.inverted()
    start_positions = np.asarray(
        [tuple(garment.matrix_world @ vertex.co) for vertex in garment.data.vertices],
        dtype=np.float64,
    )
    movements = np.zeros(len(garment.data.vertices), dtype=np.float64)
    iteration_changes = []
    for iteration in range(iterations):
        changed = 0
        maximum_step = 0.0
        for vertex in garment.data.vertices:
            point = garment.matrix_world @ vertex.co
            original = point.copy()
            for obj in body:
                result = _closest_signed_distance(point, obj, inside_mode)
                if result is None:
                    continue
                signed_distance, _, normal = result
                if signed_distance < clearance_m:
                    point += normal * (clearance_m - signed_distance)
            step = float((point - original).length)
            if step > 1e-9:
                vertex.co = inverse_garment @ point
                movements[vertex.index] += step
                changed += 1
                maximum_step = max(maximum_step, step)
        garment.data.update()
        bpy.context.view_layer.update()
        iteration_changes.append(
            {
                "iteration": iteration + 1,
                "changed_vertices": changed,
                "maximum_step_m": maximum_step,
            }
        )
        if changed == 0:
            break
    after = penetration_stats(garment, body, inside_mode)
    end_positions = np.asarray(
        [tuple(garment.matrix_world @ vertex.co) for vertex in garment.data.vertices],
        dtype=np.float64,
    )
    largest_indices = np.argsort(movements)[-10:][::-1]
    return {
        "method": (
            "nearest surface plus ray-parity inside test"
            if inside_mode == "ray_parity"
            else "nearest-surface signed normal projection for convex procedural body parts"
        ),
        "inside_mode": inside_mode,
        "clearance_m": clearance_m,
        "maximum_iterations": iterations,
        "before": before,
        "after": after,
        "movement_m": vector_stats(movements),
        "largest_movements": [
            {
                "vertex": int(index),
                "movement_m": float(movements[index]),
                "start_world_m": start_positions[index].tolist(),
                "end_world_m": end_positions[index].tolist(),
                "net_displacement_m": float(
                    np.linalg.norm(end_positions[index] - start_positions[index])
                ),
            }
            for index in largest_indices
            if movements[index] > 0
        ],
        "iterations": iteration_changes,
        "limitation": (
            "Ray parity requires a watertight, consistently scaled body mesh."
            if inside_mode == "ray_parity"
            else "Use ray parity for non-convex external scans."
        ),
    }


def create_garment(
    columns: int,
    rows: int,
    garment_kind: str,
    pattern: PanelPattern,
) -> tuple[bpy.types.Object, np.ndarray, dict[str, tuple[int, int]]]:
    if garment_kind == "sewn-long-sleeve":
        return create_sewn_long_sleeve(pattern, columns, rows)
    vertices = []
    faces = []
    for row in range(rows):
        t = row / (rows - 1)
        z = 0.78 - 1.48 * t
        shoulder_t = min(t / 0.20, 1.0)
        shoulder_blend = shoulder_t * shoulder_t * (3.0 - 2.0 * shoulder_t)
        radius_x = 0.34 + (0.56 - 0.34) * shoulder_blend + 0.06 * max(t - 0.20, 0.0) / 0.80
        radius_y = 0.22 + (0.355 - 0.22) * shoulder_blend + 0.025 * max(t - 0.20, 0.0) / 0.80
        for column in range(columns):
            theta = 2 * math.pi * column / columns
            asymmetric = 1.0 + 0.018 * math.sin(3 * theta + 1.7 * t)
            x = radius_x * asymmetric * math.cos(theta)
            y = radius_y * asymmetric * math.sin(theta)
            vertices.append((x, y, z))
    for row in range(rows - 1):
        for column in range(columns):
            next_column = (column + 1) % columns
            a = row * columns + column
            b = row * columns + next_column
            c = (row + 1) * columns + column
            d = (row + 1) * columns + next_column
            faces.append((a, c, d, b))
    components = {"torso": (0, len(vertices))}
    pinned = [0, 1, columns - 1, columns // 2 - 1, columns // 2, columns // 2 + 1]
    if garment_kind == "long-sleeve":
        sleeve_columns = 24
        sleeve_rows = 34
        for side in (-1, 1):
            offset = len(vertices)
            for sleeve_row in range(sleeve_rows):
                u = sleeve_row / (sleeve_rows - 1)
                x = side * (0.30 + 1.05 * u)
                center_z = 0.60 - 0.12 * u
                radius = 0.19 - 0.045 * u
                for column in range(sleeve_columns):
                    theta = 2 * math.pi * column / sleeve_columns
                    y = radius * math.cos(theta)
                    z = center_z + radius * math.sin(theta)
                    vertices.append((x, y, z))
            for sleeve_row in range(sleeve_rows - 1):
                for column in range(sleeve_columns):
                    next_column = (column + 1) % sleeve_columns
                    a = offset + sleeve_row * sleeve_columns + column
                    b = offset + sleeve_row * sleeve_columns + next_column
                    c = offset + (sleeve_row + 1) * sleeve_columns + column
                    d = offset + (sleeve_row + 1) * sleeve_columns + next_column
                    faces.append((a, c, d, b) if side > 0 else (a, b, d, c))
            pinned.extend(range(offset, offset + sleeve_columns))
            components[f"sleeve_{'l' if side < 0 else 'r'}"] = (offset, len(vertices))
    mesh = bpy.data.meshes.new("garment_shell_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    garment = bpy.data.objects.new("garment_shell", mesh)
    bpy.context.scene.collection.objects.link(garment)

    pin = garment.vertex_groups.new(name="top_pin")
    pin.add(pinned, 1.0, "REPLACE")
    return garment, np.asarray(vertices, dtype=np.float64), components


def configure_cloth(
    garment: bpy.types.Object,
    preset: dict,
    frames: int,
    use_self_collision: bool,
    use_sewing: bool,
    solver_quality: int,
    collision_quality: int,
    sewing_force: float,
    surface_area_m2: float,
    mass_gsm_override: float | None,
    pilot_material: dict[str, object],
    solver_mass_scale: float,
    damping_scale: float,
    pin_stiffness: float,
    use_pin_group: bool,
    rest_shape_key_name: str | None = None,
    collision_distance_m: float = 0.008,
) -> tuple[bpy.types.Modifier, dict[str, object]]:
    cloth = garment.modifiers.new(name="cloth", type="CLOTH")
    settings = cloth.settings
    settings.quality = solver_quality
    settings.bending_model = "ANGULAR"
    settings.vertex_group_mass = "top_pin" if use_pin_group else ""
    settings.pin_stiffness = pin_stiffness
    if rest_shape_key_name:
        key_blocks = garment.data.shape_keys.key_blocks if garment.data.shape_keys else None
        if key_blocks is None or rest_shape_key_name not in key_blocks:
            raise ValueError(f"Missing cloth rest shape key: {rest_shape_key_name}")
        settings.rest_shape_key = key_blocks[rest_shape_key_name]
    if mass_gsm_override is not None:
        mass_gsm, mass_source = float(mass_gsm_override), "cli_override"
    else:
        mass_gsm, mass_source = resolve_mass_gsm(
            pilot_material, surface_area_m2, preset["mass_gsm"]
        )
    if solver_mass_scale <= 0:
        raise ValueError("solver_mass_scale must be positive.")
    if damping_scale <= 0:
        raise ValueError("damping_scale must be positive.")
    physical_vertex_mass, total_mass = vertex_mass_from_gsm(
        surface_area_m2,
        len(garment.data.vertices),
        mass_gsm,
    )
    solver_vertex_mass = physical_vertex_mass * solver_mass_scale
    settings.mass = solver_vertex_mass
    for name, value in preset.items():
        if name == "mass_gsm":
            continue
        setattr(settings, name, value)
    settings.tension_damping = min(10.0 * damping_scale, 50.0)
    settings.compression_damping = min(10.0 * damping_scale, 50.0)
    settings.shear_damping = min(10.0 * damping_scale, 50.0)
    settings.bending_damping = min(5.0 * damping_scale, 50.0)
    settings.air_damping = min(5.0 * damping_scale, 10.0)
    settings.use_sewing_springs = use_sewing
    settings.sewing_force_max = sewing_force if use_sewing else 0.0
    collision = cloth.collision_settings
    collision.use_collision = True
    collision.collision_quality = collision_quality
    collision.distance_min = collision_distance_m
    collision.use_self_collision = use_self_collision
    collision.self_distance_min = 0.006
    cloth.point_cache.frame_start = 1
    cloth.point_cache.frame_end = frames
    return cloth, {
        "mass_gsm": mass_gsm,
        "mass_source": mass_source,
        "surface_area_m2": surface_area_m2,
        "physical_total_mass_kg": total_mass,
        "physical_vertex_mass_kg": physical_vertex_mass,
        "solver_mass_scale": solver_mass_scale,
        "solver_vertex_mass": solver_vertex_mass,
        "damping_scale": damping_scale,
    }


def surface_area(obj: bpy.types.Object) -> float:
    area = 0.0
    vertices = obj.data.vertices
    for polygon in obj.data.polygons:
        origin = np.asarray(vertices[polygon.vertices[0]].co, dtype=np.float64)
        for index in range(1, len(polygon.vertices) - 1):
            left = np.asarray(vertices[polygon.vertices[index]].co, dtype=np.float64) - origin
            right = np.asarray(vertices[polygon.vertices[index + 1]].co, dtype=np.float64) - origin
            area += 0.5 * float(np.linalg.norm(np.cross(left, right)))
    return area


def evaluated_vertices(obj: bpy.types.Object) -> np.ndarray:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return np.asarray([tuple(evaluated.matrix_world @ vertex.co) for vertex in mesh.vertices])
    finally:
        evaluated.to_mesh_clear()


def vector_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "maximum": float(values.max()),
    }


def pair_distance_stats(vertices: np.ndarray, pairs) -> dict[str, float | int] | None:
    if not pairs:
        return None
    pair_array = np.asarray(pairs, dtype=np.int64)
    distances = np.linalg.norm(vertices[pair_array[:, 0]] - vertices[pair_array[:, 1]], axis=1)
    return {
        "pairs": int(len(distances)),
        "mean": float(distances.mean()),
        "p90": float(np.percentile(distances, 90)),
        "maximum": float(distances.max()),
        "within_1mm": int((distances <= 0.001).sum()),
    }


def structural_ratio_stats(vertices: np.ndarray, rest_vertices, faces) -> dict | None:
    if rest_vertices is None or not faces:
        return None
    edges = set()
    for face in faces:
        for left, right in zip(face, face[1:] + face[:1]):
            edges.add(tuple(sorted((left, right))))
    edge_array = np.asarray(sorted(edges), dtype=np.int64)
    rest = np.asarray(rest_vertices, dtype=np.float64)
    rest_lengths = np.linalg.norm(rest[edge_array[:, 1]] - rest[edge_array[:, 0]], axis=1)
    lengths = np.linalg.norm(vertices[edge_array[:, 1]] - vertices[edge_array[:, 0]], axis=1)
    ratios = lengths / np.maximum(rest_lengths, 1e-12)
    return {
        "edges": int(len(ratios)),
        "minimum": float(ratios.min()),
        "p01": float(np.percentile(ratios, 1)),
        "p10": float(np.percentile(ratios, 10)),
        "p90": float(np.percentile(ratios, 90)),
        "p99": float(np.percentile(ratios, 99)),
        "maximum": float(ratios.max()),
        "mean_absolute_error": float(np.mean(np.abs(ratios - 1.0))),
        "over_1_5x": int((ratios > 1.5).sum()),
        "under_0_67x": int((ratios < 0.67).sum()),
    }


def collision_aware_predrape(
    garment: bpy.types.Object,
    body: list[bpy.types.Object],
    rest_vertices,
    faces,
    clearance_m: float,
    iterations: int,
    relaxation: float = 0.8,
    collision_interval: int = 40,
    tether: float = 1e-5,
    inside_mode: str = "normal",
) -> dict:
    if iterations < 1 or collision_interval < 1:
        raise ValueError("Collision-aware pre-drape iterations must be positive.")
    rest = np.asarray(rest_vertices, dtype=np.float64)
    edges = set()
    for face in faces:
        for left, right in zip(face, face[1:] + face[:1]):
            edges.add(tuple(sorted((left, right))))
    edge_array = np.asarray(sorted(edges), dtype=np.int64)
    rest_lengths = np.linalg.norm(rest[edge_array[:, 1]] - rest[edge_array[:, 0]], axis=1)
    result = np.asarray([tuple(vertex.co) for vertex in garment.data.vertices], dtype=np.float64)
    anchor = result.copy()
    projection_passes = []
    for iteration in range(iterations):
        difference = result[edge_array[:, 1]] - result[edge_array[:, 0]]
        lengths = np.linalg.norm(difference, axis=1)
        correction = (0.5 * relaxation * (lengths - rest_lengths) / np.maximum(lengths, 1e-12))[
            :, None
        ] * difference
        delta = np.zeros_like(result)
        degree = np.zeros(len(result), dtype=np.float64)
        np.add.at(delta, edge_array[:, 0], correction)
        np.add.at(delta, edge_array[:, 1], -correction)
        np.add.at(degree, edge_array[:, 0], 1.0)
        np.add.at(degree, edge_array[:, 1], 1.0)
        result += delta / np.maximum(degree[:, None], 1.0)
        result += (anchor - result) * tether
        if (iteration + 1) % collision_interval == 0 or iteration + 1 == iterations:
            for vertex, coordinates in zip(garment.data.vertices, result):
                vertex.co = coordinates
            bpy.context.view_layer.update()
            projection = resolve_body_penetration(
                garment,
                body,
                clearance_m=clearance_m,
                iterations=2,
                inside_mode=inside_mode,
            )
            projection_passes.append(
                {
                    "iteration": iteration + 1,
                    "changed_vertices": sum(
                        item["changed_vertices"] for item in projection["iterations"]
                    ),
                    "movement_mean_m": projection["movement_m"]["mean"],
                    "movement_maximum_m": projection["movement_m"]["maximum"],
                }
            )
            result = np.asarray(
                [tuple(vertex.co) for vertex in garment.data.vertices], dtype=np.float64
            )
    return {
        "iterations": iterations,
        "relaxation": relaxation,
        "collision_interval": collision_interval,
        "tether": tether,
        "projection_passes": projection_passes,
        "inside_mode": inside_mode,
        "final_penetration": penetration_stats(garment, body, inside_mode),
        "final_structure": structural_ratio_stats(result, rest_vertices, faces),
    }


def component_values(values: np.ndarray, descriptor: object) -> np.ndarray:
    if isinstance(descriptor, dict) and "vertex_indices" in descriptor:
        return values[np.asarray(descriptor["vertex_indices"], dtype=np.int64)]
    if isinstance(descriptor, dict) and "vertex_range" in descriptor:
        start, end = descriptor["vertex_range"]
        return values[start:end]
    start, end = descriptor
    return values[start:end]


def mesh_topology_stats(obj: bpy.types.Object) -> dict[str, int]:
    adjacency = [set() for _ in obj.data.vertices]
    for edge in obj.data.edges:
        left, right = edge.vertices
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(range(len(adjacency)))
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            current = stack.pop()
            neighbors = adjacency[current].intersection(unseen)
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
    pin_group = obj.vertex_groups.get("top_pin")
    pinned = 0
    if pin_group:
        for vertex in obj.data.vertices:
            if any(group.group == pin_group.index and group.weight > 0 for group in vertex.groups):
                pinned += 1
    return {
        "connected_components": components,
        "isolated_vertices": sum(not neighbors for neighbors in adjacency),
        "pinned_vertices": pinned,
    }


def freeze_evaluated_object(obj: bpy.types.Object, name: str) -> bpy.types.Object:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(evaluated, depsgraph=depsgraph)
    frozen = bpy.data.objects.new(name, mesh)
    frozen.matrix_world = evaluated.matrix_world.copy()
    bpy.context.scene.collection.objects.link(frozen)
    obj.hide_render = True
    obj.hide_viewport = True
    return frozen


def export_selected(objects: list[bpy.types.Object], path: Path) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.hide_viewport = False
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.wm.obj_export(
        filepath=str(path),
        export_selected_objects=True,
        export_materials=False,
        forward_axis="NEGATIVE_Z",
        up_axis="Y",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="medium")
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--columns", type=int, default=40)
    parser.add_argument("--rows", type=int, default=48)
    parser.add_argument(
        "--garment-kind",
        choices=("tube", "long-sleeve", "sewn-long-sleeve", "garmentcode-sewn"),
        default="tube",
    )
    parser.add_argument("--garmentcode-mesh", type=Path)
    parser.add_argument("--settle-window", type=int, default=10)
    parser.add_argument("--min-frames", type=int, default=80)
    parser.add_argument("--settle-consecutive", type=int, default=3)
    parser.add_argument("--dress-frames", type=int)
    parser.add_argument("--gravity-scale", type=float, default=1.0)
    parser.add_argument("--solver-quality", type=int, default=8)
    parser.add_argument("--collision-quality", type=int, default=4)
    parser.add_argument("--disable-self-collision", action="store_true")
    parser.add_argument("--disable-body-collision", action="store_true")
    parser.add_argument("--disable-penetration-resolution", action="store_true")
    parser.add_argument("--resolve-external-penetration", action="store_true")
    parser.add_argument("--commit-penetration-to-rest-shape", action="store_true")
    parser.add_argument("--disable-rest-shape-key", action="store_true")
    parser.add_argument("--collision-aware-predrape-iterations", type=int, default=0)
    parser.add_argument("--collision-aware-predrape-collision-interval", type=int, default=40)
    parser.add_argument("--body-collision-thickness", type=float, default=0.008)
    parser.add_argument("--cloth-collision-distance", type=float, default=0.008)
    parser.add_argument("--penetration-clearance", type=float, default=0.008)
    parser.add_argument("--penetration-iterations", type=int, default=5)
    parser.add_argument("--disable-pinning", action="store_true")
    parser.add_argument("--sewing-force", type=float, default=12.0)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--mass-gsm", type=float)
    parser.add_argument("--solver-mass-scale", type=float, default=2500.0)
    parser.add_argument("--damping-scale", type=float, default=1.0)
    parser.add_argument("--tension-stiffness", type=float)
    parser.add_argument("--compression-stiffness", type=float)
    parser.add_argument("--shear-stiffness", type=float)
    parser.add_argument("--bending-stiffness", type=float)
    parser.add_argument("--body-mesh", type=Path)
    parser.add_argument("--merge-procedural-body", action="store_true")
    parser.add_argument("--body-scale-unit", choices=("m", "cm", "mm"))
    parser.add_argument(
        "--body-alignment",
        choices=("prealigned", "full-scan-bbox"),
    )
    args = parser.parse_args(script_args())
    if args.settle_window < 1 or args.min_frames < 2 or args.settle_consecutive < 1:
        raise ValueError("Settle window, minimum frames, and consecutive count must be positive.")
    if args.min_frames > args.frames:
        raise ValueError("min_frames cannot exceed the maximum --frames value.")
    if args.garment_kind == "garmentcode-sewn" and not args.garmentcode_mesh:
        raise ValueError("--garmentcode-mesh is required for garmentcode-sewn mode.")
    dress_frames = args.dress_frames
    if dress_frames is None:
        dress_frames = 60 if args.garment_kind == "garmentcode-sewn" else 0
    if dress_frames < 0 or dress_frames >= args.frames:
        raise ValueError("dress_frames must be nonnegative and less than --frames.")
    if args.gravity_scale < 0:
        raise ValueError("gravity_scale must be nonnegative.")
    if args.body_collision_thickness <= 0 or args.cloth_collision_distance <= 0:
        raise ValueError("Collision distances must be positive.")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    body_dimensions, panel_pattern, geometry_source = load_pilot_geometry(args.pilot)
    pilot_material = load_pilot_material(args.pilot)
    body_mesh_path, body_scale_unit, body_alignment = resolve_body_input(
        args.pilot,
        args.body_mesh,
        args.body_scale_unit,
        args.body_alignment,
    )
    clear_scene()
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = args.frames
    scene.gravity = (
        (0.0, 0.0, 0.0)
        if dress_frames or args.gravity_scale == 0
        else (0.0, 0.0, -9.81 * args.gravity_scale)
    )
    if body_mesh_path:
        body, body_source = prepare_external_body(
            body_mesh_path.resolve(),
            body_scale_unit,
            body_alignment,
            body_dimensions,
            panel_pattern,
        )
    else:
        body = create_body(args.garment_kind, body_dimensions)
        body_source = {"mode": "procedural_proxy"}
        if args.merge_procedural_body:
            source_parts = len(body)
            body = merge_body_colliders(body)
            body_source = {
                "mode": "procedural_boolean_union",
                "source_parts": source_parts,
                "collision_shells": len(body),
            }
    body_topology = []
    for obj in body:
        topology = {"object": obj.name, **body_mesh_topology(obj)}
        obj["outward_normal_sign"] = topology["outward_normal_sign"]
        body_topology.append(topology)
    body_source["topology"] = body_topology
    penetration_inside_mode = "ray_parity" if body_source["mode"] == "external_mesh" else "normal"
    body_source["penetration_inside_mode"] = penetration_inside_mode
    if args.resolve_external_penetration:
        if body_source["mode"] != "external_mesh":
            raise ValueError("--resolve-external-penetration requires an external --body-mesh.")
        invalid = [item["object"] for item in body_topology if not item["watertight"]]
        if invalid:
            raise ValueError(
                "External penetration resolution requires watertight body meshes; "
                f"invalid objects: {', '.join(invalid)}"
            )
    if (
        args.collision_aware_predrape_iterations
        and body_source["mode"] == "external_mesh"
        and not args.resolve_external_penetration
    ):
        raise ValueError(
            "Collision-aware pre-drape on an external body requires "
            "--resolve-external-penetration."
        )
    if args.disable_body_collision:
        for obj in body:
            for modifier in list(obj.modifiers):
                if modifier.type == "COLLISION":
                    obj.modifiers.remove(modifier)
    else:
        for obj in body:
            for modifier in obj.modifiers:
                if modifier.type == "COLLISION" and hasattr(modifier.settings, "thickness_outer"):
                    modifier.settings.thickness_outer = args.body_collision_thickness
    garmentcode_contract = None
    if args.garment_kind == "garmentcode-sewn":
        garment, initial, components, garmentcode_contract = create_garmentcode_sewn(
            args.garmentcode_mesh.resolve()
        )
    else:
        garment, initial, components = create_garment(
            args.columns,
            args.rows,
            args.garment_kind,
            panel_pattern,
        )
    sewing_edges = garmentcode_contract.pop("_sewing_edges", []) if garmentcode_contract else []
    contract_rest_vertices = (
        garmentcode_contract.pop("_contract_rest_vertices", None) if garmentcode_contract else None
    )
    contract_faces = garmentcode_contract.pop("_contract_faces", []) if garmentcode_contract else []
    seam_before_penetration = pair_distance_stats(initial, sewing_edges)
    structure_before_penetration = structural_ratio_stats(
        initial, contract_rest_vertices, contract_faces
    )
    penetration_contract = None
    rest_shape_commit = None
    penetration_allowed = body_source["mode"] in {
        "procedural_proxy",
        "procedural_boolean_union",
    } or (body_source["mode"] == "external_mesh" and args.resolve_external_penetration)
    if (
        args.garment_kind == "garmentcode-sewn"
        and penetration_allowed
        and not args.disable_penetration_resolution
    ):
        penetration_contract = resolve_body_penetration(
            garment,
            body,
            args.penetration_clearance,
            args.penetration_iterations,
            inside_mode=penetration_inside_mode,
        )
        initial = np.asarray(
            [tuple(garment.matrix_world @ vertex.co) for vertex in garment.data.vertices],
            dtype=np.float64,
        )
    seam_after_penetration = pair_distance_stats(initial, sewing_edges)
    structure_after_penetration = structural_ratio_stats(
        initial, contract_rest_vertices, contract_faces
    )
    collision_predrape_contract = None
    if args.collision_aware_predrape_iterations:
        if contract_rest_vertices is None or not contract_faces:
            raise ValueError("Collision-aware pre-drape requires contract rest geometry.")
        collision_predrape_contract = collision_aware_predrape(
            garment,
            body,
            contract_rest_vertices,
            contract_faces,
            args.penetration_clearance,
            args.collision_aware_predrape_iterations,
            collision_interval=args.collision_aware_predrape_collision_interval,
            inside_mode=penetration_inside_mode,
        )
        initial = np.asarray(
            [tuple(garment.matrix_world @ vertex.co) for vertex in garment.data.vertices],
            dtype=np.float64,
        )
    structure_after_collision_predrape = structural_ratio_stats(
        initial, contract_rest_vertices, contract_faces
    )
    if args.commit_penetration_to_rest_shape:
        if args.disable_rest_shape_key:
            raise ValueError("Cannot commit a disabled rest shape key.")
        if not garmentcode_contract or not garmentcode_contract.get("rest_shape_key_name"):
            raise ValueError("Rest-shape commit requires a GarmentCode rest shape key.")
        rest_key = garment.data.shape_keys.key_blocks[garmentcode_contract["rest_shape_key_name"]]
        before = np.asarray([tuple(point.co) for point in rest_key.data], dtype=np.float64)
        after = np.asarray([tuple(vertex.co) for vertex in garment.data.vertices], dtype=np.float64)
        movement = np.linalg.norm(after - before, axis=1)
        for point, coordinates in zip(rest_key.data, after):
            point.co = coordinates
        rest_shape_commit = {
            "enabled": True,
            "reason": "collision-safe pre-drape becomes the cloth spring rest state",
            "movement_m": vector_stats(movement),
        }
    if args.disable_rest_shape_key and garment.data.shape_keys:
        garment.shape_key_clear()
    cloth_surface_area = surface_area(garment)
    physics_preset = dict(PRESETS[args.preset])
    for key, value in {
        "tension_stiffness": args.tension_stiffness,
        "compression_stiffness": args.compression_stiffness,
        "shear_stiffness": args.shear_stiffness,
        "bending_stiffness": args.bending_stiffness,
    }.items():
        if value is not None:
            if value <= 0:
                raise ValueError(f"{key} override must be positive.")
            physics_preset[key] = value
    garmentcode_uses_sewing = bool(
        garmentcode_contract and garmentcode_contract.get("use_sewing_springs")
    )
    cloth_modifier, mass_contract = configure_cloth(
        garment,
        physics_preset,
        args.frames,
        use_self_collision=args.garment_kind != "long-sleeve" and not args.disable_self_collision,
        use_sewing=args.garment_kind == "sewn-long-sleeve" or garmentcode_uses_sewing,
        solver_quality=args.solver_quality,
        collision_quality=args.collision_quality,
        sewing_force=args.sewing_force,
        surface_area_m2=cloth_surface_area,
        mass_gsm_override=args.mass_gsm,
        pilot_material=pilot_material,
        solver_mass_scale=args.solver_mass_scale,
        damping_scale=args.damping_scale,
        pin_stiffness=50.0 if args.garment_kind == "garmentcode-sewn" else 1.0,
        use_pin_group=not args.disable_pinning,
        rest_shape_key_name=(
            garmentcode_contract.get("rest_shape_key_name")
            if garmentcode_contract and not args.disable_rest_shape_key
            else None
        ),
        collision_distance_m=args.cloth_collision_distance,
    )

    scene.frame_set(1)
    simulation_started = time.perf_counter()
    previous_check = initial
    previous_check_frame = 1
    final = initial
    settling_drift = np.zeros(len(initial), dtype=np.float64)
    settle_streak = 0
    settle_checks = []
    actual_frame = 1
    for frame in range(2, args.frames + 1):
        if dress_frames and frame == dress_frames + 1 and args.gravity_scale:
            scene.gravity = (0.0, 0.0, -9.81 * args.gravity_scale)
        scene.frame_set(frame)
        if (frame - 1) % args.settle_window != 0 and frame != args.frames:
            continue
        final = evaluated_vertices(garment)
        settling_drift = np.linalg.norm(final - previous_check, axis=1)
        stats = vector_stats(settling_drift)
        eligible = frame >= max(args.min_frames, dress_frames + args.settle_window)
        passed = bool(eligible and stats["mean"] <= 0.002 and stats["p90"] <= 0.005)
        settle_streak = settle_streak + 1 if passed else 0
        settle_checks.append(
            {
                "frame": frame,
                "window_frames": frame - previous_check_frame,
                "eligible": eligible,
                "pass": passed,
                **stats,
            }
        )
        actual_frame = frame
        previous_check = final
        previous_check_frame = frame
        if settle_streak >= args.settle_consecutive:
            break
    simulation_seconds = time.perf_counter() - simulation_started
    displacement = np.linalg.norm(final - initial, axis=1)
    seam_final = pair_distance_stats(final, sewing_edges)
    structure_final = structural_ratio_stats(final, contract_rest_vertices, contract_faces)
    frozen = freeze_evaluated_object(garment, "garment_simulated")
    body_path = out / "body.obj"
    garment_path = out / "garment_simulated.obj"
    export_selected(body, body_path)
    export_selected([frozen], garment_path)

    initial_min, initial_max = initial.min(axis=0), initial.max(axis=0)
    final_min, final_max = final.min(axis=0), final.max(axis=0)
    external_body = body_source["mode"] == "external_mesh"
    metadata = {
        "schema_version": 1,
        "purpose": (
            "external_body_stationary_drape_gate"
            if external_body
            else "synthetic_stationary_drape_pipeline_validation"
        ),
        "warning": (
            "An external body mesh does not validate garment fit or fabric parameters; compare "
            "against a fixed-camera worn ground truth."
            if external_body
            else "Preset values and procedural body are not calibrated to real fabric or a person."
        ),
        "garment_kind": args.garment_kind,
        "geometry_source": geometry_source,
        "body_proxy": (
            body_dimensions.to_dict()
            if args.garment_kind in {"sewn-long-sleeve", "garmentcode-sewn"}
            else None
        ),
        "body_source": body_source,
        "panel_pattern": (
            panel_pattern.to_dict()
            if args.garment_kind in {"sewn-long-sleeve", "garmentcode-sewn"}
            else None
        ),
        "garmentcode_mesh_contract": garmentcode_contract,
        "initial_penetration": penetration_contract,
        "rest_shape_commit_after_penetration": rest_shape_commit,
        "collision_aware_predrape": collision_predrape_contract,
        "seam_distance_m": (
            {
                "before_penetration_resolution": seam_before_penetration,
                "after_penetration_resolution": seam_after_penetration,
                "final": seam_final,
            }
            if sewing_edges
            else None
        ),
        "structural_edge_ratio_to_contract_rest": (
            {
                "before_penetration_resolution": structure_before_penetration,
                "after_penetration_resolution": structure_after_penetration,
                "after_collision_aware_predrape": structure_after_collision_predrape,
                "final": structure_final,
            }
            if contract_rest_vertices is not None
            else None
        ),
        "topology_warning": (
            "Long-sleeve mode uses disconnected overlapping torso and sleeve shells, not sewn panels."
            if args.garment_kind == "long-sleeve"
            else None
        ),
        "preset": args.preset,
        "physics": {**physics_preset, **mass_contract},
        "solver_damping": {
            "scale": args.damping_scale,
            "tension": min(10.0 * args.damping_scale, 50.0),
            "compression": min(10.0 * args.damping_scale, 50.0),
            "shear": min(10.0 * args.damping_scale, 50.0),
            "bending": min(5.0 * args.damping_scale, 50.0),
            "air": min(5.0 * args.damping_scale, 10.0),
        },
        "solver": {
            "quality": args.solver_quality,
            "collision_quality": args.collision_quality,
            "sewing_force_max": (
                args.sewing_force
                if args.garment_kind == "sewn-long-sleeve" or garmentcode_uses_sewing
                else 0.0
            ),
            "pin_stiffness": 50.0 if args.garment_kind == "garmentcode-sewn" else 1.0,
            "pin_group_enabled": not args.disable_pinning,
            "self_collision": args.garment_kind != "long-sleeve"
            and not args.disable_self_collision,
            "body_collision": not args.disable_body_collision,
            "body_collision_thickness_m": args.body_collision_thickness,
            "cloth_collision_distance_m": args.cloth_collision_distance,
            "scene_use_gravity": bool(scene.use_gravity),
            "scene_gravity": list(scene.gravity),
            "cloth_effector_gravity": float(cloth_modifier.settings.effector_weights.gravity),
            "rest_shape_key": (
                cloth_modifier.settings.rest_shape_key.name
                if cloth_modifier.settings.rest_shape_key
                else None
            ),
        },
        "frames": actual_frame,
        "max_frames": args.frames,
        "dressing": {
            "zero_gravity_frames": dress_frames,
            "gravity_enabled_from_frame": dress_frames + 1 if dress_frames else 1,
            "gravity_scale": args.gravity_scale,
        },
        "simulation_seconds": simulation_seconds,
        "seconds_per_frame": simulation_seconds / max(actual_frame - 1, 1),
        "mesh": {
            "columns": args.columns,
            "rows": args.rows,
            "vertices": int(len(initial)),
            **mesh_topology_stats(garment),
        },
        "displacement_m": vector_stats(displacement),
        "settling_drift_m": {
            **vector_stats(settling_drift),
            "window_frames": settle_checks[-1]["window_frames"],
            "pass": settle_streak >= args.settle_consecutive,
            "consecutive_required": args.settle_consecutive,
            "consecutive_achieved": settle_streak,
            "minimum_frames": args.min_frames,
            "thresholds": {"mean": 0.002, "p90": 0.005},
            "checks": settle_checks,
        },
        "components": {
            name: {
                "vertex_count": int(len(component_values(displacement, descriptor))),
                "displacement_m": vector_stats(component_values(displacement, descriptor)),
                "settling_drift_m": vector_stats(component_values(settling_drift, descriptor)),
            }
            for name, descriptor in components.items()
        },
        "initial_bbox_m": {"min": initial_min.tolist(), "max": initial_max.tolist()},
        "final_bbox_m": {"min": final_min.tolist(), "max": final_max.tolist()},
        "outputs": {"body": str(body_path), "garment": str(garment_path)},
    }
    metadata_path = out / "simulation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metadata_path)


if __name__ == "__main__":
    main()
