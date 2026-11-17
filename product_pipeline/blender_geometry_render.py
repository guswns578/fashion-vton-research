"""Headless Blender adapter for geometry-conditioning render outputs.

Run through Blender, for example:
blender --background --python blender_geometry_render.py -- --smoke-test --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def script_args() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for item in list(collection):
            collection.remove(item)


def import_mesh(path: Path, role: str) -> list[bpy.types.Object]:
    before = set(bpy.context.scene.objects)
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
        raise ValueError(f"Unsupported mesh format: {path}")
    objects = [obj for obj in bpy.context.scene.objects if obj not in before and obj.type == "MESH"]
    if not objects:
        raise ValueError(f"No mesh objects imported from {path}")
    for index, obj in enumerate(objects):
        obj.name = f"{role}_{index:03d}"
        obj["geometry_role"] = role
    return objects


def create_smoke_scene() -> tuple[list[bpy.types.Object], list[bpy.types.Object]]:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, location=(0, 0, 0.25))
    body = bpy.context.object
    body.name = "body_smoke"
    body.scale = (0.42, 0.27, 1.0)
    body["geometry_role"] = "body"

    garment = []
    columns, rows = 48, 56
    vertices = []
    faces = []
    for row in range(rows):
        z = -0.27 + 1.24 * row / (rows - 1)
        for column in range(columns):
            x = -0.55 + 1.10 * column / (columns - 1)
            y = -0.34 + 0.035 * math.sin(x * 24.0) * math.sin((z + 0.3) * 13.0)
            y += 0.025 * (x / 0.55) ** 2
            vertices.append((x, y, z))
    for row in range(rows - 1):
        for column in range(columns - 1):
            a = row * columns + column
            b = a + 1
            c = a + columns
            d = c + 1
            faces.append((a, c, d, b))
    mesh = bpy.data.meshes.new("garment_torso_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    torso = bpy.data.objects.new("garment_torso", mesh)
    bpy.context.scene.collection.objects.link(torso)
    torso.name = "garment_torso"
    garment.append(torso)
    solidify = torso.modifiers.new(name="cloth_thickness", type="SOLIDIFY")
    solidify.thickness = 0.025
    for side in (-1, 1):
        bpy.ops.mesh.primitive_cube_add(
            location=(0.72 * side, 0, 0.62),
            scale=(0.28, 0.24, 0.18),
            rotation=(0, math.radians(10 * side), 0),
        )
        sleeve = bpy.context.object
        sleeve.name = f"garment_sleeve_{'l' if side < 0 else 'r'}"
        garment.append(sleeve)
    for obj in garment:
        obj["geometry_role"] = "garment"
        bevel = obj.modifiers.new(name="soft_edges", type="BEVEL")
        bevel.width = 0.08
        bevel.segments = 3
    return [body], garment


def world_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    lower = Vector(tuple(min(point[index] for point in points) for index in range(3)))
    upper = Vector(tuple(max(point[index] for point in points) for index in range(3)))
    return lower, upper


def camera_from_config(
    objects: list[bpy.types.Object],
    resolution: tuple[int, int],
    config_path: Path | None,
) -> tuple[bpy.types.Object, dict]:
    lower, upper = world_bounds(objects)
    center = (lower + upper) * 0.5
    span = upper - lower
    width, height = resolution
    defaults = {
        "projection": "orthographic",
        "location": [center.x, lower.y - max(span.length, 1.0) * 2.0, center.z],
        "target": list(center),
        "ortho_scale": max(span.z, span.x * height / width) * 1.15,
        "clip_start": 0.01,
        "clip_end": max(span.length, 1.0) * 8.0,
    }
    supplied = json.loads(config_path.read_text(encoding="utf-8")) if config_path else {}
    config = {**defaults, **supplied}

    camera_data = bpy.data.cameras.new("geometry_camera")
    camera = bpy.data.objects.new("geometry_camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = Vector(config["location"])
    target = Vector(config["target"])
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "ORTHO" if config["projection"] == "orthographic" else "PERSP"
    if camera_data.type == "ORTHO":
        camera_data.ortho_scale = float(config["ortho_scale"])
    else:
        camera_data.lens = float(config.get("focal_length_mm", 50.0))
        camera_data.sensor_width = float(config.get("sensor_width_mm", 36.0))
        camera_data.sensor_fit = str(config.get("sensor_fit", "AUTO"))
        camera_data.shift_x = float(config.get("shift_x", 0.0))
        camera_data.shift_y = float(config.get("shift_y", 0.0))
    camera_data.clip_start = float(config["clip_start"])
    camera_data.clip_end = float(config["clip_end"])
    bpy.context.scene.camera = camera
    return camera, config


def add_garment_depth_range(
    camera: bpy.types.Object,
    garment: list[bpy.types.Object],
    config: dict,
) -> None:
    bpy.context.view_layer.update()
    inverse = camera.matrix_world.inverted()
    distances = [
        -(inverse @ (obj.matrix_world @ Vector(corner))).z
        for obj in garment
        for corner in obj.bound_box
    ]
    near, far = min(distances), max(distances)
    padding = max((far - near) * 0.05, 1e-4)
    config["depth_near"] = near - padding
    config["depth_far"] = far + padding


def emission_material(name: str, color) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 1.0
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def depth_material(camera_config: dict) -> bpy.types.Material:
    material = bpy.data.materials.new("depth_camera_space")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    geometry = nodes.new("ShaderNodeNewGeometry")
    transform = nodes.new("ShaderNodeVectorTransform")
    transform.vector_type = "POINT"
    transform.convert_from = "WORLD"
    transform.convert_to = "CAMERA"
    separate = nodes.new("ShaderNodeSeparateXYZ")
    multiply = nodes.new("ShaderNodeMath")
    multiply.operation = "MULTIPLY"
    multiply.inputs[1].default_value = -1.0
    mapping = nodes.new("ShaderNodeMapRange")
    mapping.clamp = True
    mapping.inputs["From Min"].default_value = float(camera_config["depth_near"])
    mapping.inputs["From Max"].default_value = float(camera_config["depth_far"])
    mapping.inputs["To Min"].default_value = 1.0
    mapping.inputs["To Max"].default_value = 0.0
    links.new(geometry.outputs["Position"], transform.inputs["Vector"])
    links.new(transform.outputs["Vector"], separate.inputs["Vector"])
    links.new(separate.outputs["Z"], multiply.inputs[0])
    links.new(multiply.outputs[0], mapping.inputs["Value"])
    links.new(mapping.outputs["Result"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def normal_material() -> bpy.types.Material:
    material = bpy.data.materials.new("normal_camera_space")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    geometry = nodes.new("ShaderNodeNewGeometry")
    transform = nodes.new("ShaderNodeVectorTransform")
    transform.vector_type = "NORMAL"
    transform.convert_from = "WORLD"
    transform.convert_to = "CAMERA"
    scale = nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    scale.inputs[3].default_value = 0.5
    add = nodes.new("ShaderNodeVectorMath")
    add.operation = "ADD"
    add.inputs[1].default_value = (0.5, 0.5, 0.5)
    links.new(geometry.outputs["Normal"], transform.inputs["Vector"])
    links.new(transform.outputs["Vector"], scale.inputs[0])
    links.new(scale.outputs["Vector"], add.inputs[0])
    links.new(add.outputs["Vector"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def assign_material(objects: list[bpy.types.Object], material: bpy.types.Material) -> None:
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)


def configure_scene(resolution: tuple[int, int]) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.world.color = (0.0, 0.0, 0.0)
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"


def render_pass(
    path: Path,
    garment: list[bpy.types.Object],
    body: list[bpy.types.Object],
    material: bpy.types.Material,
    color_depth: str,
) -> None:
    assign_material(garment, material)
    for obj in body:
        obj.hide_render = True
    for obj in garment:
        obj.hide_render = False
    scene = bpy.context.scene
    scene.render.image_settings.color_depth = color_depth
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", type=Path)
    parser.add_argument("--garment", type=Path)
    parser.add_argument("--camera", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolution", default="1000x1500")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(script_args())
    if not args.smoke_test and not args.garment:
        raise ValueError("--garment is required unless --smoke-test is used")

    width, height = (int(value) for value in args.resolution.lower().split("x"))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    clear_scene()
    if args.smoke_test:
        body, garment = create_smoke_scene()
    else:
        body = import_mesh(args.body.resolve(), "body") if args.body else []
        garment = import_mesh(args.garment.resolve(), "garment")
    configure_scene((width, height))
    camera, camera_config = camera_from_config(body + garment, (width, height), args.camera)
    add_garment_depth_range(camera, garment, camera_config)

    silhouette_path = out / "garment_silhouette.png"
    depth_path = out / "depth.png"
    normal_path = out / "normal.png"
    render_pass(silhouette_path, garment, body, emission_material("silhouette", (1, 1, 1, 1)), "8")
    render_pass(depth_path, garment, body, depth_material(camera_config), "16")
    render_pass(normal_path, garment, body, normal_material(), "16")

    metadata = {
        "schema_version": 1,
        "renderer": "Blender 4.5 LTS",
        "engine": "BLENDER_EEVEE_NEXT",
        "color_management": {"view_transform": "Raw", "look": "None"},
        "coordinate_convention": {
            "world_up": "+Z",
            "camera_forward": "-Z in camera space",
            "normal_encoding": "camera-space XYZ mapped from [-1,1] to [0,1]",
            "depth_encoding": "linear camera distance mapped garment depth_near->1 and depth_far->0",
        },
        "resolution": [width, height],
        "camera": {
            **camera_config,
            "matrix_world": [list(row) for row in camera.matrix_world],
        },
        "inputs": {
            "body": str(args.body.resolve()) if args.body else None,
            "garment": str(args.garment.resolve()) if args.garment else None,
            "smoke_test": args.smoke_test,
        },
        "outputs": {
            "garment_silhouette": str(silhouette_path.resolve()),
            "depth": str(depth_path.resolve()),
            "normal": str(normal_path.resolve()),
        },
    }
    metadata_path = out / "render_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metadata_path)


if __name__ == "__main__":
    main()
