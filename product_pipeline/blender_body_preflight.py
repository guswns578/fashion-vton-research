"""Audit an imported body collider before expensive cloth preparation and simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from blender_cloth_sim import (
    body_mesh_topology,
    clear_scene,
    import_mesh_objects,
    object_bounds,
    script_args,
)


def connected_components(obj: bpy.types.Object) -> int:
    adjacency = [set() for _ in obj.data.vertices]
    for edge in obj.data.edges:
        left, right = (int(value) for value in edge.vertices)
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(range(len(adjacency)))
    count = 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            neighbors = adjacency[stack.pop()].intersection(unseen)
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--scale-unit", choices=("m", "cm", "mm"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(script_args())

    body_path = args.body.resolve()
    if not body_path.exists():
        raise FileNotFoundError(body_path)
    clear_scene()
    objects = import_mesh_objects(body_path)
    unit_scale = {"m": 1.0, "cm": 0.01, "mm": 0.001}[args.scale_unit]
    for obj in objects:
        obj.scale = tuple(value * unit_scale for value in obj.scale)
    bpy.context.view_layer.update()
    minimum, maximum = object_bounds(objects)

    audits = []
    for obj in objects:
        topology = body_mesh_topology(obj)
        components = connected_components(obj)
        audits.append(
            {
                "object": obj.name,
                **topology,
                "connected_components": components,
                "single_shell": components == 1,
            }
        )
    checks = {
        "one_mesh_object": len(objects) == 1,
        "single_connected_shell": all(item["single_shell"] for item in audits),
        "watertight": all(item["watertight"] for item in audits),
        "positive_extent": all(float(value) > 0 for value in maximum - minimum),
    }
    report = {
        "schema_version": 1,
        "purpose": "external_body_collider_preflight",
        "body": str(body_path),
        "scale_unit": args.scale_unit,
        "unit_scale": unit_scale,
        "bbox_m": {"min": minimum.tolist(), "max": maximum.tolist()},
        "objects": audits,
        "checks": checks,
        "pass": all(checks.values()),
        "repair_required": not all(checks.values()),
        "warning": (
            "This validates collider topology and scale extent only. It does not validate body "
            "shape accuracy, pose, landmark alignment, or scan consent."
        ),
    }
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(out)
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
