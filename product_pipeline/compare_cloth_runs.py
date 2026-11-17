"""Compare no-gravity and gravity cloth runs and reject static false positives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("Silhouette dimensions differ")
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def load_obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
    if not vertices:
        raise ValueError(f"No OBJ vertices found: {path}")
    return np.asarray(vertices, dtype=np.float64)


def compare(control: Path, gravity: Path) -> dict:
    control_meta = json.loads((control / "simulation_metadata.json").read_text(encoding="utf-8"))
    gravity_meta = json.loads((gravity / "simulation_metadata.json").read_text(encoding="utf-8"))
    control_render = json.loads(
        (control / "render" / "render_validation.json").read_text(encoding="utf-8")
    )
    gravity_render = json.loads(
        (gravity / "render" / "render_validation.json").read_text(encoding="utf-8")
    )
    control_mask = load_mask(control / "render" / "garment_silhouette.png")
    gravity_mask = load_mask(gravity / "render" / "garment_silhouette.png")
    iou = mask_iou(control_mask, gravity_mask)
    union = np.logical_or(control_mask, gravity_mask).sum()
    xor_fraction = float(np.logical_xor(control_mask, gravity_mask).sum() / union) if union else 0.0
    control_vertices = load_obj_vertices(control / "garment_simulated.obj")
    gravity_vertices = load_obj_vertices(gravity / "garment_simulated.obj")
    if control_vertices.shape != gravity_vertices.shape:
        raise ValueError("Garment topology differs between runs")
    displacement = np.linalg.norm(gravity_vertices - control_vertices, axis=1)
    final_structure = gravity_meta["structural_edge_ratio_to_contract_rest"]["final"]
    meaningful = bool(iou <= 0.98 and displacement.mean() >= 0.005)
    structure_pass = bool(
        final_structure["p01"] >= 0.80
        and final_structure["p99"] <= 1.25
        and final_structure["over_1_5x"] / final_structure["edges"] <= 0.001
    )
    checks = {
        "control_settled": bool(control_meta["settling_drift_m"]["pass"]),
        "gravity_settled": bool(gravity_meta["settling_drift_m"]["pass"]),
        "control_render_valid": bool(control_render["pass"]),
        "gravity_render_valid": bool(gravity_render["pass"]),
        "meaningful_gravity_deformation": meaningful,
        "structural_length_preserved": structure_pass,
    }
    return {
        "schema_version": 1,
        "purpose": "geometry_mechanics_gate",
        "warning": (
            "This confirms non-static gravity deformation and structural stability; it does not "
            "validate fit or fabric behavior without a fixed-camera worn ground truth."
        ),
        "control": str(control.resolve()),
        "gravity": str(gravity.resolve()),
        "silhouette": {
            "iou": iou,
            "xor_fraction_of_union": xor_fraction,
            "control_pixels": int(control_mask.sum()),
            "gravity_pixels": int(gravity_mask.sum()),
        },
        "gravity_vs_control_vertex_displacement_m": {
            "mean": float(displacement.mean()),
            "p50": float(np.percentile(displacement, 50)),
            "p90": float(np.percentile(displacement, 90)),
            "maximum": float(displacement.max()),
            "fraction_over_5mm": float((displacement > 0.005).mean()),
            "fraction_over_10mm": float((displacement > 0.010).mean()),
        },
        "gravity_final_structural_edge_ratio": final_structure,
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--gravity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.control.resolve(), args.gravity.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), "pass": report["pass"]}))


if __name__ == "__main__":
    main()
