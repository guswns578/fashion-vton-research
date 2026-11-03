"""Run synthetic cloth presets through simulation and the geometry renderer."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe")
DEFAULT_OUT = Path(r"fashion_pipeline/out/cloth_ablation_v1")


def mask_iou(left: Path, right: Path) -> float:
    with Image.open(left) as source:
        a = np.asarray(source.convert("L")) >= 128
    with Image.open(right) as source:
        image = source.convert("L")
        if image.size != (a.shape[1], a.shape[0]):
            image = image.resize((a.shape[1], a.shape[0]), Image.Resampling.NEAREST)
        b = np.asarray(image) >= 128
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def silhouette_stats(path: Path) -> dict[str, float]:
    with Image.open(path) as source:
        mask = np.asarray(source.convert("L")) >= 128
    ys, xs = np.where(mask)
    height, width = mask.shape
    if not len(xs):
        return {
            "foreground_fraction": 0.0,
            "height_image_ratio": None,
            "width_image_ratio": None,
        }
    return {
        "foreground_fraction": float(mask.mean()),
        "height_image_ratio": float((ys.max() - ys.min() + 1) / height),
        "width_image_ratio": float((xs.max() - xs.min() + 1) / width),
    }


def normal_mae(left: Path, right: Path, left_mask: Path, right_mask: Path) -> float:
    with Image.open(left) as source:
        a = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    with Image.open(right) as source:
        b = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    with Image.open(left_mask) as source:
        ma = np.asarray(source.convert("L")) >= 128
    with Image.open(right_mask) as source:
        mb = np.asarray(source.convert("L")) >= 128
    overlap = np.logical_and(ma, mb)
    return float(np.abs(a[overlap] - b[overlap]).mean()) if overlap.any() else None


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def make_qc_sheet(rows: list[dict], path: Path) -> None:
    cell_w, cell_h = 260, 390
    canvas = Image.new("RGB", (cell_w * len(rows), 40 + cell_h * 2), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for column, row in enumerate(rows):
        x = column * cell_w
        draw.text((x + 10, 12), row["preset"], fill=(240, 240, 240))
        for image_row, key in enumerate(("silhouette", "normal")):
            with Image.open(row[key]) as source:
                image = source.convert("RGB")
                image.thumbnail((cell_w - 20, cell_h - 20), Image.Resampling.LANCZOS)
            px = x + (cell_w - image.width) // 2
            py = 40 + image_row * cell_h + (cell_h - image.height) // 2
            canvas.paste(image, (px, py))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--presets", nargs="+", default=["soft", "medium", "stiff"])
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--min-frames", type=int, default=80)
    parser.add_argument("--settle-consecutive", type=int, default=3)
    parser.add_argument("--resolution", default="480x720")
    parser.add_argument("--camera", type=Path)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--allow-unsettled", action="store_true")
    parser.add_argument("--solver-quality", type=int, default=8)
    parser.add_argument("--collision-quality", type=int, default=4)
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
    parser.add_argument("--garmentcode-mesh", type=Path)
    parser.add_argument("--body-scale-unit", choices=("m", "cm", "mm"))
    parser.add_argument(
        "--body-alignment",
        choices=("prealigned", "full-scan-bbox"),
    )
    parser.add_argument(
        "--garment-kind",
        choices=("tube", "long-sleeve", "sewn-long-sleeve", "garmentcode-sewn"),
        default="tube",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    camera_path = args.camera.resolve() if args.camera else out / "fixed_camera.json"
    if not args.camera:
        camera_path.write_text(json.dumps({
            "projection": "orthographic",
            "location": [0.0, -5.0, 0.03],
            "target": [0.0, 0.0, 0.03],
            "ortho_scale": 3.0 if args.garment_kind != "tube" else 2.4,
            "clip_start": 0.01,
            "clip_end": 20.0,
        }, indent=2), encoding="utf-8")

    rows = []
    for preset in args.presets:
        root = out / preset
        simulation = root / "simulation"
        render = root / "render"
        if not args.report_only:
            simulation_command = [
                str(BLENDER), "--background", "--python", str(HERE / "blender_cloth_sim.py"),
                "--", "--out", str(simulation), "--preset", preset, "--frames", str(args.frames),
                "--min-frames", str(args.min_frames),
                "--settle-consecutive", str(args.settle_consecutive),
                "--garment-kind", args.garment_kind,
                "--solver-quality", str(args.solver_quality),
                "--collision-quality", str(args.collision_quality),
                "--sewing-force", str(args.sewing_force),
                "--solver-mass-scale", str(args.solver_mass_scale),
                "--damping-scale", str(args.damping_scale),
            ]
            if args.pilot:
                simulation_command.extend(("--pilot", str(args.pilot.resolve())))
            if args.mass_gsm:
                simulation_command.extend(("--mass-gsm", str(args.mass_gsm)))
            if args.body_mesh:
                simulation_command.extend(("--body-mesh", str(args.body_mesh.resolve())))
            if args.garmentcode_mesh:
                simulation_command.extend(("--garmentcode-mesh", str(args.garmentcode_mesh.resolve())))
            if args.body_scale_unit:
                simulation_command.extend(("--body-scale-unit", args.body_scale_unit))
            if args.body_alignment:
                simulation_command.extend(("--body-alignment", args.body_alignment))
            for flag, value in (
                ("--tension-stiffness", args.tension_stiffness),
                ("--compression-stiffness", args.compression_stiffness),
                ("--shear-stiffness", args.shear_stiffness),
                ("--bending-stiffness", args.bending_stiffness),
            ):
                if value is not None:
                    simulation_command.extend((flag, str(value)))
            run(simulation_command)
        metadata = json.loads((simulation / "simulation_metadata.json").read_text(encoding="utf-8"))
        if not args.report_only:
            run([
                sys.executable, str(HERE / "run_geometry_render.py"),
                "--body", str(simulation / "body.obj"),
                "--garment", str(simulation / "garment_simulated.obj"),
                "--camera", str(camera_path),
                "--out", str(render), "--resolution", args.resolution,
                "--allow-invalid",
            ])
        silhouette_path = render / "garment_silhouette.png"
        rows.append({
            "preset": preset,
            "simulation": metadata,
            "render_validation": json.loads((render / "render_validation.json").read_text(encoding="utf-8")),
            "silhouette": str(silhouette_path.resolve()),
            "normal": str((render / "normal.png").resolve()),
            "silhouette_stats": silhouette_stats(silhouette_path),
        })
        rows[-1]["accepted"] = bool(
            rows[-1]["simulation"]["settling_drift_m"]["pass"]
            and rows[-1]["render_validation"]["pass"]
        )

    comparisons = []
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            comparisons.append({
                "left": left["preset"],
                "right": right["preset"],
                "silhouette_iou": mask_iou(Path(left["silhouette"]), Path(right["silhouette"])),
                "normal_mae_on_overlap": normal_mae(
                    Path(left["normal"]), Path(right["normal"]),
                    Path(left["silhouette"]), Path(right["silhouette"]),
                ),
            })
    qc_sheet = out / "qc_sheet.jpg"
    make_qc_sheet(rows, qc_sheet)
    report = {
        "schema_version": 1,
        "purpose": "verify_that_material_presets_change_stationary_drape",
        "warning": "This is a synthetic mechanics test, not real-fabric calibration.",
        "garment_kind": args.garment_kind,
        "presets": rows,
        "comparisons": comparisons,
        "summary": {
            "all_converged": all(row["simulation"]["settling_drift_m"]["pass"] for row in rows),
            "all_render_pass": all(row["render_validation"]["pass"] for row in rows),
            "all_accepted": all(row["accepted"] for row in rows),
        },
        "qc_sheet": str(qc_sheet.resolve()),
    }
    path = out / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path)
    if not report["summary"]["all_accepted"] and not args.allow_unsettled:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
