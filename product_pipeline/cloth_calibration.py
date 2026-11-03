"""Plan, resume, and score coarse real-garment cloth calibration trials."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

from geometry_eval import evaluate


HERE = Path(__file__).resolve().parent


def build_trials() -> list[dict]:
    trials = []
    for mass_scale, damping_scale, bending in product(
        (1500.0, 2500.0, 4000.0),
        (0.8, 1.0, 1.2),
        (0.15, 0.35, 0.8),
    ):
        trial_id = f"m{int(mass_scale)}_d{int(damping_scale * 10):02d}_b{int(bending * 100):03d}"
        trials.append({
            "trial_id": trial_id,
            "solver_mass_scale": mass_scale,
            "damping_scale": damping_scale,
            "tension_stiffness": 20.0,
            "compression_stiffness": 20.0,
            "shear_stiffness": 8.0,
            "bending_stiffness": bending,
        })
    return trials


def run_trial(
    trial: dict,
    pilot: Path,
    camera: Path,
    out: Path,
    resolution: str,
    max_frames: int,
) -> None:
    command = [
        sys.executable,
        str(HERE / "run_cloth_ablation.py"),
        "--out", str(out),
        "--presets", "medium",
        "--frames", str(max_frames),
        "--min-frames", "160",
        "--settle-consecutive", "3",
        "--resolution", resolution,
        "--camera", str(camera),
        "--garment-kind", "sewn-long-sleeve",
        "--solver-quality", "20",
        "--collision-quality", "8",
        "--sewing-force", "12",
        "--pilot", str(pilot),
        "--solver-mass-scale", str(trial["solver_mass_scale"]),
        "--damping-scale", str(trial["damping_scale"]),
        "--tension-stiffness", str(trial["tension_stiffness"]),
        "--compression-stiffness", str(trial["compression_stiffness"]),
        "--shear-stiffness", str(trial["shear_stiffness"]),
        "--bending-stiffness", str(trial["bending_stiffness"]),
        "--allow-unsettled",
    ]
    subprocess.run(command, check=True)


def score_trial(trial: dict, root: Path, real_mask: Path) -> dict:
    report_path = root / "report.json"
    if not report_path.exists():
        return {**trial, "status": "missing"}
    run_report = json.loads(report_path.read_text(encoding="utf-8"))
    row = run_report["presets"][0]
    if not row["accepted"]:
        return {
            **trial,
            "status": "rejected_unsettled_or_invalid",
            "run_summary": run_report["summary"],
        }
    overlay = root / "calibration_overlay.png"
    geometry = evaluate(
        real_mask,
        Path(row["silhouette"]),
        None,
        None,
        overlay,
        0.80,
        0.05,
    )
    boundary_ratio = geometry["boundary"]["symmetric_distance_torso_ratio"]
    score = geometry["mask_iou"] - 0.5 * boundary_ratio
    return {
        **trial,
        "status": "scored",
        "score": score,
        "mask_iou": geometry["mask_iou"],
        "boundary_symmetric_torso_ratio": boundary_ratio,
        "geometry_report": geometry,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--real-mask", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolution", default="1000x1500")
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    trials = build_trials()
    if args.limit:
        trials = trials[:args.limit]
    plan = {
        "schema_version": 1,
        "purpose": "coarse_real_worn_cloth_parameter_calibration",
        "warning": (
            "Plan creation does not validate capture truth. Execute only with a visually reviewed "
            "real worn mask and the camera calibrated to that same fixed capture."
        ),
        "pilot": str(args.pilot.resolve()),
        "real_mask": str(args.real_mask.resolve()),
        "camera": str(args.camera.resolve()),
        "resolution": args.resolution,
        "max_frames": args.max_frames,
        "selection_score": "mask_iou - 0.5 * symmetric_boundary_distance_torso_ratio",
        "trials": trials,
    }
    plan_path = out / "calibration_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if not args.execute:
        print(plan_path)
        return

    for required in (args.pilot, args.real_mask, args.camera):
        if not required.exists():
            raise FileNotFoundError(required)
    rows = []
    for trial in trials:
        trial_root = out / "trials" / trial["trial_id"]
        if not (trial_root / "report.json").exists():
            run_trial(trial, args.pilot.resolve(), args.camera.resolve(), trial_root, args.resolution, args.max_frames)
        rows.append(score_trial(trial, trial_root, args.real_mask.resolve()))
    ranked = sorted(
        (row for row in rows if row["status"] == "scored"),
        key=lambda row: row["score"],
        reverse=True,
    )
    report = {
        **plan,
        "results": rows,
        "ranked_trial_ids": [row["trial_id"] for row in ranked],
        "selected": ranked[0] if ranked else None,
        "decision": "selected" if ranked else "no_converged_valid_trial",
    }
    report_path = out / "calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
