"""Run the real-body G-C1 geometry gate as a resumable staged pipeline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jsonschema

from validate_geometry_pilot import build_readiness_report, resolve_asset


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path(r"fashion_pipeline/out/geometry_pilot_v1")
DEFAULT_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe")
DEFAULT_GARMENTCODE_PYTHON = (
    HERE.parent / ".venv_garmentcode" / "Scripts" / "python.exe"
)
STAGES = (
    "pattern",
    "mesh",
    "body",
    "control",
    "gravity",
    "render",
    "compare",
    "evaluate",
)


def relative_or_absolute(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def stage_blockers(stage: str, readiness: dict[str, Any]) -> list[str]:
    ready = readiness["readiness"]
    missing = readiness["missing"]
    assets = readiness["assets"]
    blockers: list[str] = []
    if stage == "pattern":
        if not ready["consent"]:
            blockers.append("required consent flags are not all true")
        if missing["body_measurements"]:
            blockers.append("body measurements: " + ", ".join(missing["body_measurements"]))
        if missing["garment_measurements"]:
            blockers.append("garment measurements: " + ", ".join(missing["garment_measurements"]))
        if missing["minimum_material"]:
            blockers.append("material fields: " + ", ".join(missing["minimum_material"]))
        for key in ("garment_flat_front", "garment_flat_back"):
            if not assets[key]["exists"]:
                blockers.append(f"missing asset: {key}")
    elif stage in {"body", "control", "gravity"}:
        if not ready["simulation"]:
            blockers.extend(stage_blockers("pattern", readiness))
            if not assets["body_mesh"]["exists"]:
                blockers.append("missing asset: body_mesh")
            if missing["body_mesh_metadata"]:
                blockers.append(
                    "body mesh metadata: " + ", ".join(missing["body_mesh_metadata"])
                )
    elif stage == "render":
        if not ready["capture_pair"]:
            for key in ("calibration_image", "reference_pose_image"):
                if not assets[key]["exists"]:
                    blockers.append(f"missing asset: {key}")
            if not readiness["worn_ground_truth_images"] or not all(
                item["exists"] for item in readiness["worn_ground_truth_images"]
            ):
                blockers.append("missing fixed-camera worn ground-truth image")
            if missing["camera"]:
                blockers.append("camera metadata: " + ", ".join(missing["camera"]))
    elif stage == "evaluate":
        if not assets["real_worn_mask"]["exists"]:
            blockers.append("reviewed real_worn_mask is missing")
    return list(dict.fromkeys(blockers))


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    result = {
        "command": command,
        "returncode": completed.returncode,
        "seconds": time.perf_counter() - started,
        "stdout_tail": completed.stdout.splitlines()[-20:],
        "stderr_tail": completed.stderr.splitlines()[-20:],
    }
    if completed.returncode:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def one_specification(pattern_root: Path) -> Path:
    specifications = list(pattern_root.rglob("*_specification.json"))
    if len(specifications) != 1:
        raise RuntimeError(
            f"Expected one GarmentCode specification under {pattern_root}, got {specifications}"
        )
    return specifications[0]


def bending_preset(pilot: dict[str, Any]) -> str:
    value = pilot["garment"]["material"].get("bending_class")
    if value in {"very_soft", "soft"}:
        return "soft"
    if value in {"stiff", "very_stiff"}:
        return "stiff"
    return "medium"


def same_resolved_asset(root: Path, left: str | None, right: str | None) -> bool:
    left_path = resolve_asset(root, left)
    right_path = resolve_asset(root, right)
    return bool(left_path and right_path and left_path.resolve() == right_path.resolve())


def simulation_command(
    blender: Path,
    pilot_path: Path,
    contract: Path,
    body_mesh: Path,
    body_scale_unit: str,
    body_alignment: str,
    out: Path,
    gravity: bool,
    preset: str,
) -> list[str]:
    frames, minimum = (240, 140) if gravity else (160, 110)
    return [
        str(blender),
        "--background",
        "--python",
        str(HERE / "blender_cloth_sim.py"),
        "--",
        "--out",
        str(out),
        "--pilot",
        str(pilot_path),
        "--preset",
        preset,
        "--garment-kind",
        "garmentcode-sewn",
        "--garmentcode-mesh",
        str(contract),
        "--body-mesh",
        str(body_mesh),
        "--body-scale-unit",
        body_scale_unit,
        "--body-alignment",
        body_alignment,
        "--resolve-external-penetration",
        "--frames",
        str(frames),
        "--min-frames",
        str(minimum),
        "--settle-window",
        "10",
        "--settle-consecutive",
        "2",
        "--dress-frames",
        "0",
        "--gravity-scale",
        "1" if gravity else "0",
        "--solver-quality",
        "6",
        "--collision-quality",
        "4",
        "--disable-self-collision",
        "--body-collision-thickness",
        "0.003",
        "--cloth-collision-distance",
        "0.003",
        "--penetration-clearance",
        "0.007",
        "--penetration-iterations",
        "8",
        "--collision-aware-predrape-iterations",
        "800",
        "--collision-aware-predrape-collision-interval",
        "40",
        "--disable-rest-shape-key",
        "--damping-scale",
        "5",
    ]


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--run-id", default="real_gc1_v1")
    parser.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
    parser.add_argument("--through-stage", choices=STAGES, default=STAGES[-1])
    parser.add_argument("--resolution-cm", type=float, default=1.4)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument(
        "--garmentcode-python", type=Path, default=DEFAULT_GARMENTCODE_PYTHON
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-self-evaluation-smoke",
        action="store_true",
        help="Allow the rendered mask to also be the ground truth for wiring tests only.",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id):
        raise ValueError("run-id may contain only letters, digits, dot, underscore, and dash.")
    if STAGES.index(args.from_stage) > STAGES.index(args.through_stage):
        raise ValueError("from-stage cannot come after through-stage.")
    if args.resolution_cm <= 0:
        raise ValueError("resolution-cm must be positive.")

    root = args.root.resolve()
    pilot_path = (args.pilot or root / "pilot.json").resolve()
    schema_path = (args.schema or root / "geometry_pilot.schema.json").resolve()
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(pilot, schema)
    readiness = build_readiness_report(root, pilot)

    run_dir = root / "runs" / args.run_id
    report_path = run_dir / "pipeline_report.json"
    started_at = datetime.now(timezone.utc)
    attempt_id = started_at.strftime("%Y%m%dT%H%M%S.%fZ")
    attempt_report_path = run_dir / "attempts" / f"pipeline_report_{attempt_id}.json"

    def persist_report() -> None:
        report["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_report(attempt_report_path, report)
        write_report(report_path, report)

    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "real_body_garment_geometry_gate",
        "pilot_id": pilot["pilot_id"],
        "run_id": args.run_id,
        "attempt_id": attempt_id,
        "started_at_utc": started_at.isoformat(),
        "attempt_report": str(attempt_report_path),
        "evidence_class": (
            "orchestration_smoke_only"
            if args.allow_self_evaluation_smoke else "real_gc1_candidate"
        ),
        "status": "planned" if args.dry_run else "running",
        "dry_run": args.dry_run,
        "from_stage": args.from_stage,
        "through_stage": args.through_stage,
        "readiness_at_start": readiness,
        "accepted_settings": {
            "resolution_cm": args.resolution_cm,
            "collision_clearance_m": 0.007,
            "body_collision_thickness_m": 0.003,
            "cloth_collision_distance_m": 0.003,
            "predrape_iterations": 800,
            "ray_parity_directions": 3,
        },
        "stages": {},
    }
    planned = STAGES[STAGES.index(args.from_stage): STAGES.index(args.through_stage) + 1]
    report["planned_stages"] = list(planned)

    initial_blockers = stage_blockers(planned[0], readiness)
    if args.dry_run or initial_blockers:
        report["status"] = "planned" if args.dry_run else "awaiting_inputs"
        report["stopped_stage"] = planned[0]
        report["blockers"] = initial_blockers
        persist_report()
        print(json.dumps({
            "status": report["status"],
            "stopped_stage": planned[0],
            "blockers": initial_blockers,
            "report": str(report_path),
            "attempt_report": str(attempt_report_path),
        }, ensure_ascii=False))
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    pattern_root = run_dir / "pattern"
    contract_path = run_dir / "garment_mesh_contract.json"
    control_dir = run_dir / "control"
    gravity_dir = run_dir / "gravity"
    body_preflight_path = run_dir / "body_preflight.json"
    camera_path = run_dir / "camera.json"
    mechanics_path = run_dir / "mechanics_gate.json"

    try:
        if "pattern" in planned:
            if not args.garmentcode_python.exists():
                raise FileNotFoundError(args.garmentcode_python)
            if args.force or not list(pattern_root.rglob("*_specification.json")):
                command = [
                    str(args.garmentcode_python),
                    str(HERE / "garmentcode_adapter.py"),
                    "--pilot",
                    str(pilot_path),
                    "--out",
                    str(pattern_root),
                ]
                report["stages"]["pattern"] = run_command(command, HERE)
            else:
                report["stages"]["pattern"] = {"status": "reused"}

        if "mesh" in planned:
            specification = one_specification(pattern_root)
            if args.force or not contract_path.exists():
                command = [
                    sys.executable,
                    str(HERE / "garmentcode_mesh_contract.py"),
                    "--spec",
                    str(specification),
                    "--pilot",
                    str(pilot_path),
                    "--resolution-cm",
                    str(args.resolution_cm),
                    "--weld-relaxed-preclose",
                    "--out",
                    str(contract_path),
                ]
                report["stages"]["mesh"] = run_command(command, HERE)
            else:
                report["stages"]["mesh"] = {"status": "reused"}
        if any(stage in planned for stage in ("control", "gravity")) and not contract_path.exists():
            raise FileNotFoundError(contract_path)

        if any(stage in planned for stage in ("body", "control", "gravity")):
            body_config = pilot["subject"]["body_mesh"]
            body_mesh = resolve_asset(root, body_config["path"])
            if body_mesh is None:
                raise ValueError("body mesh path is missing")
        if "body" in planned:
            blockers = stage_blockers("body", readiness)
            if blockers:
                report.update(status="awaiting_inputs", stopped_stage="body", blockers=blockers)
                persist_report()
                print(json.dumps({"status": report["status"], "blockers": blockers}, ensure_ascii=False))
                return
            if args.force or not body_preflight_path.exists():
                report["stages"]["body"] = run_command([
                    str(args.blender),
                    "--background",
                    "--python",
                    str(HERE / "blender_body_preflight.py"),
                    "--",
                    "--body",
                    str(body_mesh),
                    "--scale-unit",
                    body_config["scale_unit"],
                    "--out",
                    str(body_preflight_path),
                ], HERE)
            else:
                report["stages"]["body"] = {"status": "reused"}
            body_preflight = json.loads(body_preflight_path.read_text(encoding="utf-8"))
            report["body_preflight"] = {
                "pass": bool(body_preflight["pass"]),
                "checks": body_preflight["checks"],
                "report": str(body_preflight_path),
            }
            if not body_preflight["pass"]:
                report.update(
                    status="gate_failed",
                    stopped_stage="body",
                    blockers=[
                        name for name, passed in body_preflight["checks"].items() if not passed
                    ],
                )
                persist_report()
                return

        if any(stage in planned for stage in ("control", "gravity")):
            preset = bending_preset(pilot)
            for stage, gravity, output in (
                ("control", False, control_dir),
                ("gravity", True, gravity_dir),
            ):
                if stage not in planned:
                    continue
                blockers = stage_blockers(stage, readiness)
                if blockers:
                    report.update(status="awaiting_inputs", stopped_stage=stage, blockers=blockers)
                    persist_report()
                    print(json.dumps({"status": report["status"], "blockers": blockers}, ensure_ascii=False))
                    return
                metadata = output / "simulation_metadata.json"
                if args.force or not metadata.exists():
                    command = simulation_command(
                        args.blender,
                        pilot_path,
                        contract_path,
                        body_mesh,
                        body_config["scale_unit"],
                        body_config["alignment"],
                        output,
                        gravity,
                        preset,
                    )
                    report["stages"][stage] = run_command(command, HERE)
                else:
                    report["stages"][stage] = {"status": "reused"}

        if "render" in planned:
            blockers = stage_blockers("render", readiness)
            if blockers:
                report.update(status="awaiting_inputs", stopped_stage="render", blockers=blockers)
                persist_report()
                print(json.dumps({"status": report["status"], "blockers": blockers}, ensure_ascii=False))
                return
            if args.force or not camera_path.exists():
                report["stages"]["camera"] = run_command([
                    sys.executable,
                    str(HERE / "pilot_camera.py"),
                    "--pilot",
                    str(pilot_path),
                    "--out",
                    str(camera_path),
                ], HERE)
            else:
                report["stages"]["camera"] = {"status": "reused"}
            pilot["capture"]["camera_config"] = relative_or_absolute(root, camera_path)
            pilot_path.write_text(json.dumps(pilot, ensure_ascii=False, indent=2), encoding="utf-8")
            for name, source, accept in (
                ("control_render", control_dir, False),
                ("gravity_render", gravity_dir, True),
            ):
                render_dir = source / "render"
                validation = render_dir / "render_validation.json"
                if args.force or not validation.exists():
                    command = [
                        sys.executable,
                        str(HERE / "run_geometry_render.py"),
                        "--root",
                        str(root),
                        "--pilot",
                        str(pilot_path),
                        "--body",
                        str(source / "body.obj"),
                        "--garment",
                        str(source / "garment_simulated.obj"),
                        "--camera",
                        str(camera_path),
                        "--out",
                        str(render_dir),
                    ]
                    if accept:
                        command.append("--accept")
                    report["stages"][name] = run_command(command, HERE)
                else:
                    report["stages"][name] = {"status": "reused"}
            gravity_render = gravity_dir / "render"
            gravity_validation = json.loads(
                (gravity_render / "render_validation.json").read_text(encoding="utf-8")
            )
            if not gravity_validation["pass"]:
                raise RuntimeError("Gravity render validation did not pass.")
            pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
            pilot["geometry_outputs"].update({
                "garment_silhouette": relative_or_absolute(
                    root, gravity_render / "garment_silhouette.png"
                ),
                "depth": relative_or_absolute(root, gravity_render / "depth.png"),
                "normal": relative_or_absolute(root, gravity_render / "normal.png"),
                "render_metadata": relative_or_absolute(
                    root, gravity_render / "render_metadata.json"
                ),
            })
            pilot_path.write_text(
                json.dumps(pilot, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        if "compare" in planned:
            if args.force or not mechanics_path.exists():
                report["stages"]["compare"] = run_command([
                    sys.executable,
                    str(HERE / "compare_cloth_runs.py"),
                    "--control",
                    str(control_dir),
                    "--gravity",
                    str(gravity_dir),
                    "--out",
                    str(mechanics_path),
                ], HERE)
            else:
                report["stages"]["compare"] = {"status": "reused"}
            mechanics = json.loads(mechanics_path.read_text(encoding="utf-8"))
            report["mechanics_decision"] = {
                "pass": bool(mechanics["pass"]),
                "checks": mechanics["checks"],
                "report": str(mechanics_path),
            }
            if not mechanics["pass"]:
                report.update(
                    status="gate_failed",
                    stopped_stage="compare",
                    blockers=[
                        name for name, passed in mechanics["checks"].items() if not passed
                    ],
                )
                persist_report()
                print(json.dumps({
                    "status": report["status"],
                    "failed_checks": report["blockers"],
                    "report": str(report_path),
                    "attempt_report": str(attempt_report_path),
                }, ensure_ascii=False))
                return

        if "evaluate" in planned:
            pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
            readiness = build_readiness_report(root, pilot)
            blockers = stage_blockers("evaluate", readiness)
            if blockers:
                report.update(status="awaiting_inputs", stopped_stage="evaluate", blockers=blockers)
                report["readiness_after_run"] = readiness
                persist_report()
                print(json.dumps({"status": report["status"], "blockers": blockers}, ensure_ascii=False))
                return
            if same_resolved_asset(
                root,
                pilot["evaluation"]["real_worn_mask"],
                pilot["geometry_outputs"]["garment_silhouette"],
            ):
                if not args.allow_self_evaluation_smoke:
                    report.update(
                        status="gate_failed",
                        stopped_stage="evaluate",
                        blockers=[
                            "real_worn_mask resolves to the rendered mask; independent ground truth is required"
                        ],
                    )
                    persist_report()
                    print(json.dumps({
                        "status": report["status"],
                        "failed_checks": report["blockers"],
                        "report": str(report_path),
                        "attempt_report": str(attempt_report_path),
                    }, ensure_ascii=False))
                    return
                report.setdefault("warnings", []).append(
                    "Self-evaluation was explicitly allowed. This run proves wiring only, not fit."
                )
            report["stages"]["evaluate"] = run_command([
                sys.executable,
                str(HERE / "geometry_eval.py"),
                "--root",
                str(root),
                "--pilot",
                str(pilot_path),
                "--out",
                str(run_dir / "geometry_eval.json"),
                "--overlay",
                str(run_dir / "geometry_overlay.png"),
            ], HERE)
            evaluation = json.loads(
                (run_dir / "geometry_eval.json").read_text(encoding="utf-8")
            )
            report["geometry_decision"] = evaluation["decision"]
            if evaluation["decision"]["status"] != "pass":
                incomplete = evaluation["decision"]["status"] == "incomplete"
                report.update(
                    status="awaiting_inputs" if incomplete else "gate_failed",
                    stopped_stage="evaluate",
                    blockers=(
                        [evaluation["landmarks"].get("reason", "landmark evidence is incomplete")]
                        if incomplete else ["G-C1 geometry thresholds did not pass"]
                    ),
                )
                report["readiness_after_run"] = build_readiness_report(
                    root, json.loads(pilot_path.read_text(encoding="utf-8"))
                )
                persist_report()
                print(json.dumps({
                    "status": report["status"],
                    "geometry_status": evaluation["decision"]["status"],
                    "report": str(report_path),
                    "attempt_report": str(attempt_report_path),
                }, ensure_ascii=False))
                return

        report["status"] = "complete"
        report["readiness_after_run"] = build_readiness_report(
            root, json.loads(pilot_path.read_text(encoding="utf-8"))
        )
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        persist_report()
        raise

    persist_report()
    print(json.dumps({
        "status": report["status"],
        "report": str(report_path),
        "attempt_report": str(attempt_report_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
