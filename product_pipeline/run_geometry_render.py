"""Run the Blender geometry adapter, validate outputs, and optionally update the pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOT = Path(r"fashion_pipeline/out/geometry_pilot_v1")
DEFAULT_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe")
HERE = Path(__file__).resolve().parent


def resolve_asset(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def relative_or_absolute(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--body", type=Path)
    parser.add_argument("--garment", type=Path)
    parser.add_argument("--camera", type=Path)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--resolution", default="1000x1500")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--accept", action="store_true")
    parser.add_argument("--allow-invalid", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    pilot_path = args.pilot or root / "pilot.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    body = args.body or resolve_asset(root, pilot["subject"]["body_mesh"]["path"])
    garment = args.garment or resolve_asset(root, pilot["garment"]["assets"]["pattern_or_mesh"])
    if not args.smoke_test and (not body or not garment):
        raise ValueError("Body and simulated garment mesh paths are required.")
    if not args.blender.exists():
        raise FileNotFoundError(args.blender)

    out = (args.out or root / "renders").resolve()
    command = [
        str(args.blender),
        "--background",
        "--python",
        str(HERE / "blender_geometry_render.py"),
        "--",
        "--out",
        str(out),
        "--resolution",
        args.resolution,
    ]
    if args.smoke_test:
        command.append("--smoke-test")
    else:
        command.extend(("--body", str(body), "--garment", str(garment)))
    if args.camera:
        command.extend(("--camera", str(args.camera.resolve())))
    subprocess.run(command, check=True)
    validation = subprocess.run(
        [sys.executable, str(HERE / "validate_geometry_render.py"), str(out)],
        check=False,
    )
    if validation.returncode and not args.allow_invalid:
        raise subprocess.CalledProcessError(validation.returncode, validation.args)
    validation_report = json.loads((out / "render_validation.json").read_text(encoding="utf-8"))

    if args.accept:
        if args.smoke_test:
            raise ValueError("Smoke-test outputs cannot be accepted into the real pilot.")
        pilot["geometry_outputs"].update(
            {
                "garment_silhouette": relative_or_absolute(root, out / "garment_silhouette.png"),
                "depth": relative_or_absolute(root, out / "depth.png"),
                "normal": relative_or_absolute(root, out / "normal.png"),
                "render_metadata": relative_or_absolute(root, out / "render_metadata.json"),
            }
        )
        pilot_path.write_text(json.dumps(pilot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "validated": validation_report["pass"],
                "accepted_into_pilot": args.accept,
                "out": str(out),
            }
        )
    )


if __name__ == "__main__":
    main()
