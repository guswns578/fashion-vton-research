"""Validate Blender geometry-render outputs before using them as conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(path) as source:
        array = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
        size = source.size
    return array, size


def image_stats(path: Path, foreground: np.ndarray | None = None) -> dict:
    array, size = load_image(path)
    nonblack = np.any(array > 1 / 255, axis=2)
    result = {
        "path": str(path.resolve()),
        "size": list(size),
        "nonblack_fraction": float(nonblack.mean()),
        "channel_mean": [float(value) for value in array.mean(axis=(0, 1))],
        "channel_std": [float(value) for value in array.std(axis=(0, 1))],
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }
    if foreground is not None and foreground.any():
        values = array[foreground]
        result["foreground_channel_std"] = [float(value) for value in values.std(axis=0)]
        result["foreground_channel_range"] = [
            float(value) for value in (values.max(axis=0) - values.min(axis=0))
        ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    metadata = json.loads((root / "render_metadata.json").read_text(encoding="utf-8"))
    paths = {name: Path(value) for name, value in metadata["outputs"].items()}
    silhouette, _ = load_image(paths["garment_silhouette"])
    foreground = silhouette.mean(axis=2) >= 0.5
    stats = {
        "garment_silhouette": image_stats(paths["garment_silhouette"]),
        "depth": image_stats(paths["depth"], foreground),
        "normal": image_stats(paths["normal"], foreground),
    }
    expected_size = metadata["resolution"]
    depth_std = stats["depth"].get("foreground_channel_std", [])
    normal_std = stats["normal"].get("foreground_channel_std", [])
    checks = {
        "files_exist": all(path.exists() for path in paths.values()),
        "matching_size": all(item["size"] == expected_size for item in stats.values()),
        "silhouette_nonblank": 0.001 < stats["garment_silhouette"]["nonblack_fraction"] < 0.95,
        "depth_nonconstant": bool(depth_std) and max(depth_std) > 0.01,
        "normal_nonconstant": bool(normal_std) and max(normal_std) > 0.01,
    }
    report = {
        "schema_version": 1,
        "checks": checks,
        "pass": all(checks.values()),
        "stats": stats,
    }
    out = args.out or root / "render_validation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"pass": report["pass"], **checks}))
    print(out)
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
