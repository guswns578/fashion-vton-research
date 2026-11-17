"""Run the full generative chain over a base x garment matrix and score every cell.

Only one (base, garment) combination has been verified end to end. This widens
the garment axis so we can tell whether the defects seen on that one cell are
product-specific or systematic -- in particular the p2022 button count dropping
from 2 to 1, which is meaningless as a single observation.

Per cell: FASHN segfree -> residual gate (for the parse labels) -> v13 composite,
then record the numbers that decide §9:

  erase_gap        old-garment pixels not taking FASHN output   (must be 0)
  protected_ok     pixels outside the touched region unchanged  (must be true)
  identity         SFace cosine of the final vs the 10 holdout photos
  identity_delta   identity(final) - identity(base); ~0 means VTON preserved it

Nothing here decides pass/fail on its own. Automated gates have already been
wrong three times in this project -- most recently `residual_garment_gate.py`
returning PASS on a visibly grey-on-grey residue. The table exists to point at
the cells worth opening, not to replace looking at them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

WORK = Path(__file__).parent
FASHN_PY = Path(r"C:\Dev\fashn-vton-1.5\.venv\Scripts\python.exe")


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (p.stdout or "") + (p.stderr or "")


def identity_of(image: Path) -> float | None:
    out = run([str(FASHN_PY), str(WORK / "identity_score.py"), "--generated", str(image)])
    for line in out.splitlines():
        if "[verdict] generated mean=" in line:
            try:
                return float(line.split("mean=")[1].split()[0])
            except (IndexError, ValueError):
                return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", type=Path, required=True)
    ap.add_argument(
        "--garments", nargs="+", required=True, help="id=path pairs, e.g. p2022=path/to/031.jpg"
    )
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    garments = {}
    for g in args.garments:
        gid, _, gpath = g.partition("=")
        garments[gid] = Path(gpath)

    args.out_root.mkdir(parents=True, exist_ok=True)
    cells = []

    for base in args.bases:
        base_id = base.stem.replace("_detail0.5", "")
        base_identity = identity_of(base)
        for gid, gpath in garments.items():
            run_dir = args.out_root / f"{base_id}__{gid}"
            print(f"\n=== {base_id} x {gid} ===", flush=True)

            run(
                [
                    str(FASHN_PY),
                    str(WORK / "run_fashn_pair_v2.py"),
                    "--person",
                    str(base),
                    "--garment",
                    str(gpath),
                    "--out-dir",
                    str(run_dir),
                    "--seed",
                    str(args.seed),
                ]
            )
            run([str(FASHN_PY), str(WORK / "residual_garment_gate.py"), "--run-dir", str(run_dir)])
            comp = run(
                [
                    str(FASHN_PY),
                    str(WORK / "composite_erase_fix.py"),
                    "--run-dir",
                    str(run_dir),
                    "--old-dilate",
                    "4",
                    "--skin-safe",
                    "--new-dilate",
                    "6",
                    "--outer-feather",
                    "3",
                    "--tag",
                    "v13",
                ]
            )

            final = run_dir / "20_final_v13.png"
            cell = {
                "base": base_id,
                "garment": gid,
                "run_dir": str(run_dir),
                "final": str(final) if final.exists() else None,
            }

            try:
                rep = json.loads(comp[comp.index("{") : comp.rindex("}") + 1])
                cell["erase_gap"] = rep.get("erase_pixels_not_taking_fashn_after")
                cell["protected_ok"] = rep.get("byte_identical_outside")
                cell["touched_px"] = rep.get("union_pixels")
            except (ValueError, json.JSONDecodeError):
                cell["error"] = "composite report unparsable"

            if final.exists():
                fid = identity_of(final)
                cell["identity_final"] = fid
                cell["identity_base"] = base_identity
                if fid is not None and base_identity is not None:
                    cell["identity_delta"] = round(fid - base_identity, 4)

            cells.append(cell)
            print(json.dumps(cell, ensure_ascii=False), flush=True)

    (args.out_root / "matrix.json").write_text(
        json.dumps(cells, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n{'base':28s} {'garment':8s} {'erase':>6s} {'prot':>5s} {'ident':>7s} {'delta':>7s}")
    for c in cells:
        print(
            f"{c['base'][:28]:28s} {c['garment']:8s} "
            f"{str(c.get('erase_gap','?')):>6s} {str(c.get('protected_ok','?')):>5s} "
            f"{str(c.get('identity_final','?')):>7s} {str(c.get('identity_delta','?')):>7s}"
        )
    print(f"\n[done] {len(cells)} cells -> {args.out_root}")


if __name__ == "__main__":
    main()
