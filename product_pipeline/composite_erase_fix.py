"""Recomposite a FASHN run with the erase-region feather bug fixed.

catalog_eval.hard_composite blends with feather_inside(union, radius=2.0), a
Gaussian ramp that rises from 0 at the union edge inward. Inside a union band
narrower than roughly 12px the ramp never reaches 1, so those pixels stay a
partial blend of base and FASHN output.

The `old \\ new` erase strip -- the area vacated when the product garment is
shorter or narrower than whatever the base was wearing -- is exactly such a thin
band. Blending there reintroduces the OLD garment at up to ~50% alpha, which
reads as a coloured sliver along the sleeve hem plus a seam across the arm.

Fix: any pixel that contained old garment must be taken from FASHN at full
strength. Feathering is only ever legitimate against pixels the base legitimately
keeps, never against pixels we are trying to erase.

    alpha = feather_inside(union)
    alpha[old_garment] = 1.0

Outside the union nothing changes, so the byte-identical preservation guarantee
on face / hair / hands / trousers / background still holds and is re-measured.

Reads only existing run artifacts -- no GPU, no FASHN re-run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ID_TO_LABEL = {
    0: "background",
    1: "face",
    2: "hair",
    3: "top",
    4: "dress",
    5: "skirt",
    6: "pants",
    7: "belt",
    8: "bag",
    9: "hat",
    10: "scarf",
    11: "glasses",
    12: "arms",
    13: "hands",
    14: "legs",
    15: "feet",
    16: "torso",
    17: "jewelry",
}
LABEL_IDS = {name: i for i, name in ID_TO_LABEL.items()}
# Generated pixels are never taken on these, per the pipeline contract.
NEVER_TOUCH = (
    "face",
    "hair",
    "glasses",
    "hands",
    "pants",
    "belt",
    "legs",
    "feet",
    "bag",
    "hat",
    "jewelry",
)


def feather_inside(binary: np.ndarray, radius: float = 2.0) -> np.ndarray:
    """Same ramp as the main line, reproduced so the two can be compared."""
    b = (binary > 0).astype(np.uint8) * 255
    soft = (
        np.asarray(Image.fromarray(b, "L").filter(ImageFilter.GaussianBlur(radius)), np.float32)
        / 255.0
    )
    return soft * (binary > 0)


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--radius", type=float, default=2.0)
    ap.add_argument("--tag", default="v6")
    ap.add_argument(
        "--old-dilate",
        type=int,
        default=0,
        help=(
            "grow the full-strength region this many px past the old-garment mask. "
            "With 0 the alpha=1 region ends exactly on the parser mask edge and its "
            "pixel-level jaggedness becomes visible; a small margin moves the feather "
            "onto skin/background, where base and FASHN already agree."
        ),
    )
    ap.add_argument(
        "--skin-safe",
        action="store_true",
        help="do not grow the hard region into skin labels (needs 15_person_labels.png)",
    )
    ap.add_argument("--skin-labels", default="arms,hands,face,legs,feet")
    ap.add_argument(
        "--new-dilate",
        type=int,
        default=0,
        help="grow the full-strength region past the NEW garment mask, "
        "gated on FASHN actually having painted there (additive case)",
    )
    ap.add_argument(
        "--diff-thr",
        type=int,
        default=18,
        help="raw-vs-base delta that counts as FASHN having painted",
    )
    ap.add_argument(
        "--hard-union",
        action="store_true",
        help="take FASHN at full strength across the entire union (no inward feather)",
    )
    ap.add_argument(
        "--outer-feather",
        type=int,
        default=0,
        help=(
            "alpha=1 across the whole union, ramping to 0 over N px OUTSIDE it. "
            "Anti-aliases the mask edge without blending base content back in. "
            "The protected region becomes everything beyond the N px collar."
        ),
    )
    args = ap.parse_args()

    run = args.run_dir
    base = np.asarray(Image.open(run / "01_person_original.png").convert("RGB"), np.float32)
    raw = np.asarray(Image.open(run / "05_fashn_raw_fullframe.png").convert("RGB"), np.float32)
    old = load_mask(run / "06_mask_old_garment.png")
    new = load_mask(run / "07_mask_new_garment.png")

    hard = old
    if args.old_dilate > 0:
        m = Image.fromarray((old * 255).astype(np.uint8), "L")
        for _ in range(args.old_dilate):
            m = m.filter(ImageFilter.MaxFilter(3))
        hard = np.asarray(m) > 127

        # Dilating outward catches old-garment pixels the parser missed at the
        # silhouette edge. But growing into SKIN is harmful: FASHN repaints arms
        # with a slightly different silhouette than the base, so forcing FASHN
        # over a wide skin band exposes that disagreement as a jagged seam.
        # Grow over background and garment only.
        labels_path = run / "15_person_labels.png"
        if args.skin_safe and labels_path.exists():
            seg = np.asarray(Image.open(labels_path).convert("L")).astype(int)
            skin = np.zeros_like(old, dtype=bool)
            for name in [s.strip() for s in args.skin_labels.split(",") if s.strip()]:
                if name in LABEL_IDS:
                    skin |= seg == (LABEL_IDS[name] * 14) % 255
            hard &= ~(skin & ~old)

    if args.new_dilate > 0:
        # Mirror case of --old-dilate, and the one --skin-safe must NOT block.
        # With a sleeveless base and a long-sleeve product the parser under-covers
        # the sleeve's outer silhouette; those pixels fall outside the union and
        # get restored from a base that is BARE SKIN there, leaving a skin crescent
        # on top of the sleeve. Growing into "arms" is exactly right here.
        #
        # The discriminator is therefore not "is this skin" but "did FASHN actually
        # paint something different here", intersected with the labels the pipeline
        # is allowed to overwrite at all.
        m = Image.fromarray((new * 255).astype(np.uint8), "L")
        for _ in range(args.new_dilate):
            m = m.filter(ImageFilter.MaxFilter(3))
        band = (np.asarray(m) > 127) & ~new

        changed = np.abs(raw - base).max(axis=2) > args.diff_thr

        allowed = np.ones_like(new, dtype=bool)
        labels_path = run / "15_person_labels.png"
        if labels_path.exists():
            seg = np.asarray(Image.open(labels_path).convert("L")).astype(int)
            for name in NEVER_TOUCH:
                allowed &= seg != (LABEL_IDS[name] * 14) % 255

        grow = band & changed & allowed
        hard = hard | grow

    # The composite region must cover the grown hard region, otherwise the margin
    # would write outside the area we claim to touch.
    union = old | new | hard

    alpha = feather_inside(union, args.radius)
    alpha_fixed = alpha.copy()
    if args.outer_feather > 0:
        # Ramp OUTWARD instead of inward. Inside the union the base always holds
        # something being replaced (old garment, or bare skin under an added
        # sleeve), so any inward blend contaminates. Just outside the garment edge
        # base and FASHN agree -- same background, same arm -- so a ramp there
        # anti-aliases the mask step without pulling in anything wrong.
        n = args.outer_feather
        soft = (
            np.asarray(
                Image.fromarray((union * 255).astype(np.uint8), "L").filter(
                    ImageFilter.GaussianBlur(n / 2.0)
                ),
                np.float32,
            )
            / 255.0
        )
        big = Image.fromarray((union * 255).astype(np.uint8), "L")
        for _ in range(n):
            big = big.filter(ImageFilter.MaxFilter(3))
        outside_limit = np.asarray(big) > 127

        alpha_fixed = np.clip(soft, 0.0, 1.0)
        alpha_fixed[union] = 1.0
        alpha_fixed[~outside_limit] = 0.0
        union = outside_limit  # the region we now claim to touch
    elif args.hard_union:
        # Feathering inside the union always blends the BASE back in, and at every
        # union boundary the base holds something we are trying to replace: old
        # garment on the erase side, bare skin on the additive side. Forcing alpha
        # to 1 across the whole union is the only rule that is correct on both.
        alpha_fixed[union] = 1.0
    else:
        alpha_fixed[hard] = 1.0

    # Final invariant: parser disagreement can put a few face/hair/hand pixels
    # inside old/new garment masks, especially on a 2x canvas.  The union used
    # to authorize those pixels silently, violating the advertised identity
    # preservation contract.  Protection wins over every garment-mask rule.
    protected = np.zeros_like(union, dtype=bool)
    labels_path = run / "15_person_labels.png"
    if labels_path.exists():
        seg = np.asarray(Image.open(labels_path).convert("L")).astype(int)
        for name in NEVER_TOUCH:
            protected |= seg == (LABEL_IDS[name] * 14) % 255
        alpha_fixed[protected] = 0.0
        union = union & ~protected

    out = np.rint(base * (1 - alpha_fixed[..., None]) + raw * alpha_fixed[..., None])
    out = out.clip(0, 255).astype(np.uint8)
    final = Image.fromarray(out, "RGB")
    final_path = run / f"20_final_{args.tag}.png"
    final.save(final_path)

    # Re-prove the protection guarantee rather than trusting the argument.
    b8 = base.astype(np.int16)
    f8 = out.astype(np.int16)
    diff = np.abs(b8 - f8).max(axis=2)
    outside = ~union
    changed_outside = int((diff[outside] > 0).sum())

    erase = old & ~new
    prev = np.asarray(Image.open(run / "11_final_hard_composite.png").convert("RGB"), np.int16)
    raw_i = raw.astype(np.int16)
    erase_gap_before = int((np.abs(raw_i - prev).max(axis=2)[erase] > 0).sum())
    erase_gap_after = int((np.abs(raw_i - f8).max(axis=2)[erase] > 0).sum())

    report = {
        "tag": args.tag,
        "feather_radius": args.radius,
        "rule": "alpha = feather_inside(union); alpha[old_garment] = 1.0",
        "union_pixels": int(union.sum()),
        "erase_pixels_old_minus_new": int(erase.sum()),
        "erase_pixels_not_taking_fashn_before": erase_gap_before,
        "erase_pixels_not_taking_fashn_after": erase_gap_after,
        "outside_pixels": int(outside.sum()),
        "outside_changed_pixels": changed_outside,
        "byte_identical_outside": changed_outside == 0,
        "protected_pixels": int(protected.sum()),
        "protected_changed_pixels": int((diff[protected] > 0).sum()),
        "protected_byte_identical": bool((diff[protected] > 0).sum() == 0),
        "output": str(final_path),
    }
    (run / f"20_report_{args.tag}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
