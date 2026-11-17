"""Prepare IDM-VTON jobs from the catalog curated tops dataset (garment_flat product shots).
Garments are padded (not stretched) to 3:4 with their own border colour; street persons are cropped
to the upper body using the dataset's person_box. Writes work/catalog_vton/<case>/{garment,human}.jpg
and preprocess/infer job JSONs."""

import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DS = Path(r"fashion_pipeline/dataset/catalog_curated/tops_v1")
RAW = Path(r"fashion_pipeline/dataset/catalog_raw/tops")
COMFY = r"C:\Dev\StabilityMatrix\Data\Packages\ComfyUI"
OUT = ROOT / "work" / "catalog_vton"


def pad_34(im, fill):
    w, h = im.size
    tw, th = (w, round(w * 4 / 3)) if h / w < 4 / 3 else (round(h * 3 / 4), h)
    c = Image.new("RGB", (tw, th), fill)
    c.paste(im, ((tw - w) // 2, (th - h) // 2))
    return c


def border_colour(im):
    a = np.asarray(im).reshape(-1, 3) if False else np.asarray(im)
    edge = np.concatenate(
        [
            a[:8].reshape(-1, 3),
            a[-8:].reshape(-1, 3),
            a[:, :8].reshape(-1, 3),
            a[:, -8:].reshape(-1, 3),
        ]
    )
    return tuple(int(v) for v in np.median(edge, 0))


def garment(rel, crop_bottom=0.0):
    im = Image.open(RAW / rel).convert("RGB")
    if crop_bottom:
        im = im.crop((0, 0, im.width, int(im.height * (1 - crop_bottom))))
    return pad_34(im, border_colour(im)).resize((768, 1024), Image.LANCZOS)


def street_person(rel):
    """Upper-body 3:4 crop: from above the head down to ~62% of the person box height."""
    meta = next(
        json.loads(l)
        for l in open(DS / "images.jsonl", encoding="utf-8")
        if json.loads(l)["image_id"] == rel
    )
    im = Image.open(RAW / rel).convert("RGB")
    x0, y0, x1, y1 = meta["person_box"]
    ph = y1 - y0
    top = max(0, y0 - 0.04 * ph)
    bot = min(im.height, y0 + 0.62 * ph)
    h = bot - top
    w = h * 3 / 4
    cx = (x0 + x1) / 2
    left = min(max(0, cx - w / 2), im.width - w)
    if left < 0:
        raise SystemExit(f"{rel}: crop wider than image")
    return im.crop((int(left), int(top), int(left + w), int(bot))).resize(
        (768, 1024), Image.LANCZOS
    )


CASES = [
    # name, human source, current-clothing prompt, garment rel path, crop_bottom, description
    (
        "studio_p2589_denim_shirt",
        ROOT / "work" / "sample" / "human.jpg",
        "shirt",
        "p2589/detail/022.jpg",
        0.0,
        "a light blue denim western shirt with two chest flap pockets and snap buttons, long sleeves",
    ),
    (
        "studio_p2610_fisherman_knit",
        ROOT / "work" / "sample" / "human.jpg",
        "shirt",
        "p2610/detail/031.jpg",
        0.15,
        "a grey cable knit fisherman crew neck sweater, long sleeves",
    ),
    (
        "street_p2610person_p2589_denim_shirt",
        "p2610/detail/022.jpg",
        "sweater",
        "p2589/detail/022.jpg",
        0.0,
        "a light blue denim western shirt with two chest flap pockets and snap buttons, long sleeves",
    ),
]

for name, human, prompt, grel, cb, desc in CASES:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    (
        Image.open(human).convert("RGB").resize((768, 1024))
        if isinstance(human, Path)
        else street_person(human)
    ).save(d / "human.jpg", quality=95)
    garment(grel, cb).save(d / "garment.jpg", quality=95)
    json.dump(
        {
            "comfy": COMFY,
            "human": str(d / "human.jpg"),
            "prompt": prompt,
            "threshold": 0.3,
            "grow": 12,
            "mask": str(d / "mask.png"),
            "pose": str(d / "pose.png"),
        },
        open(d / "pre.json", "w"),
    )
    json.dump(
        {
            "comfy": COMFY,
            "human": str(d / "human.jpg"),
            "garment": str(d / "garment.jpg"),
            "pose": str(d / "pose.png"),
            "mask": str(d / "mask.png"),
            "output": str(ROOT / "outputs" / f"catalog_{name}.png"),
            "description": desc,
            "width": 768,
            "height": 1024,
            "steps": 20,
            "seed": 42,
            "source_garment": str(RAW / grel),
            "source_human": str(human if isinstance(human, Path) else RAW / human),
        },
        open(d / "infer.json", "w"),
        indent=1,
    )
    print("prepared", name)
