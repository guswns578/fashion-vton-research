"""End-to-end batch: catalog flat + worn photo of a DIFFERENT product -> IDM-VTON -> colour + texture refine.

  python batch_tryon.py [case ...]      (no args = all cases)

Per case (work/batch/<case>/): human.jpg (3:4 upper-body crop), garment.jpg (3:4 pad), mask.png (SAM on
the current clothing, optionally unioned with 'arm' for sleeveless/short-sleeve -> long-sleeve), pose.png,
IDM result outputs/batch/<case>_idm.png, refine via refine_texture.run(). Steps whose output exists are skipped.
"""

import json, subprocess, sys, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "work"))
import refine_texture as RT

PY = str(ROOT / "work/runtime/Scripts/python.exe")
RUNNER = str(ROOT / "ComfyUI-IDM-VTON-12GB/runner.py")
DS = Path(r"fashion_pipeline/dataset/catalog_curated/tops_v1")
RAW = Path(r"fashion_pipeline/dataset/catalog_raw/tops")
COMFY = r"C:\Dev\StabilityMatrix\Data\Packages\ComfyUI"
WB = ROOT / "work" / "batch"
OB = ROOT / "outputs" / "batch"
STUDIO = ROOT / "work" / "sample" / "human.jpg"

CASES = {
    # person: catalog image id or STUDIO; wear: what the person wears now (GroundingDINO prompt); arm: add arm mask
    "check_shirt__closeup": dict(
        person="p1922/detail/005.jpg",
        wear="sweater",
        arm=False,
        garment="p2712/detail/033.jpg",
        cut=0.0,
        desc="a navy and white gingham check short sleeve cotton shirt with roll-up cuffs",
        sam="shirt",
        prompt="navy and white small gingham check cotton shirt, crisp yarn-dyed woven cotton, fine check pattern, short sleeves with roll-up cuffs, buttons",
        d=0.45,
    ),
    "fleece_anorak__sitting": dict(
        person="p2603/detail/018.jpg",
        wear="hoodie",
        arm=False,
        garment="p2650/detail/026.jpg",
        cut=0.14,
        desc="a purple fleece half snap pullover anorak with a chest flap pocket",
        sam="jacket",
        prompt="purple polar fleece half snap pullover, soft brushed fleece pile, fluffy fleece surface, nylon flap chest pocket, snap buttons",
        d=0.5,
    ),
    "alpaca_cardigan__angle": dict(
        person="p2090/detail/027.jpg",
        wear="sweater",
        arm=False,
        garment="p2229/detail/053.jpg",
        cut=0.14,
        desc="a red brushed alpaca wool knit cardigan with buttons",
        sam="cardigan",
        prompt="red brushed alpaca wool cardigan, long hairy fibers, soft fuzzy halo, loose knit, buttons",
        d=0.5,
    ),
    "ringer_tee__pockets": dict(
        person="p2288/detail/010.jpg",
        wear="jacket",
        arm=False,
        garment="p2483/detail/025.jpg",
        cut=0.14,
        desc="a heather grey ringer t-shirt with navy rib neck and sleeve bands and a vintage red and navy college graphic print",
        sam="t-shirt",
        prompt="heather grey cotton ringer t-shirt, contrast navy ribbed neckband and sleeve bands, vintage cracked screen print graphic, soft cotton jersey knit",
        d=0.4,
    ),
    "stripe_cardigan__studio": dict(
        person=STUDIO,
        wear="shirt",
        arm=True,
        garment="p2360/detail/020.jpg",
        cut=0.14,
        desc="a white and grey horizontal stripe knit cardigan with buttons, long sleeves",
        sam="cardigan",
        prompt="white and grey horizontal stripe knit cardigan, fine gauge cotton knit, ribbed hem and cuffs, buttons",
        d=0.4,
    ),
    "black_henley__bench": dict(
        person="p2712/detail/004.jpg",
        wear="shirt",
        arm=True,
        garment="p2757/detail/076.jpg",
        cut=0.0,
        desc="a black slub cotton henley long sleeve t-shirt with a button placket",
        sam="shirt",
        prompt="black slub cotton henley long sleeve t-shirt, irregular slub jersey texture, soft matte cotton, button placket",
        d=0.5,
    ),
}


def meta(rel):
    for l in open(DS / "images.jsonl", encoding="utf-8"):
        d = json.loads(l)
        if d["image_id"] == rel:
            return d
    raise SystemExit(f"{rel} not in images.jsonl")


def person_crop(rel):
    """3:4 upper-body crop. Close-ups (person cut off at the bottom) keep everything down to the frame edge."""
    im = Image.open(RAW / rel).convert("RGB")
    x0, y0, x1, y1 = meta(rel)["person_box"]
    ph = y1 - y0
    top = max(0, y0 - 0.04 * ph)
    closeup = (
        y1 > 0.95 * im.height and (x1 - x0) / ph > 0.5
    )  # cut off at the bottom AND wide = upper-body shot
    bot = (
        im.height if closeup else min(im.height, y0 + 0.62 * ph)
    )  # (a full-body shot with feet at the edge is narrow)
    h = bot - top
    w = h * 3 / 4
    if w > im.width:  # too wide: shrink height from the bottom
        w = im.width
        h = w * 4 / 3
        bot = top + h
    left = min(max(0, (x0 + x1) / 2 - w / 2), im.width - w)
    return im.crop((int(left), int(top), int(left + w), int(bot))).resize(
        (768, 1024), Image.LANCZOS
    )


def drop_label(im):
    """Product shots carry a 'color RED' caption under the garment. Keep only the longest run of
    foreground rows (the garment); the caption is a separate short run below a background gap."""
    a = np.asarray(im, np.float32)
    edge = np.concatenate([a[:8].reshape(-1, 3), a[-8:].reshape(-1, 3)])
    fg = np.abs(a - np.median(edge, 0)).sum(2) > 30
    rows = fg.mean(1) > 0.005
    runs, start = [], None
    for y, r in enumerate(list(rows) + [False]):
        if r and start is None:
            start = y
        if not r and start is not None:
            runs.append((start, y))
            start = None
    if not runs:
        return im
    y0, y1 = max(runs, key=lambda t: t[1] - t[0])
    m = int(0.03 * im.height)
    return im.crop((0, max(0, y0 - m), im.width, min(im.height, y1 + m)))


def garment_img(rel, cut):
    im = Image.open(RAW / rel).convert("RGB")
    if cut:
        im = im.crop((0, 0, im.width, int(im.height * (1 - cut))))
    im = drop_label(im)
    a = np.asarray(im)
    edge = np.concatenate(
        [
            a[:8].reshape(-1, 3),
            a[-8:].reshape(-1, 3),
            a[:, :8].reshape(-1, 3),
            a[:, -8:].reshape(-1, 3),
        ]
    )
    fill = tuple(int(v) for v in np.median(edge, 0))
    w, h = im.size
    tw, th = (w, round(w * 4 / 3)) if h / w < 4 / 3 else (round(h * 3 / 4), h)
    c = Image.new("RGB", (tw, th), fill)
    c.paste(im, ((tw - w) // 2, (th - h) // 2))
    return c.resize((768, 1024), Image.LANCZOS)


def preprocess(d, prompt, mask_name, threshold=0.3):
    job = {
        "comfy": COMFY,
        "human": str(d / "human.jpg"),
        "prompt": prompt,
        "threshold": threshold,
        "grow": 12,
        "mask": str(d / mask_name),
        "pose": str(d / "pose.png"),
    }
    jp = d / f"pre_{prompt}.json"
    jp.write_text(json.dumps(job))
    r = subprocess.run(
        [PY, RUNNER, "preprocess", str(jp)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if r.returncode:
        raise SystemExit(f"preprocess {prompt} failed:\n{r.stderr[-2000:]}")


def preprocess_arm(d):
    """Small / far-away people: 'arm' at 0.3 can find nothing, so step the threshold down."""
    for prompt, th in (("arm", 0.3), ("arm", 0.22), ("arms", 0.18)):
        try:
            preprocess(d, prompt, "mask_arm.png", th)
            return
        except SystemExit:
            continue
    raise SystemExit("no arm detected even at 0.18")


def free_comfy():
    r = urllib.request.Request(
        "http://127.0.0.1:8188/free",
        data=b'{"unload_models":true,"free_memory":true}',
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(r, timeout=30)


def do_case(name, c):
    d = WB / name
    d.mkdir(parents=True, exist_ok=True)
    OB.mkdir(parents=True, exist_ok=True)
    if not (d / "human.jpg").exists():
        (
            Image.open(c["person"]).convert("RGB").resize((768, 1024))
            if isinstance(c["person"], Path)
            else person_crop(c["person"])
        ).save(d / "human.jpg", quality=95)
        garment_img(c["garment"], c["cut"]).save(d / "garment.jpg", quality=95)
    if not (d / "mask.png").exists():
        preprocess(d, c["wear"], "mask_wear.png")
        m = np.asarray(Image.open(d / "mask_wear.png").convert("L")) > 127
        if c["arm"]:
            preprocess_arm(d)
            m |= np.asarray(Image.open(d / "mask_arm.png").convert("L")) > 127
        Image.fromarray((m * 255).astype(np.uint8)).save(d / "mask.png")
    idm = OB / f"{name}_idm.png"
    if not idm.exists():
        free_comfy()
        job = {
            "comfy": COMFY,
            "human": str(d / "human.jpg"),
            "garment": str(d / "garment.jpg"),
            "pose": str(d / "pose.png"),
            "mask": str(d / "mask.png"),
            "output": str(idm),
            "description": c["desc"],
            "width": 768,
            "height": 1024,
            "steps": 20,
            "seed": 42,
        }
        (d / "infer.json").write_text(json.dumps(job))
        r = subprocess.run(
            [PY, RUNNER, "infer", str(d / "infer.json")],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if r.returncode:
            raise SystemExit(f"infer failed:\n{r.stderr[-2000:]}")
    info = RT.run(
        name,
        dict(vton=idm, garment=d / "garment.jpg", prompt=c["prompt"], sam_prompt=c["sam"]),
        denoise=c["d"],
        cn=0.4,
        tag=f"d{c['d']:g}",
    )
    return info


if __name__ == "__main__":
    names = sys.argv[1:] or list(CASES)
    for n in names:
        print("==", n, flush=True)
        do_case(n, CASES[n])
