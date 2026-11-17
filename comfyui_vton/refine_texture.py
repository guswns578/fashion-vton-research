"""Post-VTON garment refinement, judged by eye.

  1. colour: Lab transfer inside the garment region toward the product flat (a/b mean+std, L mean only,
     so the VTON shading/structure is kept and only the hue/saturation drift is removed).
  2. texture: crop the garment bbox, upscale x2 (Lanczos), SDXL img2img on ComfyUI 8188 with
     Union ProMax in TILE mode (locks structure: seams, pockets, buttons) + DifferentialDiffusion with a
     soft garment mask (per-pixel strength, no hard seam), then paste back into the original frame.
     Face/background never enter the sampler's changed region.

Garment region = pixels the VTON changed (|person - vton|, same trick as enhance.py), closed + feathered.

  python refine_texture.py <case> [--denoise 0.4] [--cn 0.6] [--tag t]
Cases are defined in CASES below. Outputs: outputs/refine/<case>_{color,refined_<tag>}.png and a contact sheet.
"""

import argparse
import json
import time
import urllib.request
import uuid
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent
COMFY = Path(r"C:\Dev\StabilityMatrix\Data\Packages\ComfyUI")
OUT = ROOT / "outputs" / "refine"
HV = ROOT / "work" / "catalog_vton"

CASES = {
    "studio_denim": dict(
        person=HV / "studio_p2589_denim_shirt" / "human.jpg",
        vton=ROOT / "outputs" / "catalog_studio_p2589_denim_shirt.png",
        garment=HV / "studio_p2589_denim_shirt" / "garment.jpg",
        prompt="light wash chambray denim shirt, soft worn cotton denim, faint vertical slub streaks, uneven stone-wash fading, fine fabric grain, flap pockets, metal snap buttons, topstitching",
    ),
    "studio_knit": dict(
        person=HV / "studio_p2610_fisherman_knit" / "human.jpg",
        vton=ROOT / "outputs" / "catalog_studio_p2610_fisherman_knit.png",
        garment=HV / "studio_p2610_fisherman_knit" / "garment.jpg",
        sam_prompt="sweater",
        prompt="grey wool fisherman sweater, chunky cable knit, honeycomb stitch, raised knitted yarn texture, fuzzy wool fibers, ribbed crew neck and cuffs",
    ),
    "street_denim": dict(
        person=HV / "street_p2610person_p2589_denim_shirt" / "human.jpg",
        vton=ROOT / "outputs" / "catalog_street_p2610person_p2589_denim_shirt.png",
        garment=HV / "street_p2610person_p2589_denim_shirt" / "garment.jpg",
        prompt="light wash chambray denim shirt, soft worn cotton denim, faint vertical slub streaks, uneven stone-wash fading, fine fabric grain, flap pockets, metal snap buttons, topstitching",
    ),
}
POS_TAIL = ", photo, natural daylight, sharp focus, high detail fabric close-up, realistic"
NEG = "smooth plastic fabric, blurry, lowres, cartoon, painting, cgi, oversaturated, distorted buttons, text, watermark, quilted, diamond pattern, herringbone, checkered, plaid, embossed pattern, printed pattern"


# ---------------------------------------------------------------- masks / colour
def changed_mask(person, vton, thr=10, close=11):
    d = np.abs(np.asarray(person, np.float32) - np.asarray(vton, np.float32)).mean(2)
    m = (
        Image.fromarray(((d > thr) * 255).astype(np.uint8))
        .filter(ImageFilter.MaxFilter(close))
        .filter(ImageFilter.MinFilter(close))
    )
    a = np.asarray(m) > 127
    # keep the largest blob-ish area: drop tiny speckles by an opening
    m = (
        Image.fromarray((a * 255).astype(np.uint8))
        .filter(ImageFilter.MinFilter(5))
        .filter(ImageFilter.MaxFilter(5))
    )
    return m


def sam_garment_mask(vton_path, prompt, out_path):
    """GroundingDINO + SAM on the VTON OUTPUT (runner.py preprocess, grow 0). The difference mask
    caught hands/face that IDM re-rendered, and colour transfer then turned the hands blue-grey."""
    import subprocess

    if not out_path.exists():
        job = {
            "comfy": str(COMFY),
            "human": str(vton_path),
            "prompt": prompt,
            "threshold": 0.3,
            "grow": 0,
            "mask": str(out_path),
            "pose": str(out_path.with_name(out_path.stem + "_pose.png")),
        }
        jp = out_path.with_suffix(".json")
        jp.write_text(json.dumps(job), encoding="utf-8")
        subprocess.run(
            [
                str(ROOT / "work/runtime/Scripts/python.exe"),
                str(ROOT / "ComfyUI-IDM-VTON-12GB/runner.py"),
                "preprocess",
                str(jp),
            ],
            check=True,
            capture_output=True,
        )
    return Image.open(out_path).convert("L")


def garment_pixels_flat(g):
    a = np.asarray(g.convert("RGB"), np.float32)
    edge = np.concatenate(
        [
            a[:8].reshape(-1, 3),
            a[-8:].reshape(-1, 3),
            a[:, :8].reshape(-1, 3),
            a[:, -8:].reshape(-1, 3),
        ]
    )
    bg = np.median(edge, 0)
    fg = np.abs(a - bg).sum(2) > 40
    fg = (
        np.asarray(Image.fromarray((fg * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(9)))
        > 127
    )
    return fg


def color_transfer(vton, mask_img, garment):
    import cv2

    v = cv2.cvtColor(np.asarray(vton.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
    g = cv2.cvtColor(np.asarray(garment.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
    vm = np.asarray(mask_img.filter(ImageFilter.MinFilter(9))) > 127
    gm = garment_pixels_flat(garment)
    out = v.copy()
    for c in (1, 2):  # chroma: mean + std
        vs, gs = v[..., c][vm], g[..., c][gm]
        out[..., c] = (v[..., c] - vs.mean()) * (gs.std() / max(vs.std(), 1e-3)) + gs.mean()
    out[..., 0] = (
        v[..., 0] + (g[..., 0][gm].mean() - v[..., 0][vm].mean()) * 0.5
    )  # half the lightness gap
    rgb = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    alpha = np.asarray(mask_img.filter(ImageFilter.GaussianBlur(4)), np.float32)[..., None] / 255
    return Image.fromarray(
        (np.asarray(vton.convert("RGB"), np.float32) * (1 - alpha) + rgb * alpha).astype(np.uint8)
    )


def fabric_tiles(garment, size=512, positions=((0.32, 0.62), (0.68, 0.62)), side_frac=0.22):
    """Square cloth crops from the flat's body panels (below the chest, left/right of the placket: avoids
    buttons, chest print, pockets). Returned SEPARATELY: IP-Adapter centre-crops non-square input to a square
    (comfy clip_preprocess), so a side-by-side collage lost the outer half of each tile."""
    g = garment.convert("RGB")
    fg = garment_pixels_flat(g)
    ys, xs = np.where(fg)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0, y1 - y0
    side = int(side_frac * w)
    out = []
    for fx, fy in positions:
        cx, cy = int(x0 + fx * w), int(y0 + fy * h)
        box = (cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)
        out.append((g.crop(box).resize((size, size), Image.LANCZOS), box))
    return out


def load_tiles(specs, size=512):
    """specs: list of (image_path, (x0, y0, x1, y1)) square crops, e.g. from close worn shots."""
    out = []
    for path, box in specs:
        if box[2] - box[0] != box[3] - box[1]:
            raise SystemExit(f"reference crop must be square: {path} {box}")
        out.append(
            (
                Image.open(path).convert("RGB").crop(box).resize((size, size), Image.LANCZOS),
                tuple(box),
            )
        )
    return out


def encoder_view(tile, size=224):
    """What CLIP ViT-H actually receives (comfy clip_preprocess, crop=True): short side -> 224 bicubic, centre crop."""
    w, h = tile.size
    s = size / min(w, h)
    t = tile.resize((round(w * s), round(h * s)), Image.BICUBIC)
    l, u = (t.width - size) // 2, (t.height - size) // 2
    return t.crop((l, u, l + size, u + size))


def file_hash(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------- comfy
def post(path, data):
    r = urllib.request.Request(
        "http://127.0.0.1:8188" + path,
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(r, timeout=30))


def get(path):
    return json.load(urllib.request.urlopen("http://127.0.0.1:8188" + path, timeout=30))


def comfy_refine(
    img_name,
    mask_name,
    prompt,
    denoise,
    cn,
    seed,
    prefix,
    ref_names=(),
    ipa=0.0,
    ipa_type="style transfer",
):
    g = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "RealVisXL_V5.0_fp16.safetensors"},
        },
        "2": {"class_type": "DifferentialDiffusion", "inputs": {"model": ["1", 0]}},
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["1", 1], "text": prompt + POS_TAIL},
        },
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": NEG}},
        "5": {"class_type": "LoadImage", "inputs": {"image": img_name}},
        "6": {"class_type": "LoadImage", "inputs": {"image": mask_name}},
        "7": {"class_type": "ImageToMask", "inputs": {"image": ["6", 0], "channel": "red"}},
        "8": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["1", 2]}},
        "9": {
            "class_type": "SetLatentNoiseMask",
            "inputs": {"samples": ["8", 0], "mask": ["7", 0]},
        },
        "10": {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": "diffusion_pytorch_model_promax.safetensors"},
        },
        "11": {
            "class_type": "SetUnionControlNetType",
            "inputs": {"control_net": ["10", 0], "type": "tile"},
        },
        "12": {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["3", 0],
                "negative": ["4", 0],
                "control_net": ["11", 0],
                "image": ["5", 0],
                "strength": cn,
                "start_percent": 0.0,
                "end_percent": 0.9,
                "vae": ["1", 2],
            },
        },
        "13": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "positive": ["12", 0],
                "negative": ["12", 1],
                "latent_image": ["9", 0],
                "seed": seed,
                "steps": 30,
                "cfg": 5.0,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": denoise,
            },
        },
        "14": {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["13", 0],
                "vae": ["1", 2],
                "tile_size": 1024,
                "overlap": 64,
                "temporal_size": 64,
                "temporal_overlap": 8,
            },
        },
        "15": {
            "class_type": "SaveImage",
            "inputs": {"images": ["14", 0], "filename_prefix": prefix},
        },
    }
    if ref_names and ipa > 0:
        # each square reference tile is its own batch item -> encoded independently, embeds concatenated
        g["20"] = {
            "class_type": "IPAdapterUnifiedLoader",
            "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"},
        }
        img_ref = None
        for i, rn in enumerate(ref_names):
            nid = str(30 + i)
            g[nid] = {"class_type": "LoadImage", "inputs": {"image": rn}}
            if img_ref is None:
                img_ref = [nid, 0]
            else:
                bid = str(50 + i)
                g[bid] = {
                    "class_type": "ImageBatch",
                    "inputs": {"image1": img_ref, "image2": [nid, 0]},
                }
                img_ref = [bid, 0]
        g["22"] = {
            "class_type": "IPAdapterAdvanced",
            "inputs": {
                "model": ["20", 0],
                "ipadapter": ["20", 1],
                "image": img_ref,
                "weight": ipa,
                "weight_type": ipa_type,
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 1.0,
                "embeds_scaling": "V only",
            },
        }
        g["2"]["inputs"]["model"] = ["22", 0]
    q = get("/queue")
    if q["queue_running"] or q["queue_pending"]:
        raise SystemExit(f"queue busy: {q}")
    pid = post("/prompt", {"prompt": g, "client_id": str(uuid.uuid4())})["prompt_id"]
    t0 = time.time()
    while True:
        h = get(f"/history/{pid}")
        if pid in h:
            st = h[pid]["status"]
            if st.get("status_str") != "success":
                raise SystemExit(f"comfy failed: {json.dumps(st)[:800]}")
            o = h[pid]["outputs"]["15"]["images"][0]
            return (
                Image.open(COMFY / "output" / o["subfolder"] / o["filename"]).convert("RGB"),
                time.time() - t0,
                g,
                pid,
            )
        if time.time() - t0 > 900:
            raise SystemExit("timeout")
        time.sleep(1)


# ---------------------------------------------------------------- main
IPA_COMMIT = "a0f451a5113cf9becb0847b92884cb10cbdec0ef"


def run(
    case,
    c,
    denoise=0.5,
    cn=0.4,
    scale=2.0,
    seed=42,
    tag=None,
    ipa=0.0,
    ipa_type="style transfer",
    ref="flat",
    ref_mode="batch",
    pre_color=True,
    post_color=False,
    protect=True,
    force_refine=False,
):
    """c: dict(vton, garment, prompt, [sam_prompt], [garment_full], [ref_specs]).
    ref: 'flat' = square crops of the product flat, 'specs' = c['ref_specs'] [(path, square box), ...].
    ref_mode: 'batch' = every tile as its own IP-Adapter image, 'single' = first tile only.
    pre_color / post_color apply the SAME colour transfer regardless of ipa, so conditions stay comparable.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "runs").mkdir(exist_ok=True)
    tag = tag or f"d{denoise:g}_cn{cn:g}"
    vton = Image.open(c["vton"]).convert("RGB")
    vh = file_hash(c["vton"])
    sam_prompt = c.get("sam_prompt", "shirt")
    mask = sam_garment_mask(
        c["vton"], sam_prompt, OUT / f"{case}_sam_{vh}_{sam_prompt}.png"
    )  # cache keyed by input hash
    colored = color_transfer(vton, mask, Image.open(c["garment"])) if pre_color else vton
    colored.save(OUT / f"{case}_color.png")

    prot, prot_info = None, None
    if protect:  # product design (parts + structure edges) is never re-generated
        import design_protect as DP

        pp = OUT / "protect" / f"{case}_{vh}"
        if Path(str(pp) + ".json").exists():
            prot_info = json.loads(Path(str(pp) + ".json").read_text())
            prot = Image.open(str(pp) + "_protect.png").convert("L")
        else:
            pa, prot_info = DP.build(c["vton"], mask, pp)
            prot = Image.fromarray((pa * 255).astype(np.uint8))
        if prot_info["policy"] == "skip_refine" and not force_refine:
            out = color_transfer(colored, mask, Image.open(c["garment"])) if post_color else colored
            out.save(OUT / f"{case}_refined_{tag}.png")
            info = {
                "case": case,
                "tag": tag,
                "skipped": "structure garment",
                "protected_share": prot_info["protected_share"],
                "out": str(OUT / f"{case}_refined_{tag}.png"),
            }
            (OUT / "runs").mkdir(exist_ok=True)
            (OUT / "runs" / f"{case}_{tag}.json").write_text(
                json.dumps(
                    dict(
                        info,
                        vton=str(c["vton"]),
                        vton_sha=vh,
                        protect=prot_info,
                        pre_color=pre_color,
                        post_color=post_color,
                    ),
                    indent=1,
                ),
                encoding="utf-8",
            )
            print(json.dumps(info))
            return info

    ys, xs = np.where(np.asarray(mask) > 127)
    pad = 24
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x1, y1 = min(vton.width, xs.max() + pad), min(vton.height, ys.max() + pad)
    crop = colored.crop((x0, y0, x1, y1))
    mcrop = mask.crop((x0, y0, x1, y1))
    tw, th = (int(round(crop.width * scale / 64)) * 64, int(round(crop.height * scale / 64)) * 64)
    up = crop.resize((tw, th), Image.LANCZOS)
    soft = (
        mcrop.filter(ImageFilter.MinFilter(5))
        .resize((tw, th), Image.BILINEAR)
        .filter(ImageFilter.GaussianBlur(12 * scale / 2))
    )
    # keep the change INSIDE the garment: the blur above spreads past the mask edge onto skin/background
    inside = (
        mcrop.filter(ImageFilter.MaxFilter(3))
        .resize((tw, th), Image.BILINEAR)
        .filter(ImageFilter.GaussianBlur(1.5))
    )
    soft = Image.fromarray(
        (np.asarray(soft, np.float32) * np.asarray(inside, np.float32) / 255).astype(np.uint8)
    )
    if prot is not None:
        pc = (
            prot.crop((x0, y0, x1, y1))
            .resize((tw, th), Image.BILINEAR)
            .filter(ImageFilter.GaussianBlur(1.0 * scale))
        )
        keep = 1 - np.clip(
            np.asarray(pc, np.float32) / 255 * 1.5, 0, 1
        )  # 1.5x: blur tail must not leak into the design
        soft = Image.fromarray((np.asarray(soft, np.float32) * keep).astype(np.uint8))
    name = f"refine_{case}_{uuid.uuid4().hex[:6]}"
    up.save(COMFY / "input" / f"{name}.png")
    soft.convert("RGB").save(COMFY / "input" / f"{name}_mask.png")

    ref_names, ref_record = [], []
    if ipa > 0:
        if ref == "specs":
            tiles = load_tiles(c["ref_specs"])
            srcs = [p for p, _ in c["ref_specs"]]
        else:
            gpath = c.get("garment_full", c["garment"])
            tiles = fabric_tiles(Image.open(gpath))
            srcs = [gpath] * len(tiles)
        if ref_mode == "single":
            tiles, srcs = tiles[:1], srcs[:1]
        views = []
        for i, ((t, box), src) in enumerate(zip(tiles, srcs)):
            rn = f"{name}_ref{i}.png"
            t.save(COMFY / "input" / rn)
            ref_names.append(rn)
            v = encoder_view(t)
            views.append(v)
            ref_record.append(
                {
                    "source": str(src),
                    "source_sha": file_hash(src),
                    "box": [int(b) for b in box],
                    "tile": list(t.size),
                    "encoder_input": list(v.size),
                }
            )
        sheet = Image.new("RGB", (224 * len(views), 224))
        for i, v in enumerate(views):
            sheet.paste(v, (224 * i, 0))
        sheet.save(OUT / "runs" / f"{case}_{tag}_encoder_input.png")
    ref_img, secs, graph, pid = comfy_refine(
        f"{name}.png",
        f"{name}_mask.png",
        c["prompt"],
        denoise,
        cn,
        seed,
        f"refine/{case}_{tag}",
        ref_names=ref_names,
        ipa=ipa,
        ipa_type=ipa_type,
    )

    alpha = np.asarray(soft, np.float32)[..., None] / 255
    merged = (
        np.asarray(up, np.float32) * (1 - alpha)
        + np.asarray(ref_img.resize(up.size), np.float32) * alpha
    ).astype(np.uint8)
    big = colored.resize(
        (int(colored.width * tw / crop.width), int(colored.height * th / crop.height)),
        Image.LANCZOS,
    )
    big.paste(Image.fromarray(merged), (int(x0 * tw / crop.width), int(y0 * th / crop.height)))
    prot_check = None
    if prot is not None:  # verify: protected pixels must equal the un-refined image
        ref_big = np.asarray(colored.resize(big.size, Image.LANCZOS), np.float32)
        pm = np.asarray(prot.resize(big.size, Image.NEAREST)) > 127
        dlt = np.abs(np.asarray(big, np.float32) - ref_big).mean(2)[pm]
        prot_check = {
            "protected_px": int(pm.sum()),
            "mean_abs_change": round(float(dlt.mean()), 3) if pm.any() else 0.0,
            "p99_abs_change": round(float(np.percentile(dlt, 99)), 3) if pm.any() else 0.0,
        }
    if post_color:
        big = color_transfer(big, mask.resize(big.size, Image.NEAREST), Image.open(c["garment"]))
    big.save(OUT / f"{case}_refined_{tag}.png")
    Image.fromarray(merged).save(OUT / f"{case}_refined_{tag}_crop.png")
    up.save(OUT / f"{case}_color_crop_up.png")
    info = {
        "case": case,
        "tag": tag,
        "crop": [int(x0), int(y0), int(x1), int(y1)],
        "refine_size": [tw, th],
        "seconds": round(secs, 1),
        "protect_check": prot_check,
        "out": str(OUT / f"{case}_refined_{tag}.png"),
    }
    record = dict(
        info,
        prompt_id=pid,
        vton=str(c["vton"]),
        vton_sha=vh,
        garment=str(c["garment"]),
        garment_sha=file_hash(c["garment"]),
        sam_prompt=sam_prompt,
        prompt=c["prompt"],
        neg=NEG,
        pos_tail=POS_TAIL,
        seed=seed,
        denoise=denoise,
        cn=cn,
        scale=scale,
        ipa=ipa,
        ipa_type=ipa_type,
        ref=ref if ipa > 0 else None,
        ref_mode=ref_mode if ipa > 0 else None,
        refs=ref_record,
        pre_color=pre_color,
        post_color=post_color,
        protect=prot_info,
        protect_check=prot_check,
        ipadapter_commit=IPA_COMMIT,
        models=[
            "RealVisXL_V5.0_fp16.safetensors",
            "diffusion_pytorch_model_promax.safetensors (union, tile)",
            "ip-adapter-plus_sdxl_vit-h.safetensors",
            "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
        ],
        graph=graph,
    )
    (OUT / "runs" / f"{case}_{tag}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps(info))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=list(CASES))
    ap.add_argument("--denoise", type=float, default=0.4)
    ap.add_argument("--cn", type=float, default=0.6)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--ipa", type=float, default=0.0)
    ap.add_argument("--ipa-type", default="style transfer")
    ap.add_argument("--ref-mode", default="batch", choices=["batch", "single"])
    ap.add_argument("--post-color", action="store_true")
    a = ap.parse_args()
    run(
        a.case,
        CASES[a.case],
        a.denoise,
        a.cn,
        a.scale,
        a.seed,
        a.tag,
        a.ipa,
        a.ipa_type,
        ref_mode=a.ref_mode,
        post_color=a.post_color,
    )


if __name__ == "__main__":
    main()
