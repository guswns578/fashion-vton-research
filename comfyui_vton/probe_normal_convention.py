"""Empirical check of the NormalBae colour convention (xinsir's reference preprocessor for the
Union 'normal' type). Runs in the isolated ROOT runtime (has timm) with controlnet_aux source from
ComfyUI; weights (scannet.pt) download once into ROOT/work/models/annotators.

Caveat: BAE is a monocular *estimate*, not a measurement of the true geometry. Rim values of the
synthetic sphere are distorted (the flat background is read as a floor), so the sign decision uses
interior sphere samples plus the shoulder rows of a real photo. The real CLO3D renderer's axes
must still be checked independently. Statistics are saved to outputs/probe_bae_stats.json."""

import os, sys, json
from pathlib import Path

root = Path(__file__).resolve().parent
os.environ["AUX_ANNOTATOR_CKPTS_PATH"] = str(root / "work/models/annotators")
sys.path.insert(
    0, r"C:\Dev\StabilityMatrix\Data\Packages\ComfyUI\custom_nodes\comfyui_controlnet_aux\src"
)
import numpy as np, torch
from PIL import Image
from custom_controlnet_aux.normalbae import NormalBaeDetector

S = 768
yy, xx = np.mgrid[0:S, 0:S]
u = (xx - S / 2) / (S * 0.4)
v = (yy - S / 2) / (S * 0.4)
inside = u * u + v * v < 1
z = np.sqrt(np.clip(1 - u * u - v * v, 0, 1))
n = np.stack([u, -v, z], -1)
L = np.array([-0.4, 0.6, 0.7])
L /= np.linalg.norm(L)
shade = np.clip(n @ L, 0, 1) * 0.8 + 0.2
sphere = np.repeat((np.where(inside, shade, 0.55) * 255).astype(np.uint8)[..., None], 3, -1)
Image.fromarray(sphere).save(root / "outputs/probe_sphere.png")

det = NormalBaeDetector.from_pretrained().to("cuda")
with torch.inference_mode():
    out_s = det(sphere, detect_resolution=768, image_resolution=768, output_type="np")
    human = np.asarray(Image.open(root / "outputs/sample_human.jpg").convert("RGB"))
    out_h = det(human, detect_resolution=768, image_resolution=768, output_type="np")
Image.fromarray(out_s).save(root / "outputs/probe_bae_sphere.png")
Image.fromarray(out_h).save(root / "outputs/probe_bae_human.png")

stats = {
    "convention_note": "R high = surface faces image-left (-x); G high = faces up (+y); B high = faces camera (+z). "
    "Monocular estimate, not a geometric measurement.",
    "sphere": {},
    "human": {},
}
a = out_s.astype(int)
H, W = a.shape[:2]
cx, cy, r = W // 2, H // 2, int(min(W, H) * 0.4)


def m(x, y, s=12):
    return a[y - s : y + s, x - s : x + s].reshape(-1, 3).mean(0).round().tolist()


for k in (0.3, 0.5, 0.7, 0.85):
    d = int(r * k)
    stats["sphere"][f"interior_k={k}"] = {
        "left": m(cx - d, cy),
        "right": m(cx + d, cy),
        "top": m(cx, cy - d),
        "bottom": m(cx, cy + d),
        "roi_centre_px": {
            "left": [cx - d, cy],
            "right": [cx + d, cy],
            "top": [cx, cy - d],
            "bottom": [cx, cy + d],
        },
        "roi_half_size_px": 12,
    }
stats["sphere"]["background_corner"] = m(22, 22, 17)
# human: sample just inside the silhouette (from the original photo's near-white background) on shoulder/torso rows
o = np.asarray(
    Image.open(root / "outputs/sample_human.jpg")
    .convert("RGB")
    .resize((out_h.shape[1], out_h.shape[0]))
).astype(int)
b = out_h.astype(int)
fg = np.abs(o - o[5:30, 5:30].reshape(-1, 3).mean(0)).sum(2) > 40
for y in (300, 340, 380, 420):
    cols = np.where(fg[y])[0]
    l, r_ = int(cols.min()), int(cols.max())
    stats["human"][f"row_{y}"] = {
        "in_from_left": b[y, l + 8 : l + 20].mean(0).round().tolist(),
        "in_from_right": b[y, r_ - 20 : r_ - 8].mean(0).round().tolist(),
        "roi_cols": {"left": [l + 8, l + 20], "right": [r_ - 20, r_ - 8]},
    }
(root / "outputs/probe_bae_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
print(json.dumps(stats, indent=2))
