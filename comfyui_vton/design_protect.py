"""Design-protection mask for the texture refine: which garment pixels are PRODUCT DESIGN (must keep position
and shape) vs plain FABRIC SURFACE (may be re-textured).

  detected : GroundingDINO + SAM on the VTON output for small hard parts (button, snap, zipper, logo, print)
  structure: mid-scale, high-contrast luminance edges inside the garment (difference of Gaussians, sigma 1.5 vs 4
             at the 768x1024 frame): seams, pocket borders, plackets, cable columns, check/stripe edges.
             Fine low-amplitude texture (denim grain, fuzz) stays below the threshold and remains refinable.
  protect  = (detected | structure) dilated, inside the garment.

The share of the garment that is protected decides the policy: above SKIP_AT the garment is treated as a
structure garment (cable / check / stripe) and the refine is skipped entirely (a valid, expected outcome).

  python design_protect.py <vton.png> <garment_mask.png> <out_prefix>    -> <out_prefix>_protect.png, _overlay.jpg, .json
"""
import json, subprocess, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
PY = str(ROOT / 'work/runtime/Scripts/python.exe'); RUNNER = str(ROOT / 'ComfyUI-IDM-VTON-12GB/runner.py')
COMFY = r'C:\Dev\StabilityMatrix\Data\Packages\ComfyUI'
# two separate detections: 'logo / graphic print' often boxes the whole garment, and in one union mask that
# giant component swallowed the snap buttons (dropped together). Separate calls keep the small parts.
PARTS = ('button . snap button . zipper', 'logo . graphic print')
DOG_T = 7.0        # |DoG| in Lab L units at the 768x1024 frame
SKIP_AT = 0.35     # protected share of the garment above which the refine is skipped
MAX_PART = (0.01, 0.06)   # per PARTS group: a button/snap/zipper blob > 1% of the garment is SAM grabbing the whole
                          # placket (alpaca: the whole strip stayed smooth among fuzzy fabric); logos/prints may be larger
POLICY_DILATE = 5         # footprint used only for the skip decision
DILATE = 3                # protection margin (px at the 768 frame); 5 left visibly smooth bands around seams

def detect_parts(vton_path, out_path, prompt, threshold=0.25):
    job = {'comfy': COMFY, 'human': str(vton_path), 'prompt': prompt, 'threshold': threshold, 'grow': 3,
           'mask': str(out_path), 'pose': str(Path(out_path).with_name(Path(out_path).stem + '_pose.png'))}
    jp = Path(out_path).with_suffix('.json'); jp.write_text(json.dumps(job))
    r = subprocess.run([PY, RUNNER, 'preprocess', str(jp)], capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode:
        if 'No clothing detected' in r.stderr: return None          # nothing found is a valid answer
        raise SystemExit(r.stderr[-1500:])
    return np.asarray(Image.open(out_path).convert('L')) > 127

def structure_edges(img_rgb, garment):
    # Lab, all three channels: luminance-only edges missed grey buttons on a red cardigan (same L, different hue)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    s = img_rgb.shape[1] / 768.0                                        # scale-normalise to the VTON frame
    dog = cv2.GaussianBlur(lab, (0, 0), 1.5 * s) - cv2.GaussianBlur(lab, (0, 0), 4.0 * s)
    dog = np.sqrt((dog ** 2).sum(2))                                    # colour-difference magnitude (OpenCV 8-bit Lab units)
    inner = cv2.erode(garment.astype(np.uint8), np.ones((int(7 * s) | 1,) * 2, np.uint8)).astype(bool)   # ignore the garment outline itself
    e = (dog > DOG_T) & inner
    e = cv2.morphologyEx(e.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))   # drop isolated speckle
    return e.astype(bool), dog

def small_blobs(img_rgb, garment, de_t=18.0):
    """Buttons / snaps: small compact blobs whose colour differs from the LOCAL fabric (median of the neighbourhood).
    GroundingDINO missed the denim snaps and boxed the whole alpaca placket; DoG only caught button rims."""
    s = img_rgb.shape[1] / 768.0
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    k = int(25 * s) | 1
    med = np.dstack([cv2.medianBlur(lab[..., c], min(k, 255)) for c in range(3)]).astype(np.float32)
    de = np.sqrt(((lab.astype(np.float32) - med) ** 2).sum(2))
    inner = cv2.erode(garment.astype(np.uint8), np.ones((int(5 * s) | 1,) * 2, np.uint8)).astype(bool)
    cand = ((de > de_t) & inner).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab_, st, _ = cv2.connectedComponentsWithStats(cand)
    out = np.zeros_like(garment)
    lo, hi = 15 * s * s, 500 * s * s                                     # ~4-25 px diameter at the 768 frame
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if lo <= a <= hi and max(w, h) / max(1, min(w, h)) < 2.2 and a / (w * h) > 0.45:
            out[lab_ == i] = True
    return out

def build(vton_path, garment_mask, out_prefix):
    out_prefix = Path(out_prefix); out_prefix.parent.mkdir(parents=True, exist_ok=True)
    img = np.asarray(Image.open(vton_path).convert('RGB')); H, W = img.shape[:2]
    garment = np.asarray(garment_mask.convert('L').resize((W, H), Image.NEAREST)) > 127
    # SAM leaves HOLES where buttons sit (it treats them as separate objects), so every button fell outside
    # the garment and could not be protected. Fill interior holes before anything else.
    ff = np.pad((~garment).astype(np.uint8), 1, constant_values=1); cv2.floodFill(ff, None, (0, 0), 0)
    garment = garment | ff[1:-1, 1:-1].astype(bool)
    near =cv2.dilate(garment.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    det = np.zeros_like(garment)
    for gi, prompt in enumerate(PARTS):
        d = detect_parts(vton_path, str(out_prefix) + f'_parts{gi}.png', prompt=prompt)
        if d is None: continue
        d &= near
        # parts are SMALL: drop any component larger than MAX_PART of the garment (large prints are covered by the structure edges)
        n, lab, st, _ = cv2.connectedComponentsWithStats(d.astype(np.uint8))
        for i in range(1, n):
            if st[i][4] > MAX_PART[gi] * garment.sum(): d[lab == i] = False
        det |= d
    blobs = small_blobs(img, garment)
    det |= blobs
    struct, _ = structure_edges(img, garment)
    k = int(DILATE * W / 768) | 1
    protect = cv2.dilate((det | struct).astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool) & garment
    # skip decision uses a FIXED 5px footprint so it does not move when the protection margin is tuned
    # (shrinking DILATE 5->3 dropped the knit below SKIP_AT and it got refined again)
    kp = int(POLICY_DILATE * W / 768) | 1
    share = float((cv2.dilate((det | struct).astype(np.uint8), np.ones((kp, kp), np.uint8)).astype(bool) & garment).sum() / max(1, garment.sum()))
    info = {'garment_px': int(garment.sum()), 'detected_px': int(det.sum()), 'blob_px': int(blobs.sum()), 'structure_px': int(struct.sum()),
            'protected_share': round(share, 3), 'protect_px_share': round(float(protect.sum() / max(1, garment.sum())), 3), 'policy': 'skip_refine' if share > SKIP_AT else 'refine_unprotected',
            'params': {'parts': PARTS, 'max_part': MAX_PART, 'dilate': DILATE, 'dog': 'Lab 3-channel magnitude', 'dog_sigmas': [1.5, 4.0], 'dog_t': DOG_T, 'skip_at': SKIP_AT}}
    Image.fromarray((protect * 255).astype(np.uint8)).save(str(out_prefix) + '_protect.png')
    ov = img.copy().astype(np.float32)
    ov[garment & ~protect] = ov[garment & ~protect] * 0.55 + np.array([0, 170, 255]) * 0.45     # blue = refinable fabric
    ov[protect] = ov[protect] * 0.35 + np.array([255, 60, 0]) * 0.65                            # orange = protected design
    ov[det] = ov[det] * 0.3 + np.array([255, 0, 200]) * 0.7                                     # magenta = detected parts
    Image.fromarray(ov.clip(0, 255).astype(np.uint8)).save(str(out_prefix) + '_overlay.jpg', quality=90)
    Path(str(out_prefix) + '.json').write_text(json.dumps(info, indent=1))
    return protect, info

if __name__ == '__main__':
    p, i = build(sys.argv[1], Image.open(sys.argv[2]), sys.argv[3]); print(json.dumps(i))
