"""Graphic transplant: diffusion garbles printed text/logos, so copy the print from the product flat instead.

  python graphic_transplant.py <case>          (case must exist in batch_tryon.CASES and have a refined output)

1. source (product flat, full resolution): ink = pixels darker than the fabric inside the garment,
   restricted to connected components whose centre is in the chest area (drops the contrast rib bands).
   Alpha = how much darker than the fabric, so thin strokes and the vintage crackle keep their softness.
2. target (final refined image): same detection finds the garbled print -> its bbox is where the new print
   goes; the garbled ink is erased with cv2.inpaint and the erased area gets fabric grain back (high-pass
   noise sampled from clean fabric next to the print).
3. composite: source print scaled to the target bbox (aspect kept, centred), multiplied by the target's
   low-frequency shading so folds and lighting fall on the print too.
Outputs outputs/graphic/<case>_graphic.png and <case>_graphic_sheet.jpg.
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'work'))
import batch_tryon as B

OUT = ROOT / 'outputs' / 'graphic'; R = ROOT / 'outputs' / 'refine'

def lum(a): return cv2.cvtColor(a, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)

def chest_ink(img, garment, dark=35, chest=((0.22, 0.78), (0.12, 0.72))):
    """Ink mask + bbox of the print: dark components centred in the chest part of the garment bbox."""
    L = lum(img); fab = float(np.median(L[garment]))
    ys, xs = np.where(garment); gx0, gx1, gy0, gy1 = xs.min(), xs.max(), ys.min(), ys.max(); gw, gh = gx1 - gx0, gy1 - gy0
    # LOCAL darkness: grey closing removes dark features thinner than the kernel (letter strokes) and keeps
    # broad ones (fold / side shadow), so shadows are not mistaken for ink.
    kk = max(15, int(0.08 * gw)) | 1
    local_bg = cv2.morphologyEx(L, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
    inner = cv2.erode(garment.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)   # off the garment edge / rib bands
    ink = (L < local_bg - dark) & inner
    k = max(3, int(0.012 * gw)) | 1                                   # join letters of one word
    n, lab, st, cen = cv2.connectedComponentsWithStats(cv2.dilate(ink.astype(np.uint8), np.ones((k, k), np.uint8)))
    cand = []
    for i in range(1, n):
        cx, cy = (cen[i][0] - gx0) / gw, (cen[i][1] - gy0) / gh
        x, y, w, h, area = st[i]; inkpx = int((ink & (lab == i)).sum())
        if not (chest[0][0] < cx < chest[0][1] and chest[1][0] < cy < chest[1][1] and area > 0.0005 * gw * gh): continue
        if max(w, h) / max(1, min(w, h)) > 4 and inkpx / (w * h) < 0.25: continue      # fold line / strap shadow
        cand.append((inkpx, i, (x, y, x + w, y + h)))
    if not cand: raise SystemExit('no print found in the chest area')
    # the print = biggest ink cluster + clusters touching its (expanded) bbox; drops stray dark bits elsewhere
    _, i0, (bx0, by0, bx1, by1) = max(cand); ex = 0.25 * max(bx1 - bx0, by1 - by0)
    keep = np.zeros_like(ink)
    for _, i, (x0, y0, x1, y1) in cand:
        if x1 > bx0 - ex and x0 < bx1 + ex and y1 > by0 - ex and y0 < by1 + ex: keep |= lab == i
    keep &= ink
    if not keep.any(): raise SystemExit('no print found in the chest area')
    # a fold line can touch the lettering and join its component; trim the bbox to rows/columns that carry
    # real ink mass (text rows are dense, a crease is a few px per row)
    def dense_span(counts):
        on = counts > 0.12 * counts.max(); idx = np.where(on)[0]
        runs = np.split(idx, np.where(np.diff(idx) > max(3, len(counts) // 40))[0] + 1)
        big = max(runs, key=lambda r: counts[r].sum()); return big[0], big[-1] + 1
    y0, y1 = dense_span(keep.sum(1)); x0, x1 = dense_span(keep[y0:y1].sum(0))
    box = np.zeros_like(keep); box[y0:y1, x0:x1] = True; keep &= box
    return keep, (x0, y0, x1, y1), fab

def flat_garment(a):
    edge = np.concatenate([a[:8].reshape(-1, 3), a[-8:].reshape(-1, 3), a[:, :8].reshape(-1, 3), a[:, -8:].reshape(-1, 3)]).astype(np.float32)
    fg = (np.abs(a.astype(np.float32) - np.median(edge, 0)).sum(2) > 40).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    return lab == 1 + np.argmax(st[1:, 4])                             # largest blob = garment (drops the caption)

def main(case):
    OUT.mkdir(parents=True, exist_ok=True); c = B.CASES[case]
    src = np.asarray(Image.open(B.RAW / c['garment']).convert('RGB'))
    s_mask, s_box, s_fab = chest_ink(src, flat_garment(src))
    fin_path = R / f"{case}_refined_d{c['d']:g}.png"
    tgt = np.asarray(Image.open(fin_path).convert('RGB')).copy(); H, W = tgt.shape[:2]
    t_gar = np.asarray(Image.open(R / f'{case}_sam_mask.png').convert('L').resize((W, H))) > 127
    t_ink, t_box, t_fab = chest_ink(tgt, t_gar)

    # ---- erase the garbled print, then give the patch fabric grain back
    # Erase = replace the old print's pixels with the LOCAL FABRIC BACKGROUND: a grey closing removes thin dark
    # strokes but keeps broad fold/light shading, so shading survives (cv2.inpaint over a wide area smeared
    # it into flat blotches). Region = old ink + its soft edges, but only near the old ink (not the whole box).
    x0, y0, x1, y1 = t_box; bw, bh = x1 - x0, y1 - y0
    Lab = cv2.cvtColor(tgt, cv2.COLOR_RGB2LAB).astype(np.float32); Lt = Lab[..., 0]
    kk = max(15, int(0.10 * bw)) | 1
    lbg = cv2.morphologyEx(Lt, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
    # old ink = dark strokes anywhere in the print box grown by 15% (t_ink is trimmed to the dense rows/cols,
    # which cut off a letter tail)
    e = int(0.15 * max(bw, bh)); bx = np.zeros_like(t_ink); bx[max(0, y0 - e):y1 + e, max(0, x0 - e):x1 + e] = True
    old_ink = (t_ink | (bx & (Lt < lbg - 20))) & t_gar
    near = cv2.dilate(old_ink.astype(np.uint8), np.ones((int(0.05 * bw) | 1,) * 2, np.uint8)).astype(bool)
    erase = (old_ink | (near & (Lt < lbg - 10))) & t_gar
    erase = cv2.dilate(erase.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & t_gar
    clean = t_gar & ~near
    # fill = normalized convolution: local mean of the surrounding VALID fabric (not erased, not ink), so
    # brightness and broad shading continue seamlessly into the hole (closing / global offset left halos)
    valid = (t_gar & ~erase & ~old_ink).astype(np.float32); sig = max(4.0, 0.035 * bw)
    num = cv2.GaussianBlur(Lt * valid, (0, 0), sig); den = cv2.GaussianBlur(valid, (0, 0), sig)
    for _ in range(3):                                             # widen for holes larger than the kernel
        hole = den < 0.05
        if not hole.any(): break
        sig *= 2; num2 = cv2.GaussianBlur(Lt * valid, (0, 0), sig); den2 = cv2.GaussianBlur(valid, (0, 0), sig)
        num[hole], den[hole] = num2[hole], den2[hole]
    local_mean = num / np.maximum(den, 1e-4)
    hp = Lt - cv2.GaussianBlur(Lt, (0, 0), 1.5)                   # fabric grain sample
    rng = np.random.default_rng(0); pool = hp[clean]
    newL = local_mean + pool[rng.integers(0, len(pool), Lt.shape)]
    ab = cv2.GaussianBlur(Lab[..., 1:], (0, 0), 6)                  # chroma: smooth local fabric colour
    fill = np.dstack([newL, ab]).clip(0, 255).astype(np.uint8)
    fill = cv2.cvtColor(fill, cv2.COLOR_LAB2RGB)
    fa = cv2.GaussianBlur(erase.astype(np.float32), (0, 0), 2.5)[..., None]
    base = (tgt.astype(np.float32) * (1 - fa) + fill.astype(np.float32) * fa).clip(0, 255).astype(np.uint8)

    # ---- source print as colour + alpha, scaled into the target bbox
    sx0, sy0, sx1, sy1 = s_box
    s_rgb = src[sy0:sy1, sx0:sx1].astype(np.float32); sL = lum(src[sy0:sy1, sx0:sx1])
    ink_L = np.percentile(sL[s_mask[sy0:sy1, sx0:sx1]], 5)
    s_alpha = np.clip((s_fab - sL) / max(s_fab - ink_L, 1), 0, 1)
    s_alpha[~cv2.dilate(s_mask[sy0:sy1, sx0:sx1].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)] = 0
    # ink colour: un-mix the fabric out of partially covered pixels (fabric colour = median around the print)
    fab_rgb = np.median(src[flat_garment(src) & ~s_mask].reshape(-1, 3), 0).astype(np.float32)
    a3 = s_alpha[..., None]; ink_rgb = np.where(a3 > 0.15, (s_rgb - fab_rgb * (1 - a3)) / np.maximum(a3, 0.15), 0)
    scale = min(bw / (sx1 - sx0), bh / (sy1 - sy0)) * 1.0
    nw, nh = int((sx1 - sx0) * scale), int((sy1 - sy0) * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    ink_rgb = cv2.resize(np.clip(ink_rgb, 0, 255), (nw, nh), interpolation=interp)
    alpha = cv2.resize(s_alpha, (nw, nh), interpolation=interp)
    if scale > 1.2:                         # the flat's print is small; unsharp the upscaled edges back to crisp
        sig = 0.6 * scale; alpha = np.clip(alpha + 1.0 * (alpha - cv2.GaussianBlur(alpha, (0, 0), sig)), 0, 1)
    px, py = x0 + (bw - nw) // 2, y0 + (bh - nh) // 2

    # ---- shading from the target (folds / light), relative to its mean over the print area
    Lb = lum(base); shade = cv2.GaussianBlur(Lb, (0, 0), max(3, bw / 60))
    reg = shade[py:py + nh, px:px + nw]; sh = (reg / max(float(np.median(reg)), 1))[..., None]
    out = base.astype(np.float32); roi = out[py:py + nh, px:px + nw]
    a = alpha[..., None] * t_gar[py:py + nh, px:px + nw, None]
    out[py:py + nh, px:px + nw] = roi * (1 - a) + np.clip(ink_rgb * sh, 0, 255) * a
    res = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)); res.save(OUT / f'{case}_graphic.png')

    # ---- sheet: flat print | before (refined) | after, full + zoom
    idm = Image.open(B.OB / f'{case}_idm.png').convert('RGB').resize((W, H))
    m = int(0.25 * bw); zb = (max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m))
    Z = 420; S = Image.new('RGB', (Z * 4, Z + 330), 'white'); d = ImageDraw.Draw(S)
    zs = [Image.fromarray(src[sy0:sy1, sx0:sx1]), idm.crop(zb), Image.fromarray(tgt).crop(zb), res.crop(zb)]
    for i, (t, im) in enumerate(zip(('flat print', 'IDM-VTON', 'refined (before)', 'graphic transplant'), zs)):
        d.text((i * Z + 4, 2), t, fill='black'); im.thumbnail((Z, Z)); S.paste(im, (i * Z, 16))
    for i, im in enumerate((Image.open(B.WB / case / 'human.jpg'), idm, Image.fromarray(tgt), res)):
        S.paste(im.convert('RGB').resize((240, 312)), (i * Z + 90, Z + 18))
    S.save(OUT / f'{case}_graphic_sheet.jpg', quality=93)
    print(f'src box {s_box} -> tgt box {t_box}, scale {scale:.2f}, saved {OUT / f"{case}_graphic.png"}')

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'ringer_tee__pockets')
