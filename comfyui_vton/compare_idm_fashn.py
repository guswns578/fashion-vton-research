"""IDM-VTON vs FASHN (enhance.py recipe) on the same catalog cases, compared in the SAME frame
(the 768x1024 person frame the IDM run used). FASHN outputs are in the original photo's frame, so the
street case is re-cropped with the same box catalog_vton_prep.street_person() used.

Metric: high-pass RMS (L channel minus Gaussian sigma 1.4, i.e. the band VTON destroys per the
August H(f) measurement) inside a torso region both results cover (eroded IDM clothing mask).
It measures the AMOUNT of fine detail only, not whether it is the right fabric; read with the zoom sheet.
Outputs: outputs/idm_vs_fashn_contact.jpg, outputs/idm_vs_fashn_zoom.jpg, outputs/idm_vs_fashn_metrics.json
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

ROOT = Path(__file__).resolve().parent
W = ROOT / 'work'
exec((ROOT / 'catalog_vton_prep.py').read_text(encoding='utf-8').split('CASES = [')[0], ns := {'__file__': str(ROOT / 'catalog_vton_prep.py')})

def street_box(rel):
    meta = next(json.loads(l) for l in open(ns['DS'] / 'images.jsonl', encoding='utf-8') if json.loads(l)['image_id'] == rel)
    im = Image.open(ns['RAW'] / rel); x0, y0, x1, y1 = meta['person_box']
    ph = y1 - y0; top = max(0, y0 - 0.04 * ph); bot = min(im.height, y0 + 0.62 * ph)
    h = bot - top; w = h * 3 / 4; left = min(max(0, (x0 + x1) / 2 - w / 2), im.width - w)
    return (int(left), int(top), int(left + w), int(bot))

CASES = [  # name, idm case dir, fashn file, frame box in the fashn image (None = same frame), mask file
    ('denim shirt / studio', 'studio_p2589_denim_shirt', 'studio_p2589_denim_sf1.png', None, 'mask_shirt_only.png'),
    ('fisherman knit / studio', 'studio_p2610_fisherman_knit', 'studio_p2610_knit_sf1.png', None, 'mask_shirt_only.png'),
    ('denim shirt / street', 'street_p2610person_p2589_denim_shirt', 'street_p2610person_p2589_denim_sf0.png',
     street_box('p2610/detail/022.jpg'), 'mask.png'),
]

def hp_rms(img, region):
    L = np.asarray(img.convert('L'), np.float32)
    lo = np.asarray(img.convert('L').filter(ImageFilter.GaussianBlur(1.4)), np.float32)
    return float(np.sqrt(((L - lo)[region] ** 2).mean()))

rows, zooms, metrics = [], [], []
for label, idm_dir, fashn_file, box, mask_file in CASES:
    d = W / 'catalog_vton' / idm_dir
    garment = Image.open(d / 'garment.jpg').convert('RGB')
    idm = Image.open(ROOT / 'outputs' / f'catalog_{idm_dir}.png').convert('RGB')
    fa = Image.open(W / 'fashn' / fashn_file).convert('RGB')
    if box: fa = fa.crop(box)
    fa = fa.resize(idm.size, Image.LANCZOS)
    m = Image.open(d / mask_file).convert('L').resize(idm.size).filter(ImageFilter.MinFilter(31))
    region = np.asarray(m) > 127
    rec = {'case': label, 'region_px': int(region.sum()), 'idm_hp_rms': hp_rms(idm, region), 'fashn_hp_rms': hp_rms(fa, region)}
    rec['ratio_fashn_over_idm'] = rec['fashn_hp_rms'] / rec['idm_hp_rms']; metrics.append(rec)
    rows.append((label, garment, idm, fa))
    ys, xs = np.where(region); cx = int(xs.mean()); cy = int(ys.min() + 0.28 * (ys.max() - ys.min()))
    z = (cx - 100, cy - 100, cx + 100, cy + 100)                     # 200x200 native px, shown 2x nearest
    zooms.append((label, idm.crop(z).resize((400, 400), Image.NEAREST), fa.crop(z).resize((400, 400), Image.NEAREST)))
    print(json.dumps(rec, ensure_ascii=False))

S = (384, 512)
sheet = Image.new('RGB', (S[0] * 3, (S[1] + 24) * len(rows) + 24), 'white'); dr = ImageDraw.Draw(sheet)
for c, t in enumerate(('garment (catalog flat)', 'IDM-VTON 768x1024', 'FASHN 1.5 enhance (same frame)')): dr.text((c * S[0] + 6, 6), t, fill='black')
for r, (label, g, a, b) in enumerate(rows):
    y = 24 + r * (S[1] + 24); dr.text((6, y + 4), label, fill='black')
    for c, im in enumerate((g, a, b)): sheet.paste(im.resize(S), (c * S[0], y + 20))
sheet.save(ROOT / 'outputs' / 'idm_vs_fashn_contact.jpg', quality=92)

zs = Image.new('RGB', (820, 430 * len(zooms) + 24), 'white'); dr = ImageDraw.Draw(zs)
dr.text((6, 6), 'chest 200x200 px at 2x (nearest)   left: IDM-VTON   right: FASHN', fill='black')
for r, (label, a, b) in enumerate(zooms):
    y = 24 + r * 430; dr.text((6, y + 2), label, fill='black'); zs.paste(a, (0, y + 20)); zs.paste(b, (420, y + 20))
zs.save(ROOT / 'outputs' / 'idm_vs_fashn_zoom.jpg', quality=95)
(ROOT / 'outputs' / 'idm_vs_fashn_metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
