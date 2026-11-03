"""CPU regression checks for the CLO3D preprocessing (no GPU, no ComfyUI).
Run:  work\\runtime\\Scripts\\python.exe work\\test_clo3d_prep.py
Covers the findings in outputs/CLO3D_REVIEW_V2.md:
  1. float TIFF / 16-bit PNG depth keeps its values (no 8-bit truncation before normalisation)
  2. synthetic maps: background = 0, neckline hole = 0, seam absent outside the garment,
     foreground statistics exclude the background
  3. alpha fallback + explicit mask + size-mismatch rejection
"""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'work'))
import prepare_clo3d_maps as P
import build_clo3d_workflow as B

fails = []
def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond: fails.append(msg)

tmp = Path(tempfile.mkdtemp(prefix='clo3d_test_'))

# ---- 1. float depth is preserved
h, w = 96, 128
dist = np.full((h, w), 2000.0, np.float32)           # background: far
dist[20:80, 30:100] = 1500.0; dist[40:60, 50:80] = 1000.0   # garment nearer, centre nearest
Image.fromarray(dist, 'F').save(tmp / 'd.tiff')
d, alpha = P.read_depth(tmp / 'd.tiff')
check(d.dtype == np.float64 and sorted(set(np.unique(d).tolist())) == [1000.0, 1500.0, 2000.0], 'float TIFF depth read as [1000,1500,2000]')
Image.fromarray((dist / 2000 * 65535).astype(np.uint16)).save(tmp / 'd16.png')
d16, _ = P.read_depth(tmp / 'd16.png')
check(abs(d16.max() - 65535) < 1 and len(np.unique(d16)) == 3, '16-bit PNG depth keeps 3 distinct levels')
fg = dist < 1999
normal = np.zeros((h, w, 3), np.uint8); normal[..., 2] = 255; normal[..., :2] = 128
out, meta = P.prepare(d, normal, fg, invert_depth=True, max_side=256)
dep = np.asarray(out['depth']); box = meta['paste_box_xywh']
inner = dep[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
check(inner[int(50 / h * box[3]), int(65 / w * box[2])] > inner[int(30 / h * box[3]), int(40 / w * box[2])] > 0, 'inverted distance: nearest = brightest > garment > 0')
check(dep[0, 0] == 0 and dep[-1, -1] == 0, 'prepared depth background/padding is 0')
check(meta['output_size'][0] % 64 == 0 and meta['output_size'][1] % 64 == 0, 'output dims are multiples of 64')

# NaN handling
dn = d.copy(); dn[0, 0] = np.nan
out2, _ = P.prepare(dn, normal, fg | np.isnan(dn), invert_depth=True, max_side=256)
check(np.asarray(out2['depth'])[0, 0] == 0, 'NaN depth pixel becomes background')

# ---- 2. synthetic maps
B.make_test_maps(); L = B.make_test_maps.last
dep_s = np.asarray(Image.open(root / 'outputs' / B.DEPTH_IMG)); nor_s = np.asarray(Image.open(root / 'outputs' / B.NORMAL_IMG))
corners = [dep_s[0, 0], dep_s[0, -1], dep_s[-1, 0], dep_s[-1, -1]]
check(all(c == 0 for c in corners), f'synthetic depth corners are 0 (got {corners})')
check(all(tuple(nor_s[i, j]) == (128, 128, 255) for i, j in [(0, 0), (0, -1), (-1, 0), (-1, -1)]), 'synthetic normal corners are (128,128,255)')
ncx, ncy = int(B.W * 0.5), int(B.H * 0.235)
check(dep_s[ncy, ncx] == 0 and tuple(nor_s[ncy, ncx]) == (128, 128, 255), 'neckline hole is background in both maps')
below_hem = dep_s[int(B.H * 0.96):, ncx - 3:ncx + 4]
check(below_hem.max() == 0, 'centre seam does not extend below the hem')
check(L['z'][~L['body']].max() == 0.0 and L['depth'][~L['body']].max() == 0.0, 'z and depth are exactly 0 outside the support mask')
check(L['depth'][L['body']].min() >= 0.15 - 1e-9, 'foreground depth >= 0.15 (statistics exclude background)')
frac = L['body'].mean()
check(0.25 < frac < 0.75, f'support mask covers a plausible fraction of the frame ({frac:.2f})')

# ---- 3. CLI: alpha fallback, explicit mask, size mismatch, no-mask error
Image.fromarray(dist.astype(np.float32)).save(tmp / 'd.tif')
rgba = np.zeros((h, w, 4), np.uint8); rgba[..., :3] = 128; rgba[..., 2] = 255; rgba[..., 3] = (fg * 255).astype(np.uint8)
Image.fromarray(rgba, 'RGBA').save(tmp / 'n_alpha.png'); Image.fromarray(rgba[..., :3]).save(tmp / 'n.png')
Image.fromarray((fg * 255).astype(np.uint8)).save(tmp / 'm.png')
Image.fromarray(rgba[:h - 8, :, :3]).save(tmp / 'rgb_wrong.png')
py = sys.executable; cli = str(root / 'prepare_clo3d_maps.py')
def run(*args):
    r = subprocess.run([py, cli, *args, '--no-install', '--prefix', 'unittest'], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr
rc, o = run('--depth', str(tmp / 'd.tif'), '--normal', str(tmp / 'n_alpha.png'), '--invert-depth')
check(rc == 0 and '"foreground_source": "alpha:normal"' in o, 'alpha channel of normal used as foreground')
rc, o = run('--depth', str(tmp / 'd.tif'), '--normal', str(tmp / 'n.png'), '--mask', str(tmp / 'm.png'), '--invert-depth')
check(rc == 0 and '"foreground_source": "mask"' in o, 'explicit mask takes precedence')
rc, o = run('--depth', str(tmp / 'd.tif'), '--normal', str(tmp / 'n.png'), '--invert-depth')
check(rc != 0 and 'no --mask and no alpha' in o, 'no mask/alpha -> error (heuristic is opt-in)')
rc, o = run('--depth', str(tmp / 'd.tif'), '--normal', str(tmp / 'n.png'), '--invert-depth', '--bg-from-corner')
check(rc == 0 and 'corner-heuristic' in o, '--bg-from-corner opt-in works')
rc, o = run('--depth', str(tmp / 'd.tif'), '--normal', str(tmp / 'n.png'), '--mask', str(tmp / 'm.png'), '--rgb', str(tmp / 'rgb_wrong.png'))
check(rc != 0 and '!= depth' in o, 'RGB with different size is rejected, not resized')
for f in (root / 'outputs/prepared').glob('unittest_*'): f.unlink()

print('\n%d failure(s)' % len(fails)); sys.exit(1 if fails else 0)
