"""Regression checks for the Blender map-extraction path (needs Blender 4.5, no GPU/ComfyUI).
Run:  work\\runtime\\Scripts\\python.exe work\\test_blender_maps.py [--no-render]
Covers outputs/CLO3D_BLENDER_REVIEW.md:
  1. alpha cutouts (Principled Alpha texture) stay holes in normal alpha AND depth, not filled
  2. demo goes through OBJ export -> import_garment() (importer path exercised)
  3. conventions: sphere facing B high, left R low (OpenGL) -> after prepare --flip-normal-x left R high (BAE);
     depth near < far (uint16), background 65535 -> prepare --invert-depth: near bright, bg 0
  4. prepare: RGBA reference composited over white (corners white, not black)
  5. EXR: multi-channel without --exr-channel and --exr-channel Z are rejected
"""
import json, subprocess, sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'work'))
import prepare_clo3d_maps as P
BLENDER = r'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe'
demo = root / 'work' / 'blender_demo'
fails = []
def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond: fails.append(msg)

if '--no-render' not in sys.argv:
    r = subprocess.run([BLENDER, '-b', '-P', str(root / 'blender_render_maps.py'), '--', '--demo', '--out', str(demo),
                        '--prefix', 'demo', '--res', '832', '1216', '--rgb'], capture_output=True, text=True)
    check(r.returncode == 0 and 'DONE' in r.stdout, 'blender demo render exits 0')
    if r.returncode != 0: print(r.stdout[-3000:], r.stderr[-3000:]); sys.exit(1)
cam = json.loads((demo / 'demo_camera.json').read_text(encoding='utf-8'))
check((demo / 'demo_src.obj').exists(), 'demo was round-tripped through an OBJ file')

nrm = np.asarray(Image.open(demo / 'demo_normal.png'))            # RGBA uint8
dep = np.asarray(Image.open(demo / 'demo_depth.png')).astype(int)   # uint16
rgb = np.asarray(Image.open(demo / 'demo_rgb.png'))
H, W = dep.shape
check(nrm.shape == (H, W, 4) and rgb.shape[:2] == (H, W), 'normal/depth/rgb have one resolution')
check(dep[0, 0] == 65535 and nrm[0, 0, 3] == 0, 'background: depth 65535 (far), normal alpha 0')

# --- 1. alpha plane: 0.3 m plane at world (-0.3,-0.7,0.7), in front of the sphere; checker scale 6 -> holes
# project world -> pixel with the ortho camera: x_px = (x - cx)/ortho*W' ... use camera meta
lo, hi = np.array(cam['bbox_min']), np.array(cam['bbox_max']); centre = (lo + hi) / 2
ortho = cam['ortho_scale']; px_per_unit = max(W, H) / ortho
def to_px(x, z):
    return int(round(W / 2 + (x - centre[0]) * px_per_unit)), int(round(H / 2 - (z - centre[2]) * px_per_unit))
x0, y1 = to_px(-0.44, 0.56); x1, y0 = to_px(-0.16, 0.84)           # plane extents in pixels (inset 1 cm)
region_a = nrm[y0:y1, x0:x1, 3]; region_d = dep[y0:y1, x0:x1]; region_n = nrm[y0:y1, x0:x1, :3].astype(int)
# the plane is perpendicular to the ortho camera, so its depth is one constant = the nearest value in the region;
# pixels noticeably farther than that are holes (alpha stays 1 there because the sphere is behind the plane)
plane_depth = region_d.min(); solid = region_d < plane_depth + 300; holes = ~solid
check(0.25 < holes.mean() < 0.75, f'checker holes in the alpha plane: {holes.mean():.2f} of its area sees past the plane')
check(np.median(region_d[holes]) > plane_depth + 500, f'depth through holes is farther than the plane ({np.median(region_d[holes]):.0f} vs {plane_depth})')
sphere_seen_through = (region_d[holes] < 65535).mean()
check(sphere_seen_through > 0.5, f'sphere behind the cutout is visible through holes ({sphere_seen_through:.2f})')
check(region_a[holes].mean() > 200, 'coverage alpha stays 1 through holes (sphere behind), i.e. cutout != background')
# plane faces the camera (R~128); the sphere surface behind it on the left has R well below 128
check(abs(region_n[solid][:, 0].mean() - 128) < 8 and region_n[holes][:, 0].mean() < 100,
      f'normals through holes are the sphere surface (R={region_n[holes][:, 0].mean():.0f}), plane R={region_n[solid][:, 0].mean():.0f}')

# --- 3. conventions on the sphere (centre (0,0,0.9), r 0.5): sample interior points
sx, sy = to_px(0.0, 0.9); lx, _ = to_px(-0.42, 0.9); rx, _ = to_px(0.42, 0.9); _, ty = to_px(0.0, 1.32); _, by = to_px(0.0, 0.48)
def m(x, y, s=4): return nrm[y - s:y + s, x - s:x + s, :3].reshape(-1, 3).mean(0)
c, l, r_, t, b = m(sx, sy), m(lx, sy), m(rx, sy), m(sx, ty), m(sx, by)
check(c[2] > 240, f'sphere centre faces camera: B={c[2]:.0f}')
check(l[0] < 80 < 180 < r_[0], f'OpenGL x: left R={l[0]:.0f} < right R={r_[0]:.0f}')
check(t[1] > 180 > 80 > b[1], f'y up: top G={t[1]:.0f} > bottom G={b[1]:.0f}')
check(dep[sy, sx] < dep[sy, lx] < 65535, f'depth: centre {dep[sy, sx]} nearer than rim {dep[sy, lx]} < bg')

# prepare with the documented flags
fg = nrm[..., 3] > 127
out, meta = P.prepare(dep.astype(np.float64), nrm[..., :3], fg, invert_depth=True, flip_x=True, rgb=rgb, max_side=1216)
pn = np.asarray(out['normal']); pd = np.asarray(out['depth']); pr = np.asarray(out['rgb']); pm = np.asarray(out['mask'])
check(pn[sy, lx, 0] > 180 and pn[sy, rx, 0] < 80, f'after --flip-normal-x: BAE left R={pn[sy, lx, 0]} high, right R={pn[sy, rx, 0]} low')
check(pd[sy, sx] > pd[sy, lx] > 0 and pd[0, 0] == 0, f'after --invert-depth: centre {pd[sy, sx]} > rim {pd[sy, lx]} > bg 0')
check(tuple(pn[0, 0]) == (128, 128, 255), 'prepared normal background (128,128,255)')
check(pm[y0:y1, x0:x1].mean() > 250, 'prepared mask = coverage (sphere behind holes -> region fully covered)')
# --- 4. RGBA reference composited over white
check(tuple(pr[0, 0]) == (255, 255, 255) and tuple(pr[-1, -1]) == (255, 255, 255), f'prepared rgb corners are white ({pr[0, 0].tolist()})')

# --- 5. EXR channel policy (cv2 decode mocked: BGR array)
import cv2
fake = np.zeros((4, 4, 3), np.float32); fake[:] = (0.2, 0.4, 0.8)
with patch.object(cv2, 'imread', return_value=fake):
    for chan, expect_exit in ((None, True), ('Z', True), ('R', False)):
        try:
            d, _ = P.read_depth('fake.exr', chan); exited = False
        except SystemExit:
            exited = True
        check(exited == expect_exit, f'EXR --exr-channel {chan}: {"rejected" if exited else "accepted"}')
    d, _ = P.read_depth('fake.exr', 'R'); check(abs(d[0, 0] - 0.8) < 1e-6, 'EXR --exr-channel R returns the R value (0.8), not B')

print('\n%d failure(s)' % len(fails)); sys.exit(1 if fails else 0)
