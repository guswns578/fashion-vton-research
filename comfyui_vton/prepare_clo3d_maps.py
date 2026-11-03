"""Prepare real CLO3D/Blender depth + normal (+ optional RGB / mask) exports for the
CLO3D-ControlNet-Union workflow. ONE geometric transform (uniform scale + centre pad) is applied
to every map so they stay pixel-aligned; the transform is saved as JSON so results can be mapped back.

Input formats
  depth : PNG 8/16-bit, TIFF 8/16/32-bit float (PIL, tifffile fallback), EXR via cv2 ONLY for single-channel
          files or an explicit RGB(A) channel (--exr-channel R|G|B|A). cv2 cannot address named channels
          such as "Z" in multi-pass EXRs, so those are rejected: export a single-channel EXR/TIFF/PNG16 instead.
          Float values are kept as float until normalisation. NaN/Inf are treated as background.
  normal: 8-bit RGB(A) PNG/TIFF. rgb / mask: same pixel size as depth (no silent resizing).

Foreground policy (first available wins)
  1. --mask FILE            explicit mask (>127 = garment)
  2. alpha channel          of --alpha-from {normal,rgb,depth} (default: auto = first file that has alpha)
  3. --bg-from-corner       opt-in heuristic: background = depth within --depth-bg-tol of the corner value
  otherwise the script stops with an error.

Output conventions (what the workflow expects)
  depth  : 8-bit, near = white, foreground normalised to [0.15, 1.0] by percentiles, background 0
  normal : BAE convention, R = -x (left-facing red), G = +y (up), B = +z (toward camera), bg (128,128,255).
           --flip-normal-x / --flip-normal-y if the renderer's axes differ (OpenGL/Blender camera-space
           normals usually need --flip-normal-x). Verify on the real render, do not assume.
  size   : aspect kept, longest side <= --max-side (1216), both dims multiples of 64 (or --size WxH).

Usage
  python prepare_clo3d_maps.py --depth d.png --normal n.png --rgb rgb.png --mask m.png
         [--invert-depth] [--flip-normal-x] [--flip-normal-y] [--prefix clo3d_jacket]
Outputs ROOT/outputs/prepared/<prefix>_{depth,normal,rgb,mask}.png (+ copies in COMFY/input unless --no-install)
        ROOT/outputs/prepared/<prefix>_transform.json
"""
import argparse, json, os, shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parent
COMFY = Path(r'C:\Dev\StabilityMatrix\Data\Packages\ComfyUI')

def read_depth(path, exr_channel=None):
    """Return (float64 HxW array in the file's own units, alpha or None)."""
    p = Path(path); ext = p.suffix.lower()
    if ext in ('.tif', '.tiff'):
        try:                                   # PIL reads 8/16-bit and 32-bit float ('F') single-page TIFF
            im = Image.open(p); a = np.asarray(im)
            if im.mode not in ('F', 'I', 'I;16', 'I;16B', 'I;16L', 'L', 'RGB', 'RGBA'): raise ValueError(im.mode)
        except Exception:                      # multi-sample / unusual TIFF: fall back to tifffile
            import tifffile
            a = tifffile.imread(str(p))
    elif ext == '.exr':
        os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
        import cv2
        a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if a is None: sys.exit(f'cannot read EXR {p} (cv2 without OpenEXR support?) — export depth as 16/32-bit TIFF instead')
        if a.ndim == 3:
            # cv2 returns BGR(A) by position; it cannot look up a channel by name, so "Z" is not supported.
            if exr_channel is None: sys.exit(f'EXR {p} has {a.shape[2]} channels: pass --exr-channel R|G|B|A '
                                             '(cv2 cannot read a named "Z" pass; export a single-channel depth instead)')
            ch = {'B': 0, 'G': 1, 'R': 2, 'A': 3}.get(exr_channel.upper())
            if ch is None or ch >= a.shape[2]: sys.exit(f'--exr-channel must be one of R,G,B,A present in the file (got {exr_channel!r}, {a.shape[2]} channels)')
            a = a[..., ch]
    else:
        im = Image.open(p)
        if im.mode in ('F', 'I', 'I;16', 'I;16B', 'I;16L'):
            a = np.asarray(im)
        else:
            a = np.asarray(im)          # 8-bit RGB(A)/L
    a = np.asarray(a)
    alpha = None
    if a.ndim == 3:
        if a.shape[2] == 4: alpha = a[..., 3]
        a = a[..., 0]                   # single channel: R (identical channels for grey exports)
    return a.astype(np.float64), alpha

def read_rgb(path):
    im = Image.open(path)
    a = np.asarray(im.convert('RGBA')) if 'A' in im.getbands() else np.asarray(im.convert('RGB'))
    return a

def alpha_to_mask(alpha):
    alpha = np.asarray(alpha).astype(np.float64)
    if alpha.max() > 1.0: alpha = alpha / (65535.0 if alpha.max() > 255 else 255.0)
    return alpha > 0.5

def fit_size(w, h, max_side, fixed=None):
    if fixed: return fixed
    s = max_side / max(w, h)
    return max(int(round(w * s / 64)) * 64, 64), max(int(round(h * s / 64)) * 64, 64)

def transform(arr, tw, th, scale, fill, resample):
    """Uniform-scale then centre-pad a HxW or HxWxC float array in [0,1]. Returns (PIL image, paste box)."""
    h, w = arr.shape[:2]; nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8)).resize((nw, nh), resample)
    canvas = Image.new(im.mode, (tw, th), fill)
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    canvas.paste(im, (ox, oy))
    return canvas, (ox, oy, nw, nh)

def prepare(depth, normal, fg, *, invert_depth=False, flip_x=False, flip_y=False, rgb=None,
            max_side=1216, size=None):
    """Pure function used by the CLI and the regression test. depth: float HxW, normal: uint8 HxWx3,
    fg: bool HxW, rgb: uint8 HxWx3 or HxWx4 (alpha composited over white) or None. Returns (dict of PIL images, meta)."""
    h, w = depth.shape
    assert normal.shape[:2] == (h, w), f'normal {normal.shape[:2]} != depth {(h, w)}'
    assert fg.shape == (h, w), 'mask size mismatch'
    if rgb is not None: assert rgb.shape[:2] == (h, w), f'rgb {rgb.shape[:2]} != depth {(h, w)}'
    valid = fg & np.isfinite(depth)
    if valid.sum() < 16: raise ValueError('foreground is empty — check the mask/alpha or use --bg-from-corner')
    d = -depth if invert_depth else depth.copy()          # larger = nearer after this
    zf = d[valid]; lo, hi = np.percentile(zf, 1), np.percentile(zf, 99.9)
    depth_n = np.where(valid, np.clip((d - lo) / max(hi - lo, 1e-12), 0, 1) * 0.85 + 0.15, 0.0)
    n = normal[..., :3].astype(np.float64) / 255.0
    if flip_x: n[..., 0] = 1 - n[..., 0]
    if flip_y: n[..., 1] = 1 - n[..., 1]
    n = np.where(valid[..., None], n, np.array([0.5, 0.5, 1.0]))
    tw, th = fit_size(w, h, max_side, size); scale = min(tw / w, th / h)
    out = {}
    out['depth'], box = transform(depth_n, tw, th, scale, 0, Image.BILINEAR)
    out['normal'], _ = transform(n, tw, th, scale, (128, 128, 255), Image.BILINEAR)
    out['mask'], _ = transform(valid.astype(np.float64), tw, th, scale, 0, Image.NEAREST)
    if rgb is not None:
        rgbf = rgb[..., :3].astype(np.float64) / 255.0
        if rgb.shape[2] == 4:                                   # composite transparent render over white
            al = rgb[..., 3:4].astype(np.float64) / 255.0
            rgbf = rgbf * al + (1.0 - al)
        out['rgb'], _ = transform(rgbf, tw, th, scale, (255, 255, 255), Image.LANCZOS)
    meta = {'source_size': [w, h], 'output_size': [tw, th], 'scale': scale, 'paste_box_xywh': list(box),
            'invert_depth': invert_depth, 'flip_normal_x': flip_x, 'flip_normal_y': flip_y,
            'depth_src_range_p1_p99.9': [float(lo), float(hi)], 'foreground_pixels': int(valid.sum())}
    return out, meta

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--depth', required=True); ap.add_argument('--normal', required=True)
    ap.add_argument('--rgb'); ap.add_argument('--mask'); ap.add_argument('--exr-channel', default=None)
    ap.add_argument('--alpha-from', choices=['auto', 'normal', 'rgb', 'depth', 'none'], default='auto')
    ap.add_argument('--bg-from-corner', action='store_true', help='opt-in heuristic when no mask/alpha')
    ap.add_argument('--depth-bg-tol', type=float, default=0.02, help='relative tolerance for --bg-from-corner')
    ap.add_argument('--invert-depth', action='store_true', help='input is distance/Z (far = larger)')
    ap.add_argument('--flip-normal-x', action='store_true'); ap.add_argument('--flip-normal-y', action='store_true')
    ap.add_argument('--max-side', type=int, default=1216); ap.add_argument('--size', help='WxH, overrides --max-side')
    ap.add_argument('--prefix', default='clo3d'); ap.add_argument('--no-install', action='store_true')
    a = ap.parse_args()

    depth, depth_alpha = read_depth(a.depth, a.exr_channel)
    normal = read_rgb(a.normal)
    rgb = read_rgb(a.rgb) if a.rgb else None
    h, w = depth.shape
    if normal.shape[:2] != (h, w): sys.exit(f'normal {normal.shape[1]}x{normal.shape[0]} != depth {w}x{h}: export from the same camera/resolution')
    if rgb is not None and rgb.shape[:2] != (h, w): sys.exit(f'rgb {rgb.shape[1]}x{rgb.shape[0]} != depth {w}x{h}')

    fg, fg_source = None, None
    if a.mask:
        m = Image.open(a.mask).convert('L')
        if m.size != (w, h): sys.exit(f'mask {m.size} != depth {(w, h)}')
        fg, fg_source = np.asarray(m) > 127, 'mask'
    else:
        alphas = {'normal': normal[..., 3] if normal.shape[2] == 4 else None,
                  'rgb': rgb[..., 3] if rgb is not None and rgb.shape[2] == 4 else None,
                  'depth': depth_alpha}
        order = ['normal', 'rgb', 'depth'] if a.alpha_from == 'auto' else ([a.alpha_from] if a.alpha_from != 'none' else [])
        for k in order:
            if alphas.get(k) is not None: fg, fg_source = alpha_to_mask(alphas[k]), f'alpha:{k}'; break
    if fg is None:
        if not a.bg_from_corner:
            sys.exit('no --mask and no alpha channel found. Export a garment mask/alpha, or pass --bg-from-corner to '
                     'use the corner-value heuristic (verify the resulting *_mask.png).')
        finite = np.isfinite(depth)
        corners = np.concatenate([depth[:8, :8].ravel(), depth[:8, -8:].ravel(), depth[-8:, :8].ravel(), depth[-8:, -8:].ravel()])
        corners = corners[np.isfinite(corners)]
        if corners.size == 0:
            fg, fg_source = finite, 'corner:non-finite-background'
        else:
            c = float(np.median(corners)); span = float(np.nanmax(depth[finite]) - np.nanmin(depth[finite])) or 1.0
            fg, fg_source = finite & (np.abs(depth - c) > a.depth_bg_tol * span), 'corner-heuristic'

    fixed = tuple(int(v) for v in a.size.lower().split('x')) if a.size else None
    out, meta = prepare(depth, normal, fg, invert_depth=a.invert_depth, flip_x=a.flip_normal_x, flip_y=a.flip_normal_y,
                        rgb=rgb, max_side=a.max_side, size=fixed)
    meta['foreground_source'] = fg_source; meta['inputs'] = {'depth': a.depth, 'normal': a.normal, 'rgb': a.rgb, 'mask': a.mask}
    outdir = root / 'outputs' / 'prepared'; outdir.mkdir(exist_ok=True)
    meta['files'] = {}
    for k, im in out.items():
        fn = f'{a.prefix}_{k}.png'; im.save(outdir / fn); meta['files'][k] = fn
        if not a.no_install: shutil.copyfile(outdir / fn, COMFY / 'input' / fn)
    (outdir / f'{a.prefix}_transform.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))

if __name__ == '__main__':
    main()
