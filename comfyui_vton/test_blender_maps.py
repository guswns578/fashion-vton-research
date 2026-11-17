"""Regression checks for the Blender map-extraction path (needs Blender 4.5, no GPU/ComfyUI).
Run (from comfyui_vton/):  python -m unittest test_blender_maps -v
Set BLENDER_NO_RENDER=1 to reuse an existing demo render instead of running Blender again.
Covers outputs/CLO3D_BLENDER_REVIEW.md:
  1. alpha cutouts (Principled Alpha texture) stay holes in normal alpha AND depth, not filled
  2. demo goes through OBJ export -> import_garment() (importer path exercised)
  3. conventions: sphere facing B high, left R low (OpenGL) -> after prepare --flip-normal-x left R high (BAE);
     depth near < far (uint16), background 65535 -> prepare --invert-depth: near bright, bg 0
  4. prepare: RGBA reference composited over white (corners white, not black)
  5. EXR: multi-channel without --exr-channel and --exr-channel Z are rejected
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
import prepare_clo3d_maps as P  # noqa: E402

BLENDER = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
demo = root / "work" / "blender_demo"
NO_RENDER = os.environ.get("BLENDER_NO_RENDER") == "1"


class Checks(unittest.TestCase):
    def check(self, cond, msg):
        with self.subTest(msg):
            self.assertTrue(cond, msg)


@unittest.skipUnless(
    Path(BLENDER).exists() or (NO_RENDER and (demo / "demo_camera.json").exists()),
    "Blender 4.5 not installed and no existing demo render",
)
class BlenderMapsTest(Checks):
    @classmethod
    def setUpClass(cls):
        if not NO_RENDER:
            r = subprocess.run(
                [
                    BLENDER,
                    "-b",
                    "-P",
                    str(root / "blender_render_maps.py"),
                    "--",
                    "--demo",
                    "--out",
                    str(demo),
                    "--prefix",
                    "demo",
                    "--res",
                    "832",
                    "1216",
                    "--rgb",
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0 or "DONE" not in r.stdout:
                raise RuntimeError(
                    "blender demo render failed\n" + r.stdout[-3000:] + r.stderr[-3000:]
                )
        cls.cam = json.loads((demo / "demo_camera.json").read_text(encoding="utf-8"))
        cls.nrm = np.asarray(Image.open(demo / "demo_normal.png"))  # RGBA uint8
        cls.dep = np.asarray(Image.open(demo / "demo_depth.png")).astype(int)  # uint16
        cls.rgb = np.asarray(Image.open(demo / "demo_rgb.png"))
        cls.H, cls.W = cls.dep.shape
        # project world -> pixel with the ortho camera, using the camera meta
        lo, hi = np.array(cls.cam["bbox_min"]), np.array(cls.cam["bbox_max"])
        cls.centre = (lo + hi) / 2
        cls.px_per_unit = max(cls.W, cls.H) / cls.cam["ortho_scale"]

    def to_px(self, x, z):
        return int(round(self.W / 2 + (x - self.centre[0]) * self.px_per_unit)), int(
            round(self.H / 2 - (z - self.centre[2]) * self.px_per_unit)
        )

    def plane_region(self):
        x0, y1 = self.to_px(-0.44, 0.56)
        x1, y0 = self.to_px(-0.16, 0.84)  # plane extents in pixels (inset 1 cm)
        return x0, x1, y0, y1

    def test_render_outputs(self):
        nrm, dep, rgb, H, W = self.nrm, self.dep, self.rgb, self.H, self.W
        self.check((demo / "demo_src.obj").exists(), "demo was round-tripped through an OBJ file")
        self.check(
            nrm.shape == (H, W, 4) and rgb.shape[:2] == (H, W),
            "normal/depth/rgb have one resolution",
        )
        self.check(
            dep[0, 0] == 65535 and nrm[0, 0, 3] == 0,
            "background: depth 65535 (far), normal alpha 0",
        )

    # --- 1. alpha plane: 0.3 m plane at world (-0.3,-0.7,0.7), in front of the sphere; checker scale 6 -> holes
    def test_alpha_cutout_holes(self):
        x0, x1, y0, y1 = self.plane_region()
        region_a = self.nrm[y0:y1, x0:x1, 3]
        region_d = self.dep[y0:y1, x0:x1]
        region_n = self.nrm[y0:y1, x0:x1, :3].astype(int)
        # the plane is perpendicular to the ortho camera, so its depth is one constant = the nearest value in the region;
        # pixels noticeably farther than that are holes (alpha stays 1 there because the sphere is behind the plane)
        plane_depth = region_d.min()
        solid = region_d < plane_depth + 300
        holes = ~solid
        self.check(
            0.25 < holes.mean() < 0.75,
            f"checker holes in the alpha plane: {holes.mean():.2f} of its area sees past the plane",
        )
        self.check(
            np.median(region_d[holes]) > plane_depth + 500,
            f"depth through holes is farther than the plane ({np.median(region_d[holes]):.0f} vs {plane_depth})",
        )
        sphere_seen_through = (region_d[holes] < 65535).mean()
        self.check(
            sphere_seen_through > 0.5,
            f"sphere behind the cutout is visible through holes ({sphere_seen_through:.2f})",
        )
        self.check(
            region_a[holes].mean() > 200,
            "coverage alpha stays 1 through holes (sphere behind), i.e. cutout != background",
        )
        # plane faces the camera (R~128); the sphere surface behind it on the left has R well below 128
        self.check(
            abs(region_n[solid][:, 0].mean() - 128) < 8 and region_n[holes][:, 0].mean() < 100,
            f"normals through holes are the sphere surface (R={region_n[holes][:, 0].mean():.0f}), plane R={region_n[solid][:, 0].mean():.0f}",
        )

    # --- 3. conventions on the sphere (centre (0,0,0.9), r 0.5): sample interior points
    # --- 4. RGBA reference composited over white
    def test_conventions_and_prepare(self):
        nrm, dep, rgb = self.nrm, self.dep, self.rgb
        sx, sy = self.to_px(0.0, 0.9)
        lx, _ = self.to_px(-0.42, 0.9)
        rx, _ = self.to_px(0.42, 0.9)
        _, ty = self.to_px(0.0, 1.32)
        _, by = self.to_px(0.0, 0.48)

        def m(x, y, s=4):
            return nrm[y - s : y + s, x - s : x + s, :3].reshape(-1, 3).mean(0)

        c, l, r_, t, b = m(sx, sy), m(lx, sy), m(rx, sy), m(sx, ty), m(sx, by)
        self.check(c[2] > 240, f"sphere centre faces camera: B={c[2]:.0f}")
        self.check(l[0] < 80 < 180 < r_[0], f"OpenGL x: left R={l[0]:.0f} < right R={r_[0]:.0f}")
        self.check(t[1] > 180 > 80 > b[1], f"y up: top G={t[1]:.0f} > bottom G={b[1]:.0f}")
        self.check(
            dep[sy, sx] < dep[sy, lx] < 65535,
            f"depth: centre {dep[sy, sx]} nearer than rim {dep[sy, lx]} < bg",
        )

        # prepare with the documented flags
        fg = nrm[..., 3] > 127
        out, meta = P.prepare(
            dep.astype(np.float64),
            nrm[..., :3],
            fg,
            invert_depth=True,
            flip_x=True,
            rgb=rgb,
            max_side=1216,
        )
        pn = np.asarray(out["normal"])
        pd = np.asarray(out["depth"])
        pr = np.asarray(out["rgb"])
        pm = np.asarray(out["mask"])
        x0, x1, y0, y1 = self.plane_region()
        self.check(
            pn[sy, lx, 0] > 180 and pn[sy, rx, 0] < 80,
            f"after --flip-normal-x: BAE left R={pn[sy, lx, 0]} high, right R={pn[sy, rx, 0]} low",
        )
        self.check(
            pd[sy, sx] > pd[sy, lx] > 0 and pd[0, 0] == 0,
            f"after --invert-depth: centre {pd[sy, sx]} > rim {pd[sy, lx]} > bg 0",
        )
        self.check(tuple(pn[0, 0]) == (128, 128, 255), "prepared normal background (128,128,255)")
        self.check(
            pm[y0:y1, x0:x1].mean() > 250,
            "prepared mask = coverage (sphere behind holes -> region fully covered)",
        )
        self.check(
            tuple(pr[0, 0]) == (255, 255, 255) and tuple(pr[-1, -1]) == (255, 255, 255),
            f"prepared rgb corners are white ({pr[0, 0].tolist()})",
        )


# --- 5. EXR channel policy (cv2 decode mocked: BGR array); needs no Blender
class ExrChannelTest(Checks):
    def test_exr_channel_policy(self):
        fake = np.zeros((4, 4, 3), np.float32)
        fake[:] = (0.2, 0.4, 0.8)
        with patch.object(cv2, "imread", return_value=fake):
            for chan, expect_exit in ((None, True), ("Z", True), ("R", False)):
                try:
                    d, _ = P.read_depth("fake.exr", chan)
                    exited = False
                except SystemExit:
                    exited = True
                self.check(
                    exited == expect_exit,
                    f'EXR --exr-channel {chan}: {"rejected" if exited else "accepted"}',
                )
            d, _ = P.read_depth("fake.exr", "R")
            self.check(
                abs(d[0, 0] - 0.8) < 1e-6, "EXR --exr-channel R returns the R value (0.8), not B"
            )


if __name__ == "__main__":
    unittest.main()
