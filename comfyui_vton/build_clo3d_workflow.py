"""CLO3D Depth+Normal -> SDXL ControlNet Union chain workflow builder (v2).

Generates:
  outputs/CLO3D-ControlNet-Union.json   ComfyUI UI workflow (drag into canvas)
  work/clo3d_api_prompt.json            /prompt API payload (same graph)
  outputs/clo3d_test_depth.png          synthetic garment-like test maps (also copied to COMFY/input)
  outputs/clo3d_test_normal.png

v2 changes (after review, see outputs/CLO3D_REVIEW.md):
  * no square centre crop: output size is taken from the depth map via GetImageSize, so maps of any
    aspect ratio work as long as they are SDXL-friendly sizes (multiples of 64, ~1MP). Use
    work/prepare_clo3d_maps.py to letterbox real CLO3D exports with one identical transform.
  * synthetic maps are a physically consistent pair: depth is a heightfield, normal is its gradient.
  * normal colour convention measured from NormalBae (xinsir's reference preprocessor), see
    work/probe_normal_convention.py:  R = -x (left-facing is red), G = +y (up), B = +z (toward camera).
  * garment-like asymmetric shape (torso, two sleeves, collar band, seam, folds on one side only)
    so depth-only / normal-only / both can be compared.
  * prompt targets a product-style garment image (front view, no person) for the VTON garment input.
"""
import json, shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parent
COMFY = Path(r'C:\Dev\StabilityMatrix\Data\Packages\ComfyUI')
W, H = 832, 1216                                   # portrait SDXL size for the synthetic maps
CKPT = 'RealVisXL_V5.0_fp16.safetensors'
CN = 'diffusion_pytorch_model_promax.safetensors'   # xinsir ControlNet++ Union SDXL ProMax
DEPTH_IMG, NORMAL_IMG = 'clo3d_test_depth.png', 'clo3d_test_normal.png'
POS = ('front view product photo of a raw denim jacket, ghost mannequin style, entire garment visible '
       'including sleeves and hem, highly detailed denim texture, visible stitching and seams, '
       'photorealistic, soft studio lighting, plain white background, no person')
NEG = 'person, face, skin, hands, blurry, lowres, deformed, text, watermark, cartoon, painting, cropped'
SEED, STEPS, CFG = 42, 25, 5.0
SAMPLER, SCHED = 'dpmpp_2m', 'karras'
NORMAL_STR, NORMAL_END = 0.6, 0.8      # normal first (surface detail), depth second (volume)
DEPTH_STR, DEPTH_END = 0.75, 0.9

# ---------------------------------------------------------------- synthetic garment-like maps
def smoothmax(a, b, k=0.15):
    m = np.maximum(a, b)                          # numerically stable log-sum-exp
    return m + np.log1p(np.exp(-k * np.abs(a - b))) / k

def make_test_maps():
    """Heightfield z(x,y) in pixels (larger = closer to camera). Normal = gradient of the same field,
    encoded with the BAE convention measured in probe_normal_convention.py."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    def cyl(cx, y0, y1, hw, r_depth, tilt=0.0, top_round=0.25, bottom_round=0.9):
        """Cylinder with circular cross-section along x, spanning y0..y1, rounded ends."""
        x = xx - cx - tilt * (yy - (y0 + y1) / 2)
        u = np.clip(x / hw, -1, 1)
        vz = np.sqrt(np.clip(1 - u * u, 0, 1))
        inside_x = np.abs(x) < hw
        rt, rb = hw * top_round, hw * bottom_round
        cap = np.ones_like(yy)
        cap = np.where(yy < y0 + rt, np.sqrt(np.clip(1 - ((y0 + rt - yy) / rt) ** 2, 0, 1)), cap)
        cap = np.where(yy > y1 - rb, np.sqrt(np.clip(1 - ((yy - (y1 - rb)) / rb) ** 2, 0, 1)), cap)
        cap = np.where((yy < y0) | (yy > y1), 0.0, cap)
        return np.where(inside_x, vz * cap * r_depth, 0.0)
    torso = cyl(W * 0.5, H * 0.20, H * 0.92, W * 0.30, 170)
    # sleeves: asymmetric (left longer and more tilted than right)
    sleeve_l = cyl(W * 0.155, H * 0.22, H * 0.72, W * 0.085, 120, tilt=0.22, top_round=0.6, bottom_round=0.5)
    sleeve_r = cyl(W * 0.84, H * 0.22, H * 0.62, W * 0.085, 120, tilt=-0.14, top_round=0.6, bottom_round=0.5)
    # support mask from the raw shapes; smoothmax adds a positive offset everywhere (7.32 at 0,0,0),
    # so the blended field must be re-based and clamped to the support, not thresholded.
    body = (torso > 0) | (sleeve_l > 0) | (sleeve_r > 0)
    z = smoothmax(smoothmax(torso, sleeve_l), sleeve_r)
    z = z - smoothmax(smoothmax(0.0, 0.0), 0.0)
    z[~body] = 0.0
    # collar band: raised ring around the neckline
    ncx, ncy = W * 0.5, H * 0.235
    rr = np.sqrt(((xx - ncx) / (W * 0.13)) ** 2 + ((yy - ncy) / (H * 0.06)) ** 2)
    collar = 18 * np.exp(-((rr - 1.0) / 0.18) ** 2) * (yy > ncy - H * 0.01)
    # centre seam / placket ridge
    seam = 6 * np.exp(-((xx - W * 0.5) / 5.0) ** 2) * (yy > ncy + H * 0.05)
    # folds only on the right half of the torso (normal-only contribution test)
    win = (np.clip(np.sin(np.pi * np.clip((xx - W * 0.56) / (W * 0.20), 0, 1)), 0, 1)
           * np.clip(np.sin(np.pi * np.clip((yy - H * 0.40) / (H * 0.48), 0, 1)), 0, 1))   # smooth window
    folds = 4 * np.sin((yy - H * 0.55) / 30.0 + 1.2 * np.sin(xx / 45.0)) * win
    hole = rr < 0.82                                       # neckline opening
    body = body & ~hole
    z = np.where(body, z + collar + seam + folds, 0.0)
    z = np.clip(z, 0.0, None)
    # depth: near = white, foreground normalised, background black
    zf = z[body]; lo, hi = np.percentile(zf, 1), zf.max()   # foreground-only statistics
    depth = np.where(body, np.clip((z - lo) / (hi - lo), 0, 1) * 0.85 + 0.15, 0.0)
    # normal from gradient (x right, y down in image; y up in normal space)
    dzdy, dzdx = np.gradient(z)
    n = np.stack([-dzdx, dzdy, np.ones_like(z)], -1)      # (x_right, y_up, z_toward_camera)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    # BAE encoding measured empirically: R = (-x+1)/2, G = (y+1)/2, B = (z+1)/2
    enc = np.stack([(-n[..., 0] + 1) / 2, (n[..., 1] + 1) / 2, (n[..., 2] + 1) / 2], -1)
    bg = np.array([0.5, 0.5, 1.0])                        # flat, facing camera
    normal = np.where(body[..., None], enc, bg)
    make_test_maps.last = {'z': z, 'body': body, 'depth': depth, 'normal': normal}
    (root / 'outputs').mkdir(exist_ok=True)
    Image.fromarray((depth * 255).round().astype(np.uint8)).save(root / 'outputs' / DEPTH_IMG)
    Image.fromarray((normal * 255).round().astype(np.uint8)).save(root / 'outputs' / NORMAL_IMG)
    for f in (DEPTH_IMG, NORMAL_IMG):
        shutil.copyfile(root / 'outputs' / f, COMFY / 'input' / f)

# ---------------------------------------------------------------- UI workflow
nodes, links = [], []
def node(i, typ, pos, inputs, outputs, widgets, title=None, size=None):
    n = {'id': i, 'type': typ, 'pos': pos, 'size': size or [320, 120], 'flags': {}, 'order': i - 1, 'mode': 0,
         'inputs': [], 'outputs': [{'name': a, 'type': b, 'links': [], 'slot_index': k} for k, (a, b) in enumerate(outputs)],
         'properties': {'Node name for S&R': typ}, 'widgets_values': widgets}
    for spec in inputs:
        name, kind = spec[:2]
        d = {'name': name, 'type': kind, 'link': None}
        if len(spec) > 2: d['widget'] = {'name': name}       # widget converted to input
        n['inputs'].append(d)
    if title: n['title'] = title
    nodes.append(n)
def link(a, ao, b, bi, kind):
    idx = len(links) + 1
    links.append([idx, a, ao, b, bi, kind])
    next(n for n in nodes if n['id'] == a)['outputs'][ao]['links'].append(idx)
    next(n for n in nodes if n['id'] == b)['inputs'][bi]['link'] = idx

NOTE = ('12GB 세팅: SDXL fp16 체크포인트 + ControlNet Union 1개 로드, 배치 1 고정.\n'
        'Union 모델은 SetUnionControlNetType으로 normal/depth 타입을 나눠 같은 가중치를 두 번 적용한다.\n'
        '체인: Positive/Negative -> Apply(normal) -> Apply(depth) -> KSampler. 출력 크기는 depth 맵 크기를 그대로 쓴다(GetImageSize).\n'
        '맵 준비: work\\prepare_clo3d_maps.py 로 depth/normal/RGB에 동일한 letterbox 변환을 적용해 64의 배수 크기로 만든다.\n'
        'depth: 가까울수록 흰색, 배경 검정. normal: BAE 규약 = R 왼쪽(-x), G 위(+y), B 카메라 방향(+z), 정면=(128,128,255).\n'
        'CLO3D의 Bake Normal(UV 탱전트 공간)은 사용 불가. clo3d_test_*.png는 합성 검증용.\n'
        'OOM 시: 1) 다른 GPU 프로세스 종료 2) 해상도 축소 3) 한 ControlNet 분기를 strength 0으로 실제 비활성화(0이면 미실행)\n'
        '4) VAE decode OOM이면 VAEDecodeTiled 5) ComfyUI --lowvram. 양수 strength 축소는 텐서 크기를 줄이지 않고,\n'
        'end 축소는 초기 동시 실행 피크 해소를 보장하지 않는다.')

node(1, 'CheckpointLoaderSimple', [0, 0], [], [('MODEL', 'MODEL'), ('CLIP', 'CLIP'), ('VAE', 'VAE')], [CKPT], 'SDXL (RealVisXL V5 fp16)', [380, 100])
node(2, 'CLIPTextEncode', [420, 0], [('clip', 'CLIP')], [('CONDITIONING', 'CONDITIONING')], [POS], 'Positive: garment product photo + fabric', [420, 180])
node(3, 'CLIPTextEncode', [420, 220], [('clip', 'CLIP')], [('CONDITIONING', 'CONDITIONING')], [NEG], 'Negative', [420, 120])
node(4, 'LoadImage', [0, 400], [], [('IMAGE', 'IMAGE'), ('MASK', 'MASK')], [NORMAL_IMG, 'image'], 'CLO3D Normal map (BAE convention)', [320, 460])
node(5, 'LoadImage', [0, 900], [], [('IMAGE', 'IMAGE'), ('MASK', 'MASK')], [DEPTH_IMG, 'image'], 'CLO3D Depth map (near = white)', [320, 460])
node(6, 'GetImageSize', [360, 900], [('image', 'IMAGE')], [('width', 'INT'), ('height', 'INT'), ('batch_size', 'INT')], [], 'Output size = depth map size', [260, 80])
node(8, 'ControlNetLoader', [0, 1400], [], [('CONTROL_NET', 'CONTROL_NET')], [CN], 'ControlNet++ Union SDXL ProMax (1 model for both)', [380, 80])
node(9, 'SetUnionControlNetType', [420, 1360], [('control_net', 'CONTROL_NET')], [('CONTROL_NET', 'CONTROL_NET')], ['normal'], 'Union type = normal', [300, 80])
node(10, 'SetUnionControlNetType', [420, 1480], [('control_net', 'CONTROL_NET')], [('CONTROL_NET', 'CONTROL_NET')], ['depth'], 'Union type = depth', [300, 80])
node(11, 'ControlNetApplyAdvanced', [900, 300], [('positive', 'CONDITIONING'), ('negative', 'CONDITIONING'), ('control_net', 'CONTROL_NET'), ('image', 'IMAGE'), ('vae', 'VAE')], [('positive', 'CONDITIONING'), ('negative', 'CONDITIONING')], [NORMAL_STR, 0.0, NORMAL_END], '1) Apply NORMAL (surface detail)', [340, 200])
node(12, 'ControlNetApplyAdvanced', [1300, 300], [('positive', 'CONDITIONING'), ('negative', 'CONDITIONING'), ('control_net', 'CONTROL_NET'), ('image', 'IMAGE'), ('vae', 'VAE')], [('positive', 'CONDITIONING'), ('negative', 'CONDITIONING')], [DEPTH_STR, 0.0, DEPTH_END], '2) Apply DEPTH (volume) — chained', [340, 200])
node(13, 'EmptyLatentImage', [1300, 560], [('width', 'INT', 'w'), ('height', 'INT', 'w')], [('LATENT', 'LATENT')], [W, H, 1], 'Latent (size from depth map, batch 1)', [300, 130])
node(14, 'KSampler', [1700, 300], [('model', 'MODEL'), ('positive', 'CONDITIONING'), ('negative', 'CONDITIONING'), ('latent_image', 'LATENT')], [('LATENT', 'LATENT')], [SEED, 'fixed', STEPS, CFG, SAMPLER, SCHED, 1.0], size=[320, 280])
node(15, 'VAEDecode', [2060, 300], [('samples', 'LATENT'), ('vae', 'VAE')], [('IMAGE', 'IMAGE')], [], size=[220, 60])
node(16, 'SaveImage', [2320, 300], [('images', 'IMAGE')], [], ['CLO3D/controlnet'], size=[420, 560])
node(17, 'Note', [900, 560], [], [], [NOTE], size=[380, 300])

for a, ao, b, bi, t in [
    (1, 1, 2, 0, 'CLIP'), (1, 1, 3, 0, 'CLIP'),
    (5, 0, 6, 0, 'IMAGE'),
    (8, 0, 9, 0, 'CONTROL_NET'), (8, 0, 10, 0, 'CONTROL_NET'),
    (2, 0, 11, 0, 'CONDITIONING'), (3, 0, 11, 1, 'CONDITIONING'), (9, 0, 11, 2, 'CONTROL_NET'), (4, 0, 11, 3, 'IMAGE'), (1, 2, 11, 4, 'VAE'),
    (11, 0, 12, 0, 'CONDITIONING'), (11, 1, 12, 1, 'CONDITIONING'), (10, 0, 12, 2, 'CONTROL_NET'), (5, 0, 12, 3, 'IMAGE'), (1, 2, 12, 4, 'VAE'),
    (6, 0, 13, 0, 'INT'), (6, 1, 13, 1, 'INT'),
    (1, 0, 14, 0, 'MODEL'), (12, 0, 14, 1, 'CONDITIONING'), (12, 1, 14, 2, 'CONDITIONING'), (13, 0, 14, 3, 'LATENT'),
    (14, 0, 15, 0, 'LATENT'), (1, 2, 15, 1, 'VAE'), (15, 0, 16, 0, 'IMAGE')]:
    link(a, ao, b, bi, t)

workflow = {'last_node_id': 17, 'last_link_id': len(links), 'nodes': nodes, 'links': links, 'groups': [], 'config': {},
            'extra': {'ds': {'scale': .5, 'offset': [30, 40]}}, 'version': .4}

# ---------------------------------------------------------------- API prompt (same graph)
def api_prompt(normal_str=NORMAL_STR, depth_str=DEPTH_STR, prefix='CLO3D/controlnet', normal_img=NORMAL_IMG, depth_img=DEPTH_IMG, seed=SEED):
    return {
        '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': CKPT}},
        '2': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['1', 1], 'text': POS}},
        '3': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['1', 1], 'text': NEG}},
        '4': {'class_type': 'LoadImage', 'inputs': {'image': normal_img}},
        '5': {'class_type': 'LoadImage', 'inputs': {'image': depth_img}},
        '6': {'class_type': 'GetImageSize', 'inputs': {'image': ['5', 0]}},
        '8': {'class_type': 'ControlNetLoader', 'inputs': {'control_net_name': CN}},
        '9': {'class_type': 'SetUnionControlNetType', 'inputs': {'control_net': ['8', 0], 'type': 'normal'}},
        '10': {'class_type': 'SetUnionControlNetType', 'inputs': {'control_net': ['8', 0], 'type': 'depth'}},
        '11': {'class_type': 'ControlNetApplyAdvanced', 'inputs': {'positive': ['2', 0], 'negative': ['3', 0], 'control_net': ['9', 0], 'image': ['4', 0], 'vae': ['1', 2], 'strength': normal_str, 'start_percent': 0.0, 'end_percent': NORMAL_END}},
        '12': {'class_type': 'ControlNetApplyAdvanced', 'inputs': {'positive': ['11', 0], 'negative': ['11', 1], 'control_net': ['10', 0], 'image': ['5', 0], 'vae': ['1', 2], 'strength': depth_str, 'start_percent': 0.0, 'end_percent': DEPTH_END}},
        '13': {'class_type': 'EmptyLatentImage', 'inputs': {'width': ['6', 0], 'height': ['6', 1], 'batch_size': 1}},
        '14': {'class_type': 'KSampler', 'inputs': {'model': ['1', 0], 'positive': ['12', 0], 'negative': ['12', 1], 'latent_image': ['13', 0], 'seed': seed, 'steps': STEPS, 'cfg': CFG, 'sampler_name': SAMPLER, 'scheduler': SCHED, 'denoise': 1.0}},
        '15': {'class_type': 'VAEDecode', 'inputs': {'samples': ['14', 0], 'vae': ['1', 2]}},
        '16': {'class_type': 'SaveImage', 'inputs': {'images': ['15', 0], 'filename_prefix': prefix}},
    }
api = api_prompt()

if __name__ == '__main__':
    if '--no-maps' not in sys.argv:
        make_test_maps()
    (root / 'outputs/CLO3D-ControlNet-Union.json').write_text(json.dumps(workflow, indent=2, ensure_ascii=False), encoding='utf-8')
    (root / 'work/clo3d_api_prompt.json').write_text(json.dumps(api, indent=2, ensure_ascii=False), encoding='utf-8')
    if '--install' in sys.argv:
        shutil.copyfile(root / 'outputs/CLO3D-ControlNet-Union.json', COMFY / 'user/default/workflows/CLO3D-ControlNet-Union.json')
    print('ok')
