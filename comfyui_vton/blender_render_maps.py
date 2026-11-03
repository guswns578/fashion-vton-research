"""Render camera-space normal + linear depth (+ alpha, optional RGB) from a CLO3D garment export
(OBJ/FBX/glTF) with Blender, for prepare_clo3d_maps.py.

The CLO desktop documentation we checked has no separate depth / camera-space normal output option
(UI of the installed version not verified), so the garment is exported as geometry and the maps are
rendered here where every convention is under our control. Cycles is used with 1 sample + emission
shaders, so the output is exact (no lighting, no noise) and the same maps come out on CPU or GPU.

Material opacity IS respected: each material's Principled BSDF Alpha (value or texture, e.g. an OBJ
map_d lace/mesh cutout) drives a Transparent BSDF mix in the normal/depth shaders, so alpha holes
stay holes in the maps and in the coverage mask. Material bump/normal/displacement maps are NOT
reflected (geometry normals only). Scope: opaque or alpha-cutout meshes; the large-scale shape.

Run (Blender 4.2+, background mode):
  "C:\\Program Files\\Blender Foundation\\Blender 4.5\\blender.exe" -b -P work\\blender_render_maps.py -- ^
      --input C:\\path\\garment.obj --out C:\\path\\render --res 832 1216 [--rgb] [--view front|back] [--yaw 0]
  (no --input  ->  --demo: builds sphere + two cylinders + an alpha-checker plane, exports them to OBJ and
   re-imports that file through import_garment(), so the importer path and alpha cutouts are exercised)

Outputs in --out:
  <prefix>_normal.png   8-bit RGBA. Camera space, OpenGL: R=+x (right), G=+y (up), B=+z (toward camera),
                        encoded n*0.5+0.5. Alpha = garment coverage (film transparent).
                        prepare_clo3d_maps.py wants BAE (R=-x)  ->  pass --flip-normal-x there.
  <prefix>_depth.png    16-bit grayscale, LINEAR distance from the orthographic camera plane mapped
                        near..far -> 0..65535 (near = dark). Background = 65535 (far).
                        -> pass --invert-depth to prepare_clo3d_maps.py.
  <prefix>_rgb.png      (with --rgb) plain render of the imported materials, transparent background.
  <prefix>_camera.json  near/far distances, ortho_scale, bbox, resolution, so depth can be mapped back.

Axis notes: CLO's exporter axis/unit settings are options (see CLO API ImportExportOption), commonly
Y-up with the front facing +Z. Blender's importers default to forward=-Z / up=Y, in which case the
front faces Blender -Y and the default "front" camera (at -Y looking +Y) sees it. Do not assume:
check the RGB/normal render of the real file; use --view back or --yaw if it faces the wrong way.
"""
import argparse, json, math, sys
from pathlib import Path
import bpy
from mathutils import Vector

def parse_args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', help='OBJ / FBX / GLB / GLTF exported from CLO3D')
    ap.add_argument('--demo', action='store_true', help='render a synthetic sphere + cylinder instead')
    ap.add_argument('--out', required=True); ap.add_argument('--prefix', default='clo3d')
    ap.add_argument('--res', nargs=2, type=int, default=[832, 1216], metavar=('W', 'H'))
    ap.add_argument('--margin', type=float, default=1.08, help='bbox padding factor inside the frame')
    ap.add_argument('--view', choices=['front', 'back'], default='front')
    ap.add_argument('--yaw', type=float, default=0.0, help='extra rotation of the camera around the vertical axis (deg)')
    ap.add_argument('--rgb', action='store_true'); ap.add_argument('--rgb-samples', type=int, default=32)
    ap.add_argument('--smooth', action='store_true',
                    help='drop per-face custom normals from the file and shade smooth (for flat-exported sim meshes)')
    ap.add_argument('--relax', type=int, default=0, metavar='N',
                    help='apply a Smooth modifier (factor 0.5, N iterations) to a crumpled sim mesh; changes the shape slightly, 0 = off')
    return ap.parse_args(argv)

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def smooth_meshes(meshes):
    """Flat-exported OBJs (one `vn` per face, `s 0`) give a faceted normal map; cloth is smooth, so
    remove the imported custom normals and let Blender average them per vertex."""
    for o in meshes:
        me = o.data
        for name in ('custom_normal', 'sharp_edge', 'sharp_face'):   # Blender 4.1+: stored as attributes
            attr = me.attributes.get(name)
            if attr is not None: me.attributes.remove(attr)
        me.shade_smooth()

def relax_meshes(meshes, iterations):
    """Geometric smoothing for coarse / crumpled simulation output. Not needed for a clean CLO export."""
    for o in meshes:
        mod = o.modifiers.new('Relax', 'SMOOTH'); mod.factor = 0.5; mod.iterations = iterations

def import_garment(path):
    p = Path(path); ext = p.suffix.lower()
    if ext == '.obj': bpy.ops.wm.obj_import(filepath=str(p))
    elif ext == '.fbx': bpy.ops.import_scene.fbx(filepath=str(p))
    elif ext in ('.glb', '.gltf'): bpy.ops.import_scene.gltf(filepath=str(p))
    else: sys.exit(f'unsupported input {ext}')
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not meshes: sys.exit('no mesh objects imported')
    return meshes

def checker_alpha_material():
    """Principled material whose Alpha comes from a checker texture (stands in for an OBJ map_d cutout)."""
    m = bpy.data.materials.new('DemoAlphaChecker'); m.use_nodes = True
    nt = m.node_tree; bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
    bsdf.inputs['Base Color'].default_value = (0.8, 0.2, 0.2, 1)
    chk = nt.nodes.new('ShaderNodeTexChecker'); chk.inputs['Scale'].default_value = 6.0
    chk.inputs['Color1'].default_value = (1, 1, 1, 1); chk.inputs['Color2'].default_value = (0, 0, 0, 1)
    nt.links.new(chk.outputs['Fac'], bsdf.inputs['Alpha'])
    return m

def make_demo(out_dir):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, segments=96, ring_count=48, location=(0, 0, 0.9))
    bpy.ops.mesh.primitive_cylinder_add(radius=0.14, depth=1.0, vertices=64, location=(-0.62, 0, 0.35),
                                        rotation=(0, math.radians(20), 0))
    bpy.ops.mesh.primitive_cylinder_add(radius=0.14, depth=0.8, vertices=64, location=(0.60, 0, 0.45),
                                        rotation=(0, math.radians(-12), 0))
    # alpha-cutout plane in front of the sphere (lace / mesh stand-in), facing the front camera (-Y)
    bpy.ops.mesh.primitive_plane_add(size=0.3, location=(-0.3, -0.7, 0.7), rotation=(math.radians(90), 0, 0))
    plane = bpy.context.active_object; plane.name = 'AlphaPlane'
    for o in bpy.context.scene.objects:
        if o.type == 'MESH' and o is not plane:
            o.data.shade_smooth()
    # round-trip through OBJ so the real import path is exercised (geometry only; the procedural alpha
    # material is re-attached after import because OBJ cannot carry a checker node)
    obj_path = Path(out_dir) / 'demo_src.obj'
    bpy.ops.wm.obj_export(filepath=str(obj_path), export_materials=True)
    clear_scene()
    meshes = import_garment(obj_path)
    for o in meshes:
        if o.name.startswith('AlphaPlane'):
            o.data.materials.clear(); o.data.materials.append(checker_alpha_material())
    return meshes

def world_bbox(meshes):
    pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi

def setup_camera(scene, lo, hi, res, margin, view, yaw_deg):
    cam_data = bpy.data.cameras.new('MapCam'); cam_data.type = 'ORTHO'
    cam = bpy.data.objects.new('MapCam', cam_data); scene.collection.objects.link(cam); scene.camera = cam
    centre = (lo + hi) / 2; size = hi - lo
    radius = size.length / 2 + 1e-3                     # bounding sphere: valid for any yaw
    # direction from centre to camera in the XY plane: front = -Y, back = +Y, plus yaw about Z
    base = math.radians(-90.0 if view == 'front' else 90.0) + math.radians(yaw_deg)
    dirv = Vector((math.cos(base), math.sin(base), 0.0))
    dist = radius * 2 + 1.0
    cam.location = centre + dirv * dist
    # look at centre: camera looks down its local -Z; rotate so -Z points to -dirv, +Y stays up
    cam.rotation_euler = (math.radians(90.0), 0.0, base + math.radians(90.0))
    w, h = res
    # visible extents in camera plane: use projected bbox for a tighter fit
    right = Vector((-dirv.y, dirv.x, 0.0))
    corners = [Vector((x, y, z)) for x in (lo.x, hi.x) for y in (lo.y, hi.y) for z in (lo.z, hi.z)]
    ext_w = max(abs((c - centre).dot(right)) for c in corners) * 2
    ext_h = size.z
    # ortho_scale applies to the larger render dimension (sensor fit AUTO)
    if h >= w: ortho = max(ext_h, ext_w * h / w) * margin
    else:      ortho = max(ext_w, ext_h * w / h) * margin
    cam_data.ortho_scale = ortho
    near = dist - radius; far = dist + radius
    cam_data.clip_start = max(near * 0.5, 0.01); cam_data.clip_end = far * 1.5
    return cam, near, far, ortho


def map_emission(nt, kind, near, far):
    """Add the normal/depth emission sub-graph to node tree nt and return the Emission node."""
    em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Strength'].default_value = 1.0
    if kind == 'normal':
        geo = nt.nodes.new('ShaderNodeNewGeometry')
        xf = nt.nodes.new('ShaderNodeVectorTransform'); xf.vector_type = 'NORMAL'
        xf.convert_from = 'WORLD'; xf.convert_to = 'CAMERA'
        enc = nt.nodes.new('ShaderNodeVectorMath'); enc.operation = 'MULTIPLY_ADD'
        # Cycles camera space has +z pointing AWAY from the camera (measured on the demo sphere: facing
        # surface gave B~0), so z is negated here to get the OpenGL convention (+z toward camera).
        enc.inputs[1].default_value = (0.5, 0.5, -0.5); enc.inputs[2].default_value = (0.5, 0.5, 0.5)
        nt.links.new(geo.outputs['Normal'], xf.inputs['Vector'])   # shading normal (Cycles flips it toward the ray on backfaces)
        nt.links.new(xf.outputs['Vector'], enc.inputs[0])
        nt.links.new(enc.outputs['Vector'], em.inputs['Color'])
    else:
        cd = nt.nodes.new('ShaderNodeCameraData')
        mr = nt.nodes.new('ShaderNodeMapRange'); mr.clamp = True
        mr.inputs['From Min'].default_value = near; mr.inputs['From Max'].default_value = far
        nt.links.new(cd.outputs['View Z Depth'], mr.inputs['Value'])
        nt.links.new(mr.outputs['Result'], em.inputs['Color'])
    return em

def opacity_source(nt):
    """(socket, value): the material's opacity as a linked socket or a constant, from Principled Alpha."""
    for n in nt.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            inp = n.inputs['Alpha']
            if inp.is_linked: return inp.links[0].from_socket, None
            return None, float(inp.default_value)
    return None, 1.0

def map_material(base, kind, near, far):
    """Copy of `base` (or a new material) whose surface is the map emission, mixed with Transparent BSDF
    by the original opacity so alpha cutouts stay transparent in the map and in the coverage mask."""
    m = base.copy() if base is not None else bpy.data.materials.new(f'Map_{kind}')
    m.name = f'Map_{kind}_{base.name if base else "none"}'; m.use_nodes = True
    nt = m.node_tree
    out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None)
    if out is None: out = nt.nodes.new('ShaderNodeOutputMaterial')
    em = map_emission(nt, kind, near, far)
    src, val = opacity_source(nt)
    if src is None and val >= 1.0:
        shader = em.outputs['Emission']
    else:
        mix = nt.nodes.new('ShaderNodeMixShader'); tr = nt.nodes.new('ShaderNodeBsdfTransparent')
        nt.links.new(tr.outputs['BSDF'], mix.inputs[1]); nt.links.new(em.outputs['Emission'], mix.inputs[2])
        if src is not None: nt.links.new(src, mix.inputs['Fac'])
        else: mix.inputs['Fac'].default_value = val
        shader = mix.outputs['Shader']
    for l in list(out.inputs['Surface'].links): nt.links.remove(l)
    for l in list(out.inputs['Volume'].links): nt.links.remove(l)
    nt.links.new(shader, out.inputs['Surface'])
    return m

def assign_map_materials(meshes, originals, kind, near, far):
    """Replace every material slot with its map version (cached per source material)."""
    cache = {}
    for o in meshes:
        if not o.material_slots: o.data.materials.append(None)
        for i, slot in enumerate(o.material_slots):
            base = originals.get((o.name, i))
            key = base.name if base else None
            if key not in cache: cache[key] = map_material(base, kind, near, far)
            slot.material = cache[key]

def remember_materials(meshes):
    return {(o.name, i): slot.material for o in meshes for i, slot in enumerate(o.material_slots)}

def restore_materials(meshes, originals):
    for o in meshes:
        for i, slot in enumerate(o.material_slots):
            slot.material = originals.get((o.name, i))



def configure_render(scene, res, samples, transparent=True):
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples; scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    try: scene.cycles.device = 'CPU'                     # deterministic, tiny workload
    except Exception: pass
    scene.render.resolution_x, scene.render.resolution_y = res; scene.render.resolution_percentage = 100
    scene.render.film_transparent = transparent
    scene.render.dither_intensity = 0.0
    # 'Raw' = no display transform: 8/16-bit PNG values are the linear shader output (0.5 -> 128 / 32768)
    scene.view_settings.view_transform = 'Raw'; scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0; scene.view_settings.gamma = 1.0
    scene.display_settings.display_device = 'sRGB'
    scene.render.image_settings.file_format = 'PNG'

def render_to(scene, path, color_mode, depth_bits):
    scene.render.image_settings.color_mode = color_mode
    scene.render.image_settings.color_depth = depth_bits
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)

def main():
    a = parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    clear_scene(); scene = bpy.context.scene
    meshes = make_demo(out) if (a.demo or not a.input) else import_garment(a.input)
    if a.smooth: smooth_meshes(meshes)
    if a.relax > 0: relax_meshes(meshes, a.relax)
    scene = bpy.context.scene                      # make_demo() re-creates the scene (OBJ round-trip)
    originals = remember_materials(meshes)
    lo, hi = world_bbox(meshes)
    cam, near, far, ortho = setup_camera(scene, lo, hi, a.res, a.margin, a.view, a.yaw)
    configure_render(scene, a.res, samples=1)
    # world background black so it never leaks into RGB of transparent pixels
    scene.world = bpy.data.worlds.new('W'); scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0, 0, 0, 1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.0

    vl = scene.view_layers[0]
    # --- RGB first (uses the imported materials), optional
    if a.rgb:
        scene.cycles.samples = a.rgb_samples
        scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (1, 1, 1, 1)
        scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 1.0
        scene.view_settings.view_transform = 'Standard'         # real colours for the RGB reference
        render_to(scene, out / f'{a.prefix}_rgb.png', 'RGBA', '8')
        scene.view_settings.view_transform = 'Raw'
        scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.0
        scene.cycles.samples = 1
    # --- normal: per-material map shaders (keep each material's opacity -> alpha cutouts stay holes)
    assign_map_materials(meshes, originals, 'normal', near, far)
    render_to(scene, out / f'{a.prefix}_normal.png', 'RGBA', '8')
    # --- depth: 16-bit grayscale, background must read as FAR -> render opaque onto white
    assign_map_materials(meshes, originals, 'depth', near, far)
    scene.render.film_transparent = False
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (1, 1, 1, 1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 1.0
    render_to(scene, out / f'{a.prefix}_depth.png', 'BW', '16')
    restore_materials(meshes, originals)

    meta = {'input': a.input or 'demo', 'resolution': a.res, 'view': a.view, 'yaw_deg': a.yaw, 'smooth': a.smooth, 'relax_iterations': a.relax,
            'ortho_scale': ortho, 'near': near, 'far': far, 'bbox_min': list(lo), 'bbox_max': list(hi),
            'depth_encoding': 'uint16 = (dist - near) / (far - near) * 65535, background = 65535',
            'normal_encoding': 'camera space OpenGL (R=+x right, G=+y up, B=+z toward camera), n*0.5+0.5',
            'opacity': 'Principled Alpha (value/texture) applied as Transparent mix in normal/depth; bump/normal maps not applied',
            'prepare_flags': '--flip-normal-x --invert-depth --alpha-from normal'}
    (out / f'{a.prefix}_camera.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('DONE', json.dumps(meta))

main()
