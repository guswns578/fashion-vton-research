# Geometry Pilot V1

## Decision

The current VTON pipeline is operational, but its largest commercial failure is garment geometry. Resolution changes and the high-frequency refiner do not recover sleeve structure, length, or physical drape. Seed selection showed measurable headroom on the p2428 failure case (`+0.2225` mask IoU), but the best candidate still had severe boundary and garment-identity errors.

The next experiment must therefore test explicit geometry before more texture refinement.

## Scope

Use one consenting subject (the researcher), one physically available long-sleeve top, one fixed pose, and one fixed camera. Do not start with a catalog-only garment because the pilot needs a real worn image as ground truth.

The first target is not photorealistic cloth. It is a correctly aligned garment silhouette and coarse drape surface that can be compared with a real photograph.

## Collection Order

1. Record the consent fields in `geometry_pilot.example.json`.
2. Measure the body three times and store the median value for each field.
3. Capture or fit a metric-scale body mesh in an A-pose.
4. Measure the garment flat without stretching it. Capture front, back, label, and at least three fabric closeups.
5. Photograph the subject in the reference pose without the target garment, then wearing the target garment without moving the camera.
6. Include a scale marker or checkerboard in one calibration frame.
7. Reconstruct or author the garment mesh and simulate only the fixed pose.
8. Render aligned silhouette, depth, normal, UV/warp, and landmark outputs.

## Required Measurements

Body: height, weight, neck, shoulder breadth, chest, waist, hip, arm length, upper-arm circumference, wrist circumference, and back-waist length.

Top: shoulder, flat chest, body length, sleeve length, flat upper arm, flat cuff, flat hem, and armhole. Also record size label and fiber composition.

Material: weigh the complete dry garment when possible and record `garment_mass_g`. Begin with measured or coarse values for mass per area, thickness, warp stretch, weft stretch, bending class, and surface class. Unknown values stay `null`; they must not be guessed into ground truth. If GSM is unavailable, the pilot may derive an effective GSM from whole-garment mass and generated panel area.

The Blender pilot records physical garment mass separately from its numerical vertex-mass scale. The numerical scale is a calibration parameter and must not be reported as measured kilograms.

## Gate Sequence

### G-C0: Capture Alignment

- Same camera, crop, and pose for the empty-person and worn-ground-truth frames.
- Body landmarks align within 2% of image height.
- Manual garment mask is visually clean at sleeves, collar, hem, and occlusions.

### G-C1: Geometry Fidelity

Compare the 3D garment render with the real worn mask and landmarks.

- Primary: garment mask IoU.
- Secondary: shoulder, cuff, side seam, and hem landmark error normalized by torso height.
- Diagnostic: garment height/width ratios and hem position.

Pass for the first pilot: IoU at least 0.80 and median landmark error at most 5% of torso height. These are engineering thresholds, not claims of human-equivalent fit.

### G-C2: Selector Value

Generate at least five FASHN candidates with fixed settings and different seeds. Rank them using the 3D silhouette:

```powershell
python seed_selector_gate.py --out fashion_pipeline/out/seed_selector_gate report `
  --target-mask fashion_pipeline/out/geometry_pilot_v1/renders/garment_silhouette.png `
  --report-name report_3d.json
```

Pass: the selected candidate beats fixed seed 42 on the hidden real-worn mask, and visual review confirms that the gain is not caused by a malformed but overlapping region.

### G-C3: Conditioning Value

Only after G-C1 passes, train or adapt a geometry-conditioning path. Compare:

- FASHN baseline.
- FASHN plus candidate selection from the 3D silhouette.
- Geometry-conditioned model using silhouette only.
- Geometry-conditioned model using silhouette plus depth/normal.

This stage needs training because FASHN was not trained to interpret arbitrary depth or normal maps through its pose channel.

## Stop Rules

- If G-C1 fails, improve body/garment alignment or simulation; do not train a diffusion adapter yet.
- If G-C1 passes but G-C2 has no hidden-ground-truth gain, 3D is not useful as a selector in the current form.
- If silhouette conditioning improves shape but damages garment identity, keep geometry and product-detail losses separate.
- Texture and reflectance work begins only after sleeve, hem, length, and volume are stable.

## Immediate Deliverable

Populate `pilot.json` with measured body and garment values, one measured garment mass or GSM value, and explicit body-mesh unit/alignment metadata. Add the watertight A-pose body mesh, flat garment front/back images, calibration frame, empty-person reference, and same-camera worn ground truth. The external-body mechanics path is now validated; these physical inputs are the remaining dependency for real G-C1 evidence.

## Commands After Capture

Prepare pose-alignment and automatic garment-mask QC. The automatic mask is not accepted into the pilot unless `--accept-auto-mask` is supplied after visual review.

```powershell
& C:\Dev\fashn-vton-1.5\.venv\Scripts\python.exe prepare_geometry_capture.py
```

Check which inputs are still missing:

```powershell
python validate_geometry_pilot.py
python run_real_gc1.py --root fashion_pipeline/out/geometry_pilot_v1 --dry-run
```

When the dry-run reports no blockers, run the complete staged gate:

```powershell
python run_real_gc1.py --root fashion_pipeline/out/geometry_pilot_v1
```

The runner generates the calibrated GarmentCode pattern, welded low-strain mesh contract, no-gravity control, gravity simulation, fixed-camera renders, and mechanics report. It accepts only a validated gravity render into `pilot.json`; a reviewed real worn mask is still required before silhouette evaluation, and landmarks remain mandatory for a full G-C1 pass.

After a 3D tool has exported the aligned silhouette, depth, normal, and landmarks, run the strict G-C1 evaluation:

```powershell
python geometry_eval.py
```

The evaluator reports mask IoU, boundary F1, symmetric boundary distance, garment height/width/hem errors, and normalized landmark errors. It returns `incomplete` rather than a false pass when landmark files are absent.

For OBJ/FBX/GLB/PLY body and garment meshes, the installed Blender 4.5 LTS adapter can create and validate the geometry maps. Add `--accept` only after inspecting the outputs:

```powershell
python run_geometry_render.py --body path\to\body.glb --garment path\to\garment.obj
python run_geometry_render.py --body path\to\body.glb --garment path\to\garment.obj --accept
```

The adapter uses an external Blender process and writes camera metadata with every render. The current smoke test is:

```powershell
python run_geometry_render.py --smoke-test --out fashion_pipeline/out/geometry_render_smoke
```
