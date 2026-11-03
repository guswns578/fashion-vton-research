# Synthetic Cloth Ablation V1

Date: 2026-08-14

## Question

Can a commercially usable local physics stack produce stable, measurable geometry changes from material parameters before real body and garment data are available?

## Setup

- Solver: Blender 4.5.10 LTS cloth.
- Body: procedural ellipsoid collision mesh.
- Garment: 1,920-vertex open tube with a narrowed top and six pinned shoulder vertices.
- Simulation: 80 frames, fixed gravity, collision, and self-collision.
- Controlled variables: identical initial body, garment topology, pose, frame count, camera, and resolution.
- Changed variables: mass, tension, compression, shear, bending, and air damping presets.
- Render: fixed orthographic camera at 480x720, Raw color transform, no look transform.

This is a mechanics and pipeline test. The body, garment, and material values are not calibrated to a real person or fabric.

## Results

| Preset | Mean displacement | P90 displacement | Width/image | Height/image | Area/image |
|---|---:|---:|---:|---:|---:|
| soft | 0.1110 m | 0.1564 m | 0.7896 | 0.6528 | 0.4218 |
| medium | 0.1001 m | 0.1424 m | 0.8375 | 0.6472 | 0.4492 |
| stiff | 0.0898 m | 0.1388 m | 0.8667 | 0.6431 | 0.4618 |

Pairwise fixed-camera comparisons:

| Pair | Silhouette IoU | Normal MAE on overlap |
|---|---:|---:|
| soft vs medium | 0.9205 | 0.0337 |
| soft vs stiff | 0.8832 | 0.0415 |
| medium vs stiff | 0.9587 | 0.0415 |

All three render contracts passed: files existed, dimensions matched, silhouette was nonblank, and depth/normal varied inside the garment region.

## Interpretation

The material controls are active rather than decorative. Softer settings produced larger displacement and a narrower, longer projected shape; stiffer settings retained more width. The fixed-camera normal maps also changed.

This does not show that Blender predicts a real garment's drape. The current object has no sleeves, sewing pattern, measured body, calibrated camera, or measured material parameters. The result only proves that the selected commercial-compatible stack can carry a material signal through simulation, mesh export, and geometry-map rendering.

## Decision

Proceed to the one-person/one-garment G-C1 pilot. Do not train geometry conditioning from these synthetic outputs and do not label the presets as real fabric classes.

The next calibration dataset needs:

- A measured body mesh.
- A real long-sleeve garment with flat measurements and a reconstructed pattern or shell.
- Fixed-camera reference and worn-ground-truth captures.
- At least three plausible material settings evaluated against the hidden real silhouette and landmarks.

Select material parameters by held-out geometry agreement, not by visual preference alone.

## Artifacts

- Report: `fashion_pipeline/out/cloth_ablation_v3/report.json`
- Visual sheet: `fashion_pipeline/out/cloth_ablation_v3/qc_sheet.jpg`
- Per-preset simulation metadata, OBJ meshes, silhouette, depth, normal, and render validation are stored under the same directory.
