# 3D Stack License Decision

Date checked: 2026-08-14

This is an engineering record, not legal advice. Recheck licenses before product launch or redistribution.

## Approved for the Pilot

### Blender 4.5 LTS

- Role: external headless renderer and first-pass cloth simulator.
- License: Blender application is GPL. Blender's official manual states that the GPL applies to Blender and not to artwork created with it.
- Integration rule: invoke the installed Blender executable as an external process. Keep our manifests, training code, and service separate from Blender distribution concerns.
- Sources:
  - https://developer.blender.org/docs/license
  - https://docs.blender.org/manual/en/3.2/getting_started/about/license.html

### MakeHuman or a Researcher-Owned Scan

- Role: initial body mesh before a calibrated commercial body-reconstruction service is selected.
- License: MakeHuman states that official exports are CC0 and can be used commercially.
- Integration rule: retain the export provenance and do not assume that third-party community assets share the core CC0 license.
- Source: https://static.makehumancommunity.org/makehuman/faq/can_i_sell_models_created_with_makehuman.html

### GarmentCode Core Pattern Library

- Role: parametric sewing-pattern representation and pattern prototyping.
- License: MIT at the official repository.
- Integration rule: use the pattern library independently from its historical simulation fork.
- Source: https://github.com/maria-korosteleva/GarmentCode

## Excluded from a Commercial Runtime

### NvidiaWarp-GarmentCode Fork

- The GarmentCode documentation relies on a custom Warp fork for its full XPBD simulation.
- That fork explicitly states that its license permits only non-commercial use.
- It may be used only in a separately documented non-commercial research comparison, not as the startup product backend or as training-data infrastructure for the commercial model.
- Source: https://github.com/maria-korosteleva/NvidiaWarp-GarmentCode

### SMPL-X Model and SMPLify-X

- The standard model/software license is for non-commercial research and requires a separate commercial license.
- Do not build the startup's body representation or commercial training data around it without a signed commercial agreement.
- Source: https://smpl-x.is.tue.mpg.de/modellicense.html

## Future Option

Current NVIDIA Warp is Apache-2.0, but the old GarmentCode cloth modifications cannot simply be copied from the non-commercial fork. A commercial Warp solver would require an independent implementation based on permissively licensed current Warp and public algorithms, with a clean provenance record.

Source: https://github.com/NVIDIA/warp
