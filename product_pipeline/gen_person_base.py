"""Reproducible new-person base generator for the shop generative FASHN pipeline.

Replaces the vanished round1.py/round2.py. Talks to diffusers directly instead of
the Forge HTTP API, so it does not depend on Forge being up and does not disturb
the running Stability Matrix instance.

Produces the FASHN-ready base described in the pipeline contract:
  - solid, body-fitting NEUTRAL inner (so FASHN has little old garment to erase)
  - hands clearly separated from the torso (so the arm/torso parse is clean)
  - full or half body framing with the chest area unobstructed
  - a new pose / camera / background / lighting drawn from an approved template

--lora is wired but optional: once the seller identity LoRA is trained, the
same harness generates identity-locked bases with no other change.

Every image is written with a sidecar JSON recording model, LoRA, template, seed,
steps, guidance, resolution and output sha256, so any base can be regenerated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
)
from PIL import Image

DEFAULT_MODEL = Path(
    r"C:\Dev\StabilityMatrix\Data\Models\StableDiffusion\RealVisXL_V5.0_fp16.safetensors"
)
DEFAULT_CONTROLNET = Path(
    r"C:\Dev\StabilityMatrix\Data\Models\ControlNet\thibaud_xl_openpose.safetensors"
)

# Shared across templates: identity-neutral wording so a LoRA token can drive the
# face later without fighting the prompt.
SUBJECT = (
    "editorial fashion photograph of a Korean male fashion model, "
    "short dark hair, clean shaven, neutral calm expression"
)

# With an identity LoRA loaded, the line above actively fights it: measured
# holdout cosine was 0.198-0.339 under this prompt versus 0.366 under a bare one
# (identity_score.py, v1). "fashion model / short dark hair / clean shaven"
# describes an idealized face and overrides the learned one, so when a LoRA is
# active the subject line must stay neutral and let the token carry the face.
IDENTITY_SUBJECT = "editorial fashion photograph of {tok}, neutral calm expression"

# The neutral inner is a pipeline requirement, not a style choice: FASHN leaves
# less residue when the garment it must replace is plain, close-fitting and matte.
#
# "tank" exists because of a measured failure. With a short-sleeve inner, any
# product whose sleeve is shorter leaves an `old \ new` erase band, and the old
# sleeve bleeds back through the composite (see composite_erase_fix.py). A
# sleeveless inner makes that band ~empty: whatever the product's sleeve length,
# the change is purely additive (new \ old), which is garment-over-skin and
# cannot reintroduce old garment.
NEUTRAL_INNERS = {
    "tee": (
        "wearing a plain solid light heather gray short sleeve crew neck t-shirt, "
        "close fitting, no print, no logo, no pattern, matte cotton, "
        "tucked cleanly, plain black straight leg trousers"
    ),
    "tank": (
        "wearing a plain solid light heather gray sleeveless fitted tank top, "
        "narrow shoulder straps, bare arms and bare shoulders fully exposed, "
        "close fitting, no print, no logo, no pattern, matte cotton, "
        "tucked cleanly, plain black straight leg trousers"
    ),
}

QUALITY = (
    "sharp focus, high detail skin texture, natural anatomy, correct hands, "
    "professional color grading, 8k, photorealistic"
)

NEGATIVE = (
    "cropped head, cut off limbs, close-up portrait, sitting, crouching, "
    "crossed arms, arms touching torso, hands in pockets, hands behind back, "
    "hidden hands, holding object, phone, mirror selfie, taking a selfie, "
    "smartphone, reflection in mirror, extra fingers, missing fingers, "
    "fused fingers, extra limbs, extra arms, duplicate person, multiple people, "
    "jacket, coat, cardigan, vest, sweater, hoodie, long sleeves, layered clothing, "
    # The sleeveless inner biases the model toward a summer/athletic look and it
    # renders knee-length shorts instead of the specified trousers; name them.
    "shorts, bermuda shorts, knee length bottoms, cropped trousers, bare legs, "
    "swimwear, athletic shorts, barefoot, "
    # Anything the base wears that crosses the garment boundary gets cut in half by
    # the composite: FASHN removes it under the new garment, the hard composite
    # restores it on bare skin, and the result is a floating fragment. Observed as
    # a broken necklace chain on the neck (matrix_v1 cafe601 x p2022). The LoRA
    # reproduces his necklace faithfully from the training photos, so it has to be
    # suppressed here rather than fixed downstream. A product cut should not carry
    # accessories that are not the product anyway.
    "necklace, chain, pendant, jewelry, earrings, bracelet, watch, scarf, "
    "open shirt, print, graphic, logo, text, watermark, signature, "
    "busy background, cluttered background, illustration, painting, 3d render, cgi, "
    "plastic skin, airbrushed, blurry, low resolution, jpeg artifacts, deformed face"
)


@dataclass(frozen=True)
class Template:
    """An approved pose / camera / scene / lighting combination."""

    template_id: str
    pose: str
    camera: str
    scene: str
    light: str

    def prompt(self, inner: str = "tee", identity_token: str = "") -> str:
        subject = IDENTITY_SUBJECT.format(tok=identity_token) if identity_token else SUBJECT
        return ", ".join(
            [subject, self.pose, NEUTRAL_INNERS[inner], self.camera, self.scene, self.light, QUALITY]
        )


TEMPLATES: dict[str, Template] = {
    t.template_id: t
    for t in [
        Template(
            template_id="studio_front_full",
            pose=(
                "standing straight facing the camera, weight evenly on both feet, "
                "both arms relaxed and held clearly away from the torso, "
                "elbows slightly bent, both hands fully visible with fingers apart"
            ),
            camera=(
                "full body head to toe, eye level, 50mm lens, subject centered, "
                "slight negative space above head"
            ),
            scene="seamless plain light gray studio backdrop, empty floor",
            light=(
                "soft even large softbox lighting from the front, "
                "gentle shadow under the feet, neutral white balance"
            ),
        ),
        Template(
            template_id="studio_threequarter_half",
            pose=(
                "standing in a relaxed three quarter turn to the left, shoulders angled, "
                "head turned back toward the camera, arms hanging away from the body, "
                "both hands visible and separated from the hips"
            ),
            camera="half body from mid thigh up, chest level camera, 85mm lens, shallow depth of field",
            scene="clean warm off white cyclorama wall, no props",
            light="soft key light from the left at 45 degrees, subtle fill on the right, soft falloff",
        ),
        # The shop's own product cuts are cafe / outdoor, not studio. The three
        # templates above deliberately specify empty backdrops to make the FASHN
        # composite easy, which is why every sample came out studio-like. These
        # two carry real scenes. The hard composite protects background pixels
        # exactly, so scene complexity costs nothing at the compositing stage --
        # it only makes the human parser's job harder, which is measured per run.
        Template(
            template_id="cafe_interior_full",
            pose=(
                "standing upright facing the camera, weight on both feet, "
                "arms relaxed and clearly separated from the torso, "
                "both hands open and visible at his sides"
            ),
            camera="full body head to toe, eye level, 35mm lens, subject centered",
            scene=(
                "inside a bright minimal Korean cafe, pale oak floor, white plaster walls, "
                "wooden counter and espresso machine softly out of focus behind him, "
                "a few small potted plants, large window on the left"
            ),
            light=(
                "soft natural window daylight from the left, warm gentle shadows, "
                "clean neutral white balance"
            ),
        ),
        Template(
            template_id="street_daylight_full",
            pose=(
                "standing upright facing the camera, one foot slightly forward, "
                "arms hanging clearly away from the torso, both hands open and visible"
            ),
            camera="full body head to toe, eye level, 50mm lens, subject centered, shallow depth of field",
            scene=(
                "quiet Seoul side street, weathered brick and painted concrete building "
                "facade behind him, narrow pavement, a parked bicycle and a small shop "
                "awning blurred in the background"
            ),
            light="warm late afternoon sunlight, soft directional shadows, gentle haze",
        ),
        Template(
            template_id="daylight_wall_full",
            pose=(
                "standing upright facing the camera, one foot slightly forward, "
                "arms loose and clearly separated from the torso, "
                "both hands open and visible at his sides"
            ),
            camera="full body, low eye level, 35mm lens, subject centered with room above and below",
            scene=(
                "plain smooth concrete wall outdoors, clean empty pavement, "
                "no signage, no furniture"
            ),
            light="bright overcast daylight, soft diffused shadows, cool neutral tone",
        ),
    ]
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _chunk_ids(tokenizer, text: str, chunks_needed: int | None = None) -> list[list[int]]:
    """Split text into CLIP-sized chunks of [BOS] + 75 tokens + [EOS], padded to 77.

    CLIP hard-truncates at 77 tokens. The templates here run past 170, so without
    chunking the camera, scene, lighting and quality clauses are silently dropped
    and only the subject and pose survive.
    """
    ids = tokenizer(text, truncation=False, add_special_tokens=False).input_ids
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    window = tokenizer.model_max_length - 2

    chunks = [ids[i : i + window] for i in range(0, len(ids), window)] or [[]]
    while chunks_needed is not None and len(chunks) < chunks_needed:
        chunks.append([])

    out = []
    for chunk in chunks:
        seq = [bos] + chunk + [eos]
        seq += [pad] * (tokenizer.model_max_length - len(seq))
        out.append(seq)
    return out


def encode_long_prompt(pipe, prompt: str, negative: str, device: str):
    """Encode arbitrarily long SDXL prompts by concatenating per-chunk embeddings.

    Positive and negative are padded to the same chunk count so the CFG batch has
    matching sequence lengths.
    """
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    encoders = [pipe.text_encoder, pipe.text_encoder_2]

    n_chunks = max(
        len(_chunk_ids(tok, text))
        for tok in tokenizers
        for text in (prompt, negative)
    )

    def encode(text: str):
        per_encoder_chunks = [
            _chunk_ids(tok, text, n_chunks) for tok in tokenizers
        ]
        seq_parts, pooled = [], None
        for idx in range(n_chunks):
            hidden = []
            for enc_i, (enc, chunks) in enumerate(zip(encoders, per_encoder_chunks)):
                ids = torch.tensor([chunks[idx]], dtype=torch.long, device=device)
                out = enc(ids, output_hidden_states=True)
                # SDXL conditions on the penultimate hidden layer.
                hidden.append(out.hidden_states[-2])
                # Pooled embedding comes from encoder 2 on the first chunk only.
                if enc_i == 1 and idx == 0:
                    pooled = out[0]
            seq_parts.append(torch.cat(hidden, dim=-1))
        return torch.cat(seq_parts, dim=1), pooled

    prompt_embeds, pooled_embeds = encode(prompt)
    negative_embeds, negative_pooled = encode(negative)
    return prompt_embeds, negative_embeds, pooled_embeds, negative_pooled


def build_pipe(
    model: Path,
    lora: Path | None,
    lora_scale: float,
    offload: bool,
    controlnet: Path | None = None,
):
    if controlnet is not None:
        net = ControlNetModel.from_single_file(str(controlnet), torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetPipeline.from_single_file(
            str(model), controlnet=net, torch_dtype=torch.float16, add_watermarker=False
        )
    else:
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(model), torch_dtype=torch.float16, add_watermarker=False
        )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"
    )
    if lora is not None:
        pipe.load_lora_weights(str(lora))
        pipe.fuse_lora(lora_scale=lora_scale)
    pipe.set_progress_bar_config(disable=False)
    # 12GB card shared with the desktop: slicing keeps peak VRAM well under budget.
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def contact_sheet(paths: list[Path], out_path: Path, thumb: int = 640) -> None:
    imgs = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb, thumb), Image.LANCZOS)
            imgs.append(im.copy())
    if not imgs:
        return
    cell_w = max(i.width for i in imgs)
    cell_h = max(i.height for i in imgs)
    sheet = Image.new("RGB", (cell_w * len(imgs), cell_h), (20, 20, 20))
    for idx, im in enumerate(imgs):
        sheet.paste(im, (idx * cell_w + (cell_w - im.width) // 2, (cell_h - im.height) // 2))
    sheet.save(out_path, quality=93)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--lora", type=Path, default=None, help="identity LoRA (optional, not yet trained)")
    ap.add_argument("--lora-scale", type=float, default=0.8)
    ap.add_argument("--identity-token", default="", help="trigger token prepended to the prompt")
    ap.add_argument(
        "--templates",
        default="studio_front_full,studio_threequarter_half,daylight_wall_full",
    )
    ap.add_argument("--seeds", default="101,102,103")
    ap.add_argument("--inner", default="tee", choices=sorted(NEUTRAL_INNERS))
    ap.add_argument(
        "--control-image",
        type=Path,
        default=None,
        help="OpenPose map from extract_pose_map.py; enables ControlNet posing",
    )
    ap.add_argument("--controlnet", type=Path, default=DEFAULT_CONTROLNET)
    ap.add_argument("--controlnet-scale", type=float, default=0.75)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=1216)
    ap.add_argument("--steps", type=int, default=34)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--offload", action="store_true", help="CPU offload if VRAM is tight")
    args = ap.parse_args()

    template_ids = [t.strip() for t in args.templates.split(",") if t.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) == 1:
        seeds *= len(template_ids)
    if len(seeds) != len(template_ids):
        raise SystemExit("--seeds must be one value or match --templates count")
    unknown = [t for t in template_ids if t not in TEMPLATES]
    if unknown:
        raise SystemExit(f"unknown templates: {unknown}; available: {sorted(TEMPLATES)}")

    use_control = args.control_image is not None
    control_img = None
    if use_control:
        if not args.controlnet.exists():
            raise SystemExit(f"controlnet not found: {args.controlnet}")
        with Image.open(args.control_image) as ci:
            control_img = ci.convert("RGB")
        if control_img.size != (args.width, args.height):
            control_img = control_img.resize((args.width, args.height), Image.LANCZOS)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[load] {args.model.name}" + (f" + {args.controlnet.name}" if use_control else ""))
    pipe = build_pipe(
        args.model,
        args.lora,
        args.lora_scale,
        args.offload,
        controlnet=args.controlnet if use_control else None,
    )

    records, produced = [], []
    for template_id, seed in zip(template_ids, seeds):
        tpl = TEMPLATES[template_id]
        prompt = tpl.prompt(args.inner, args.identity_token)
        print(f"[gen] {template_id} seed={seed}")
        gen = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            # Under CPU offload the encoder module reports device 'cpu' while its
            # accelerate hook moves it to GPU on call, so the ids must be built on
            # the pipeline's execution device, not the module's reported one.
            pos_e, neg_e, pos_p, neg_p = encode_long_prompt(
                pipe, prompt, NEGATIVE, device=pipe._execution_device
            )
        call_kwargs = dict(
            prompt_embeds=pos_e,
            negative_prompt_embeds=neg_e,
            pooled_prompt_embeds=pos_p,
            negative_pooled_prompt_embeds=neg_p,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=gen,
        )
        if use_control:
            call_kwargs["image"] = control_img
            call_kwargs["controlnet_conditioning_scale"] = args.controlnet_scale
        image = pipe(**call_kwargs).images[0]

        stem = f"{template_id}_seed{seed}"
        img_path = args.out_dir / f"{stem}.png"
        image.save(img_path)
        produced.append(img_path)

        rec = {
            "image": str(img_path),
            "sha256": sha256_of(img_path),
            "template": asdict(tpl),
            "prompt": prompt,
            "negative_prompt": NEGATIVE,
            "seed": seed,
            "steps": args.steps,
            "guidance_scale": args.guidance,
            "width": args.width,
            "height": args.height,
            "scheduler": "DPMSolverMultistep karras sde-dpmsolver++",
            "model": str(args.model),
            "lora": str(args.lora) if args.lora else None,
            "lora_scale": args.lora_scale if args.lora else None,
            "identity_token": args.identity_token or None,
            "identity": "seller_identity" if args.lora else "fictional_untrained_person",
            "neutral_inner": args.inner,
            "control_image": str(args.control_image) if use_control else None,
            "controlnet": str(args.controlnet) if use_control else None,
            "controlnet_scale": args.controlnet_scale if use_control else None,
        }
        (args.out_dir / f"{stem}.json").write_text(
            json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        records.append(rec)

    sheet = args.out_dir / "contact_sheet.jpg"
    contact_sheet(produced, sheet)

    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "gen_person_base.py",
        "identity": records[0]["identity"] if records else None,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "contact_sheet": str(sheet),
        "samples": records,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    peak = manifest["peak_vram_bytes"]
    print(f"[done] {len(records)} bases -> {args.out_dir}")
    if peak:
        print(f"[vram] peak allocated {peak / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
