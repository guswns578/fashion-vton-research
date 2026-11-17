"""Generate a synthetic person and prepare the G-B generated-person VTON gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL = Path(r"C:\Dev\StabilityMatrix\Data\Models\StableDiffusion\RealVisXL_V5.0_fp16.safetensors")
BENCHMARK = Path(r"fashion_pipeline/out/startup_benchmark_v1/manifest.json")
OUT = Path(r"fashion_pipeline/out/generated_person_gate")
PROMPT = (
    "full body editorial photograph of a fictional Korean male fashion model, standing straight facing camera, "
    "arms relaxed slightly away from torso, both hands fully visible, wearing a plain fitted light gray short sleeve "
    "crew neck t-shirt and simple black straight trousers, neutral expression, plain light gray studio backdrop, "
    "soft even studio lighting, realistic skin, realistic anatomy, 50mm lens, head to toe visible"
)
NEGATIVE = (
    "cropped, close-up, sitting, crouching, crossed arms, hands in pockets, hidden hands, extra fingers, missing fingers, "
    "extra limbs, duplicate person, multiple people, jacket, coat, vest, sweater, long sleeves, logo, text, watermark, "
    "busy background, illustration, painting, CGI"
)


def command_generate(args: argparse.Namespace) -> None:
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

    args.out.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(args.model),
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        width=768,
        height=1024,
        num_inference_steps=args.steps,
        guidance_scale=5.0,
        generator=generator,
    ).images[0]
    path = args.out / "person.png"
    image.save(path)
    metadata = {
        "model": str(args.model.resolve()),
        "seed": args.seed,
        "steps": args.steps,
        "width": image.width,
        "height": image.height,
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE,
        "identity": "fictional_untrained_person",
    }
    (args.out / "generation.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(path)


def command_build(args: argparse.Namespace) -> None:
    person = (args.out / "person.png").resolve()
    if not person.exists():
        raise FileNotFoundError(f"Generate the person first: {person}")
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    garment_sample = next(row for row in benchmark["samples"] if row["sku_id"] == args.sku)
    garment = Path(garment_sample["garment"]).resolve()
    sample_id = f"synthetic_person_{args.sku}"
    manifest = {
        "schema_version": 1,
        "dataset": "generated_person_gate_v1",
        "protocol": "operational_gate_without_counterfactual_ground_truth",
        "warning": "This gate tests pipeline compatibility, not garment fit accuracy.",
        "counts": {"products": 1, "observations": 1},
        "samples": [
            {
                "sample_id": sample_id,
                "product_id": args.sku,
                "person": str(person),
                "garment": str(garment),
                "ground_truth": {
                    "image": str(person),
                    "mask": str(
                        (args.out / "masks" / "ground_truth" / f"{sample_id}.png").resolve()
                    ),
                },
                "variants": {
                    "fashn": {
                        "raw": str(
                            (args.out / "variants" / "fashn_raw" / f"{sample_id}.png").resolve()
                        ),
                        "image": str(
                            (args.out / "variants" / "fashn" / f"{sample_id}.png").resolve()
                        ),
                        "mask": str((args.out / "masks" / "fashn" / f"{sample_id}.png").resolve()),
                        "change_mask": str(
                            (args.out / "masks" / "fashn_change" / f"{sample_id}.png").resolve()
                        ),
                    }
                },
            }
        ],
    }
    path = args.out / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    sub = ap.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--model", type=Path, default=MODEL)
    generate.add_argument("--seed", type=int, default=20260814)
    generate.add_argument("--steps", type=int, default=30)
    generate.set_defaults(func=command_generate)
    build = sub.add_parser("build")
    build.add_argument("--benchmark", type=Path, default=BENCHMARK)
    build.add_argument("--sku", default="p2726")
    build.set_defaults(func=command_build)
    return ap


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
