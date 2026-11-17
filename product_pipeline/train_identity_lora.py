"""Train the seller identity LoRA on SDXL, sized to fit a 12GB card.

Route decision (see the 2026-08-18 handoff discussion): this LoRA carries BODY,
proportions, hair and general look from the 40 body crops. It is deliberately NOT
asked to carry fine facial detail -- only 5 of the 40 training images can reach a
1024px face crop without exceeding 2x upsampling (median 2.93x), so pixel-level
face training here would mostly teach LANCZOS blur. Face identity is injected
separately at inference via a face-embedding adapter, which operates at ~112px
and is therefore well matched to this data.

Fitting 12GB:
  - VAE latents and text embeddings are cached in a first pass, then the VAE and
    both text encoders are freed. Only the UNet is resident during training.
  - LoRA is applied to UNet attention projections only; the base UNet stays frozen.
  - Gradient checkpointing on.
  - bf16 autocast with fp32 master weights for the LoRA params.

Run in the forge-neo venv (it has diffusers + peft + accelerate):
    C:\\Dev\\StabilityMatrix\\Data\\Packages\\forge-neo\\venv\\Scripts\\python.exe train_identity_lora.py ...
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig, get_peft_model_state_dict
from PIL import Image

DEFAULT_MODEL = Path(
    r"C:\Dev\StabilityMatrix\Data\Models\StableDiffusion\RealVisXL_V5.0_fp16.safetensors"
)
TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]


def list_pairs(roots: list[Path]) -> list[tuple[Path, str]]:
    """Collect image/caption pairs from one or more directories.

    v2 trains on body crops and face crops together: the body crops carry
    proportion and overall look, the face crops carry the facial detail that a
    130px-in-1152 body crop cannot.
    """
    pairs = []
    for root in roots:
        for img in sorted(root.glob("*.png")):
            cap = img.with_suffix(".txt")
            if cap.exists():
                pairs.append((img, cap.read_text(encoding="utf-8").strip()))
    return pairs


@torch.no_grad()
def cache_conditioning(pipe, pairs, device, dtype):
    """Encode images to latents and captions to embeddings once, up front."""
    vae: AutoencoderKL = pipe.vae
    vae.to(device, dtype=torch.float32)  # fp16 VAE encode is numerically fragile
    cached = []

    for img_path, caption in pairs:
        with Image.open(img_path) as im:
            rgb = im.convert("RGB")
        arr = torch.from_numpy(np.asarray(rgb, dtype=np.float32).copy())
        px = (arr / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device, torch.float32)
        latent = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor

        # SDXL conditions on both encoders; pooled comes from encoder 2.
        embeds, pooled = [], None
        for tok, enc in (
            (pipe.tokenizer, pipe.text_encoder),
            (pipe.tokenizer_2, pipe.text_encoder_2),
        ):
            ids = tok(
                caption,
                padding="max_length",
                max_length=tok.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            out = enc(ids, output_hidden_states=True)
            embeds.append(out.hidden_states[-2])
            pooled = out[0]
        cached.append(
            {
                "latent": latent.squeeze(0).to(dtype).cpu(),
                "prompt_embeds": torch.cat(embeds, dim=-1).squeeze(0).to(dtype).cpu(),
                "pooled": pooled.squeeze(0).to(dtype).cpu(),
                "size": (rgb.height, rgb.width),
                "path": str(img_path),
            }
        )
    vae.to("cpu")
    return cached


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-dir",
        type=Path,
        nargs="+",
        default=[
            Path(r"fashion_pipeline/dataset/seller_identity_trainset_v2/train/body"),
            Path(r"fashion_pipeline/dataset/seller_identity_trainset_v2/train/face"),
        ],
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument(
        "--no-grad-ckpt",
        action="store_true",
        help="disable gradient checkpointing; faster when VRAM is not the constraint",
    )
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    pairs = list_pairs(args.train_dir)
    if not pairs:
        raise SystemExit(f"no image/caption pairs in {args.train_dir}")
    print(f"[data] {len(pairs)} image/caption pairs")

    print(f"[load] {args.model.name}")
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(args.model), torch_dtype=torch.float32, add_watermarker=False
    )
    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)

    print("[cache] encoding latents + text embeddings")
    cached = cache_conditioning(pipe, pairs, device, dtype)

    # Everything except the UNet can go now.
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()

    unet = pipe.unet
    unet.requires_grad_(False)
    unet.to(device, dtype=dtype)
    unet.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            init_lora_weights="gaussian",
            target_modules=TARGET_MODULES,
        )
    )
    if not args.no_grad_ckpt:
        unet.enable_gradient_checkpointing()

    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.to(torch.float32)
    print(
        f"[lora] rank={args.rank} trainable tensors={len(params)} "
        f"params={sum(p.numel() for p in params):,}"
    )

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    noise_sched = DDPMScheduler.from_config(pipe.scheduler.config)

    buckets: dict[tuple, list] = {}
    for item in cached:
        buckets.setdefault(item["latent"].shape, []).append(item)
    print("[buckets] " + ", ".join(f"{tuple(k)}x{len(v)}" for k, v in buckets.items()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    unet.train()

    for step in range(1, args.steps + 1):
        # v2 mixes 896x1152 body crops with 512x512 face crops, so a batch must be
        # drawn from a single resolution bucket -- stacking mismatched latents fails.
        anchor = cached[random.randrange(len(cached))]
        bucket = buckets[anchor["latent"].shape]
        items = [bucket[random.randrange(len(bucket))] for _ in range(args.batch_size)]
        latent = torch.stack([i["latent"] for i in items]).to(device, dtype)
        prompt_embeds = torch.stack([i["prompt_embeds"] for i in items]).to(device, dtype)
        pooled = torch.stack([i["pooled"] for i in items]).to(device, dtype)
        bsz = latent.shape[0]

        noise = torch.randn_like(latent)
        t = torch.randint(0, noise_sched.config.num_train_timesteps, (bsz,), device=device).long()
        noisy = noise_sched.add_noise(latent, noise, t)

        # SDXL micro-conditioning: original size, crop offset, target size.
        add_time_ids = torch.tensor(
            [[i["size"][0], i["size"][1], 0, 0, i["size"][0], i["size"][1]] for i in items],
            device=device,
            dtype=dtype,
        )

        pred = unet(
            noisy,
            t,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids},
        ).sample

        target = (
            noise
            if noise_sched.config.prediction_type == "epsilon"
            else noise_sched.get_velocity(latent, noise, t)
        )
        loss = F.mse_loss(pred.float(), target.float())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        if step % 25 == 0 or step == 1:
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(
                f"[{step:5d}/{args.steps}] loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.2e} peak={peak:.2f}GiB"
            )
            history.append({"step": step, "loss": float(loss.item())})

        if step % args.save_every == 0 or step == args.steps:
            # peft keys must be renamed to diffusers convention, otherwise
            # pipe.load_lora_weights silently matches nothing at inference.
            state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=str(args.out_dir),
                unet_lora_layers=state,
                weight_name=f"seller_identity_r{args.rank}_s{step}.safetensors",
                safe_serialization=True,
            )
            print(f"[save] step {step}")

    manifest = {
        "trained_utc": datetime.now(timezone.utc).isoformat(),
        "identity": "seller_identity",
        "trigger_token": "hgnst_model",
        "base_model": str(args.model),
        "train_dir": [str(p) for p in args.train_dir],
        "images": len(pairs),
        "rank": args.rank,
        "alpha": args.alpha,
        "lr": args.lr,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "gradient_checkpointing": not args.no_grad_ckpt,
        "seed": args.seed,
        "target_modules": TARGET_MODULES,
        "scope": "UNet attention only; body/proportion identity. Face detail is NOT "
        "learned here -- see face-embedding adapter route.",
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "history": history,
    }
    (args.out_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] -> {args.out_dir}  peak {manifest['peak_vram_gib']}GiB")


if __name__ == "__main__":
    main()
