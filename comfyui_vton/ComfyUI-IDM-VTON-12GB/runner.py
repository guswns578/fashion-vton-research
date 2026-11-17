"""Isolated IDM-VTON worker. All GPU models die when this process exits."""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
os.environ["HF_HOME"] = str(WORK / "hf-cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(WORK / "ComfyUI-IDM-VTON"))
sys.path.insert(0, str(WORK / "comfyui_segment_anything"))
import numpy as np
import torch
from PIL import Image, ImageFilter


def release():
    gc.collect()
    torch.cuda.empty_cache()


def preprocess(job):
    from local_groundingdino.util.slconfig import SLConfig
    from local_groundingdino.util.utils import clean_state_dict
    from local_groundingdino.models import build_model
    from local_groundingdino.datasets import transforms as T

    image = Image.open(job["human"]).convert("RGB")
    cfg = SLConfig.fromfile(str(WORK / "models/grounding-dino/GroundingDINO_SwinT_OGC.cfg.py"))
    cfg.text_encoder_type = str(WORK / "models/bert-base-uncased")
    dino = build_model(cfg)
    weights = torch.load(
        WORK / "models/grounding-dino/groundingdino_swint_ogc.pth",
        map_location="cpu",
        weights_only=False,
    )
    dino.load_state_dict(clean_state_dict(weights["model"]), strict=False)
    del weights
    dino.eval().to("cuda")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tensor, _ = transform(image, None)
    with torch.inference_mode():
        pred = dino(
            tensor[None].to("cuda"),
            captions=[job.get("prompt", "shirt").strip().lower().rstrip(".") + "."],
        )
    keep = pred["pred_logits"].sigmoid()[0].amax(-1) > job.get("threshold", 0.3)
    boxes = pred["pred_boxes"][0][keep].detach().cpu()
    if len(boxes) == 0:
        raise RuntimeError("No clothing detected. Try shirt / jacket / dress or lower threshold.")
    w, h = image.size
    boxes *= torch.tensor([w, h, w, h])
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    del dino, pred, tensor
    release()
    from segment_anything import sam_model_registry, SamPredictor

    sam = (
        sam_model_registry["vit_b"](checkpoint=str(WORK / "models/sams/sam_vit_b_01ec64.pth"))
        .eval()
        .to("cuda")
    )
    predictor = SamPredictor(sam)
    predictor.set_image(np.array(image))
    with torch.inference_mode():
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=predictor.transform.apply_boxes_torch(boxes, (h, w)).to("cuda"),
            multimask_output=False,
        )
    mask = Image.fromarray((masks.any(dim=0)[0].cpu().numpy() * 255).astype("uint8"))
    grow = int(job.get("grow", 12))
    if grow:
        mask = mask.filter(ImageFilter.MaxFilter(2 * grow + 1))
    mask.save(job["mask"])
    del sam, predictor, masks
    release()
    aux = Path(job["comfy"]) / "custom_nodes/comfyui_controlnet_aux/src"
    sys.path.insert(0, str(aux))
    import torchvision
    from custom_controlnet_aux.densepose import DenseposeDetector

    detector = DenseposeDetector(
        torch.jit.load(
            str(WORK / "models/densepose/densepose_r50_fpn_dl.torchscript"), map_location="cpu"
        )
    ).to("cuda")
    with torch.inference_mode():
        pose = detector(image, detect_resolution=512, cmap="parula")
    pose.resize(image.size).save(job["pose"])
    print("Preprocessing complete", flush=True)


def infer(job):
    overall_start = time.time()
    from diffusers import AutoencoderKL, DDPMScheduler
    from transformers import (
        AutoTokenizer,
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
        CLIPTextModel,
        CLIPTextModelWithProjection,
    )
    from src.idm_vton.unet_hacked_tryon import UNet2DConditionModel
    from src.idm_vton.unet_hacked_garmnet import UNet2DConditionModel as GarmentUNet
    from src.idm_vton.tryon_pipeline import StableDiffusionXLInpaintPipeline
    from accelerate import cpu_offload
    from torchvision import transforms

    weights = str(WORK / "models/IDM-VTON")

    def load(cls, subfolder):
        print("Loading " + subfolder, flush=True)
        return (
            cls.from_pretrained(
                weights,
                subfolder=subfolder,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            .eval()
            .requires_grad_(False)
        )

    pipe = StableDiffusionXLInpaintPipeline(
        vae=load(AutoencoderKL, "vae"),
        text_encoder=load(CLIPTextModel, "text_encoder"),
        text_encoder_2=load(CLIPTextModelWithProjection, "text_encoder_2"),
        tokenizer=AutoTokenizer.from_pretrained(
            weights, subfolder="tokenizer", local_files_only=True
        ),
        tokenizer_2=AutoTokenizer.from_pretrained(
            weights, subfolder="tokenizer_2", local_files_only=True
        ),
        unet=load(UNet2DConditionModel, "unet"),
        image_encoder=load(CLIPVisionModelWithProjection, "image_encoder"),
        feature_extractor=CLIPImageProcessor(),
        scheduler=DDPMScheduler.from_pretrained(
            weights, subfolder="scheduler", local_files_only=True
        ),
    )
    pipe.unet_encoder = load(GarmentUNet, "unet_encoder")
    # Sequential offload covers submodules, including the separate garment UNet.
    # The upstream pipeline does not register unet_encoder as a component.
    # VAE changes dtype internally; keep this small module resident so its
    # upcasting does not conflict with Accelerate's saved CPU weight copies.
    pipe._exclude_from_cpu_offload = ["vae"]
    pipe.enable_sequential_cpu_offload(device="cuda")
    cpu_offload(pipe.unet_encoder, execution_device=torch.device("cuda:0"))
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    w, h = job.get("width", 384), job.get("height", 512)
    human = Image.open(job["human"]).convert("RGB").resize((w, h))
    garment = Image.open(job["garment"]).convert("RGB").resize((w, h))
    pose = Image.open(job["pose"]).convert("RGB").resize((w, h))
    mask = Image.open(job["mask"]).convert("L").resize((w, h), Image.Resampling.NEAREST)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    desc = job.get("description", "a shirt")
    neg = "low quality, blurry, bad anatomy"
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        embeds, negative, pooled, negative_pooled = pipe.encode_prompt(
            "model is wearing " + desc,
            device=torch.device("cuda:0"),
            do_classifier_free_guidance=True,
            negative_prompt=neg,
        )
        cloth_embeds, _, _, _ = pipe.encode_prompt(
            "a photo of " + desc, device=torch.device("cuda:0"), do_classifier_free_guidance=False
        )
        start = time.time()
        result = pipe(
            prompt_embeds=embeds,
            negative_prompt_embeds=negative,
            pooled_prompt_embeds=pooled,
            negative_pooled_prompt_embeds=negative_pooled,
            num_inference_steps=job.get("steps", 20),
            generator=torch.Generator("cuda").manual_seed(job.get("seed", 42)),
            strength=1.0,
            pose_img=transform(pose)[None].to("cuda", torch.float16),
            text_embeds_cloth=cloth_embeds,
            cloth=transform(garment)[None].to("cuda", torch.float16),
            mask_image=mask,
            image=human,
            height=h,
            width=w,
            ip_adapter_image=garment,
            guidance_scale=2.0,
        )[0][0]
        result.save(job["output"])
        metrics = {
            "seconds": time.time() - start,
            "total_seconds": time.time() - overall_start,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
            "width": w,
            "height": h,
            "steps": job.get("steps", 20),
        }
        Path(job["output"]).with_suffix(".json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preprocess", "infer"])
    parser.add_argument("job")
    args = parser.parse_args()
    torch.set_num_threads(4)
    job = json.loads(Path(args.job).read_text(encoding="utf-8-sig"))
    (preprocess if args.mode == "preprocess" else infer)(job)
