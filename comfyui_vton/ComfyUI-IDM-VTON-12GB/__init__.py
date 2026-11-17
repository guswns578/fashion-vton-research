"""ComfyUI bridge to an isolated low-memory IDM-VTON subprocess."""

import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import folder_paths
import comfy.model_management as mm

CONFIG = json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))


def save_tensor(tensor, path, mask=False):
    if tensor.shape[0] != 1:
        raise ValueError("IDM-VTON 12GB supports batch size 1 only.")
    arr = (tensor[0].detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def image_tensor(path, mask=False):
    img = Image.open(path).convert("L" if mask else "RGB")
    return torch.from_numpy(np.array(img).astype(np.float32) / 255)[None]


def worker(mode, job, directory):
    mm.unload_all_models()
    mm.soft_empty_cache()
    task = Path(directory) / "job.json"
    task.write_text(json.dumps(job), encoding="utf-8")
    log = Path(directory) / "worker.log"
    # Poll so ComfyUI's cancel button also stops the isolated worker.
    with log.open("w", encoding="utf-8") as stream:
        p = subprocess.Popen(
            [CONFIG["python"], "-u", CONFIG["runner"], mode, str(task)],
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            while True:
                mm.throw_exception_if_processing_interrupted()
                try:
                    code = p.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            p.terminate()
            p.wait()
            raise
    if code:
        raise RuntimeError(
            "IDM-VTON worker failed:\n" + log.read_text(encoding="utf-8", errors="replace")[-8000:]
        )
    print(log.read_text(encoding="utf-8", errors="replace")[-2500:])


class IDMPreprocess12GB:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "human": ("IMAGE",),
                "clothing_prompt": ("STRING", {"default": "shirt"}),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 0.95, "step": 0.05}),
                "mask_grow": ("INT", {"default": 12, "min": 0, "max": 40}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("densepose", "clothing_mask")
    FUNCTION = "run"
    CATEGORY = "IDM-VTON 12GB"

    def run(self, human, clothing_prompt, threshold, mask_grow):
        with tempfile.TemporaryDirectory(
            prefix="idm_pre_", dir=folder_paths.get_temp_directory()
        ) as d:
            paths = {k: str(Path(d) / (k + ".png")) for k in ["human", "pose", "mask"]}
            save_tensor(human, paths["human"])
            worker(
                "preprocess",
                dict(
                    paths,
                    prompt=clothing_prompt,
                    threshold=threshold,
                    grow=mask_grow,
                    comfy=CONFIG["comfy"],
                ),
                d,
            )
            return image_tensor(paths["pose"]), image_tensor(paths["mask"], True)


class IDMInference12GB:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "human": ("IMAGE",),
                "garment": ("IMAGE",),
                "densepose": ("IMAGE",),
                "clothing_mask": ("MASK",),
                "garment_description": ("STRING", {"default": "a shirt", "multiline": True}),
                "width": ("INT", {"default": 384, "min": 256, "max": 768, "step": 64}),
                "height": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
                "steps": ("INT", {"default": 20, "min": 2, "max": 60}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "IDM-VTON 12GB"

    def run(
        self,
        human,
        garment,
        densepose,
        clothing_mask,
        garment_description,
        width,
        height,
        steps,
        seed,
    ):
        with tempfile.TemporaryDirectory(
            prefix="idm_infer_", dir=folder_paths.get_temp_directory()
        ) as d:
            paths = {
                k: str(Path(d) / (k + ".png"))
                for k in ["human", "garment", "pose", "mask", "output"]
            }
            for key, tensor in [
                ("human", human),
                ("garment", garment),
                ("pose", densepose),
                ("mask", clothing_mask),
            ]:
                save_tensor(tensor, paths[key])
            worker(
                "infer",
                dict(
                    paths,
                    description=garment_description,
                    width=width,
                    height=height,
                    steps=steps,
                    seed=seed,
                ),
                d,
            )
            return (image_tensor(paths["output"]),)


NODE_CLASS_MAPPINGS = {"IDMPreprocess12GB": IDMPreprocess12GB, "IDMInference12GB": IDMInference12GB}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IDMPreprocess12GB": "IDM: GroundingDINO + SAM + DensePose (12GB)",
    "IDMInference12GB": "IDM-VTON: CPU Offload (12GB)",
}
