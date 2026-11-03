"""Make a local FP16 safetensors copy without loading the 12GB pickle into RAM."""
from pathlib import Path
import gc
import torch
from safetensors.torch import save_file
root=Path(__file__).resolve().parent/'models/IDM-VTON/unet'
target=root/'diffusion_pytorch_model.safetensors'
if not target.exists():
    state=torch.load(root/'diffusion_pytorch_model.bin',map_location='cpu',mmap=True,weights_only=True)
    half={k:v.to(dtype=torch.float16).contiguous() if v.is_floating_point() else v.contiguous() for k,v in state.items()}
    save_file(half,str(target.with_suffix('.tmp')),metadata={'format':'pt','source':'yisol/IDM-VTON','conversion':'float16'})
    target.with_suffix('.tmp').replace(target)
    del half,state
    gc.collect()
print('FP16 safetensors ready',flush=True)
