# pip install diffusers peft transformers
from dotenv import load_dotenv
import torch
import os
from diffusers.pipelines.animatediff.pipeline_animatediff import AnimateDiffPipeline
from diffusers.models.unets.unet_motion_model import MotionAdapter
from diffusers.schedulers.scheduling_lcm import LCMScheduler
from diffusers.utils.export_utils import export_to_gif
from typing import List, Optional, Union

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
load_dotenv()
script_dir: str = os.path.dirname(os.path.abspath(__file__))
hf_token: Optional[str] = os.environ.get("HUGGING_FACE_TOKEN")

adapter = MotionAdapter.from_pretrained(
    "wangfuyun/AnimateLCM", torch_dtype=torch.float16, use_auth_token=hf_token
)
pipe = AnimateDiffPipeline.from_pretrained(
    "emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16, use_auth_token=hf_token
)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

pipe.load_lora_weights(
    "wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora"
)
pipe.set_adapters(["lcm-lora"], [0.8])

pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()
generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(0)
output = pipe(
    prompt="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution",
    negative_prompt="bad quality, worse quality, low resolution",
    num_frames=16,
    guidance_scale=3.0,
    num_inference_steps=6,
    generator=generator,
)  # type: ignore
frames = output.frames[0]
output_filepath = os.path.join(script_dir, "animatelcm2.gif")
export_to_gif(frames, output_filepath)
