from dotenv import load_dotenv
import torch
from PIL import Image
import os
from diffusers.pipelines.animatediff.pipeline_animatediff import AnimateDiffPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.models.unets.unet_motion_model import MotionAdapter
from diffusers.utils.export_utils import export_to_gif
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel, AutoImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from typing import List, Optional, Union

#############################

load_dotenv()

hf_token: Optional[str] = os.environ.get("HUGGING_FACE_TOKEN")

# Load the Stable Diffusion model components
base_model_id: str = (
    "runwayml/stable-diffusion-v1-5"  # You can try other base models, including pixel art ones
)

# Load VAE
vae_pretrained = AutoencoderKL.from_pretrained(
    base_model_id, subfolder="vae", torch_dtype=torch.float16, use_auth_token=hf_token
)
vae: AutoencoderKL = vae_pretrained[0] if isinstance(vae_pretrained, tuple) else vae_pretrained

# Load Text Encoder and Tokenizer
text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
    base_model_id, subfolder="text_encoder", torch_dtype=torch.float16, use_auth_token=hf_token
)
tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
    base_model_id, subfolder="tokenizer", use_auth_token=hf_token
)

# Load the base UNet
unet_pretrained = UNet2DConditionModel.from_pretrained(
    base_model_id, subfolder="unet", torch_dtype=torch.float16, use_auth_token=hf_token
)
unet: UNet2DConditionModel = unet_pretrained[0] if isinstance(unet_pretrained, tuple) else unet_pretrained


# Load the motion adapter
motion_adapter: MotionAdapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16, use_auth_token=hf_token
)
scheduler = DDIMScheduler.from_pretrained(base_model_id, subfolder="scheduler")

# Load the AnimateDiff pipeline
pipe: AnimateDiffPipeline = AnimateDiffPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    motion_adapter=motion_adapter,
)
# pipe.scheduler = scheduler # Redundant assignment
# pipe.enable_xformers_memory_efficient_attention()
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# ... rest of the script ...


def generate_animation_frames(
    prompts: List[str],
    base_image: Optional[Image.Image] = None,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    num_frames: int = 12,
) -> List[Image.Image]:
    if base_image is not None:
        # Resize the base image to a compatible size if needed
        base_image = base_image.resize((512, 512)).convert("RGB")  # Ensure RGB format

    # AnimateDiff takes a list of prompts and generates multiple frames
    with torch.no_grad():
        output = pipe(
            prompt=prompts,  # Can be a list
            initial_image=base_image,  # Pass the initial image
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_frames=num_frames,  # Specify the number of frames
        )

    frames: List[Image.Image]
    if isinstance(output, tuple):
        frames = []
        if output[0] is not None:
            frames = [
                Image.fromarray(frame.cpu().numpy()) if isinstance(frame, torch.Tensor) else frame
                for frame in output[0]
            ]  # Convert tensors to PIL Images if necessary
        else:
            raise ValueError("Output[0] is None, cannot generate frames.")
    else:
        if isinstance(output.frames, list):
            frames = [
                Image.fromarray(frame.cpu().numpy()) if isinstance(frame, torch.Tensor) else frame
                for sublist in output.frames
                for frame in (sublist if isinstance(sublist, list) else [sublist])
            ]
        elif isinstance(output.frames, torch.Tensor):
            frames = [Image.fromarray(frame.cpu().numpy()) for frame in output.frames]
        else:
            raise TypeError("Unsupported type for output.frames")

    return frames


# Example Usage
script_dir: str = os.path.dirname(os.path.abspath(__file__))
base_image_path: str = os.path.join(script_dir, "cat.png")  # Ensure your base image is here
base_image: Optional[Image.Image] = None
try:
    base_image = Image.open(base_image_path).convert("RGB")
except FileNotFoundError:
    print(f"Error: Base image not found at {base_image_path}")
    exit()

# Define the action name
action_name: str = "eating"

# Array of prompts for the specified action
prompts: List[str] = [
    "A cat, pixel art, looking intently at a small piece of food on the ground, about to lean down to eat",
    "A cat, pixel art, head starting to lower towards the food, mouth slightly open in anticipation",
    "A cat, pixel art, head lowered a bit more, mouth now wider open, food is very close to the mouth",
    "A cat, pixel art, mouth fully open, tongue slightly out, food just entering the mouth",
    # f"A cat, pixel art, mouth closed around the food, showing the action of biting down",
    # f"A cat, pixel art, chewing with its mouth slightly moving, food visible inside the mouth",
    # f"A cat, pixel art, swallowing, throat area slightly bulging, mouth now mostly closed",
    # f"A cat, pixel art, licking its lips with a small piece of food residue visible, looking satisfied",
    # f"A cat, pixel art, looking content, perhaps a slight purr animation",
    # f"A cat, pixel art, subtle idle movement, perhaps blinking slowly",
    # f"A cat, pixel art, subtle idle movement, maybe slightly shifting its weight",
    # f"A cat, pixel art, subtle idle movement, perhaps looking around calmly",
]

# Generate frames for the specified action
output_dir: str = os.path.join(script_dir, "generated_frames_animatediff")  # New output directory
os.makedirs(output_dir, exist_ok=True)

num_frames: int = len(prompts)
generated_frames: List[Image.Image] = generate_animation_frames(
    prompts, base_image=base_image, num_frames=num_frames
)

for i, frame in enumerate(generated_frames):
    frame_number: int = i + 1
    output_filename: str = f"frame_{action_name}_{frame_number:02d}.png"
    output_path: str = os.path.join(output_dir, output_filename)
    frame.save(output_path)
    print(f"Saving frame {frame_number} to: {output_path}")

print(
    f"{action_name.capitalize()} animation frames generated in the 'generated_frames_animatediff' directory using AnimateDiff!"
)
