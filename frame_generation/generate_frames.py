from dotenv import load_dotenv
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
import torch
from PIL import Image
import os

load_dotenv()

hf_token = os.environ.get("HUGGING_FACE_TOKEN")

# Load the Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"  # or another suitable model
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_auth_token=hf_token)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")


def generate_image(prompt, image_path, base_image=None):
    image = pipe(prompt, image=base_image).images[0]
    image.save(image_path)


# Example Usage
script_dir = os.path.dirname(os.path.abspath(__file__))
base_image_path = os.path.join(script_dir, "cat.png")  # Ensure your base image is here
base_image = Image.open(base_image_path)

# Define the action name
action_name = "eating"

# Array of prompts for the specified action
prompts = [
    f"A cat, pixel art, looking intently at a small piece of food on the ground, about to lean down to eat, frame 1, based on: previous frame",
    f"A cat, pixel art, head starting to lower towards the food, mouth slightly open in anticipation, frame 2, based on: previous frame",
    f"A cat, pixel art, head lowered, mouth now wider open, food is very close to the mouth, frame 3, based on: previous frame",
    f"A cat, pixel art, mouth fully open, tongue slightly out, food just entering the mouth, frame 4, based on: previous frame",
    f"A cat, pixel art, mouth closed around the food, showing the action of biting down, frame 5, based on: previous frame",
    f"A cat, pixel art, chewing with its mouth slightly moving, food visible inside the mouth, frame 6, based on: previous frame",
    f"A cat, pixel art, swallowing, throat area slightly bulging, mouth now mostly closed, frame 7, based on: previous frame",
    f"A cat, pixel art, licking its lips with a small piece of food residue visible, looking satisfied, frame 8, based on: previous frame",
    f"A cat, pixel art, looking content, perhaps a slight purr animation (optional to add to prompt later), frame 9, based on: previous frame",
    f"A cat, pixel art, subtle idle movement, perhaps blinking slowly, frame 10, based on: previous frame",
    f"A cat, pixel art, subtle idle movement, maybe slightly shifting its weight, frame 11, based on: previous frame",
    f"A cat, pixel art, subtle idle movement, perhaps looking around calmly, frame 12, based on: previous frame",
]

# Generate frames for the specified action
output_dir = os.path.join(script_dir, "generated_frames")
os.makedirs(output_dir, exist_ok=True)

previous_image = base_image  # Initialize previous_image with the base image

for i, prompt in enumerate(prompts):
    frame_number = i + 1
    output_filename = f"frame_{action_name}_{frame_number:02d}.png"
    output_path = os.path.join(output_dir, output_filename)
    print(f"Generating frame {frame_number} for {action_name}: {output_path}")

    generate_image(prompt, output_path, base_image=previous_image)  # Always use previous_image

    previous_image = Image.open(output_path)  # Set the current image for the next loop.

print(f"{action_name.capitalize()} animation frames generated in the 'generated_frames' directory!")
