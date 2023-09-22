import argparse
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import torch

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Stable Diffusion Inpainting.")
    parser.add_argument("--model", type=str, default="/data/face/models/0920_widerface_nomask", help="stable diffusion model.")
    parser.add_argument("--prompt", type=str, default="a photo of taylor hill face")
    parser.add_argument("--image", type=str, default="/workspace/test/multi.jpg", help="Path to the source image.")
    parser.add_argument("--mask_image", type=str, default="/workspace/test/multi_masked.jpg", help="Path to the mask image.")
    parser.add_argument("--save", type=str, default="/workspace/test/output.jpg", help="Path to the save image.")

    args = parser.parse_args()

    # Load the pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model,
        revision="fp16",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")

    # Load images
    image = Image.open(args.image)
    mask_image = Image.open(args.mask_image)

    # Process
    result = pipe(prompt=args.prompt, image=image, mask_image=mask_image).images[0]
    result.save(args.save)
