import os
import argparse
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import torch

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Stable Diffusion Inpainting.")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-inpainting",
                        help="stable diffusion model.")
    parser.add_argument("--prompt", type=str, default="a photo of taylor hill face")
    parser.add_argument("--image_dir", type=str, default="/workspace/data/test/images",
                        help="Path to the source image directory.")
    parser.add_argument("--mask_image_dir", type=str, default="/workspace/data/test/masks",
                        help="Path to the mask image directory.")
    parser.add_argument("--save_dir", type=str, default="/workspace/data/image",
                        help="Path to the save image directory.")
    args = parser.parse_args()

    # Load the pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model,
        revision="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to("cuda")

    image_files = [f for f in os.listdir(args.image_dir) if os.path.isfile(os.path.join(args.image_dir, f))]
    for image_file in image_files:
        # Assuming the mask images have the same filenames in their respective directory
        image_path = os.path.join(args.image_dir, image_file)
        mask_image_path = os.path.join(args.mask_image_dir, image_file)

        # Load images
        image = Image.open(image_path)
        mask_image = Image.open(mask_image_path)

        # Process
        result = pipe(prompt=args.prompt, image=image, mask_image=mask_image).images[0]
        save_path = os.path.join(args.save_dir, image_file)
        result.save(save_path)

