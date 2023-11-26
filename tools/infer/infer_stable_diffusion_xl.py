import torch
import argparse
from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image

def load_pipeline():
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")

    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")

    return pipeline, pipeline_text2image

def main():
    parser = argparse.ArgumentParser(description="Generate image from a text prompt using StableDiffusionXLPipeline.")
    parser.add_argument("--prompt", type=str, default='Cute white dog walking in the park', help="Text prompt to generate image.")
    parser.add_argument("--save_path", type=str, default='/workspace/data/results/xl_results.jpg', help="Path to save the generated image.")
    args = parser.parse_args()

    pipeline, _ = load_pipeline()
    image = pipeline(prompt=args.prompt).images[0]

    # Assuming image is a PIL Image. If not, you may need to convert it to a PIL Image first.
    image.save(args.save_path)

if __name__ == "__main__":
    main()