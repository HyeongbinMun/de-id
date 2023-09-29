import argparse
import torch
import piqa
from PIL import Image
from torchvision import transforms


def compute_psnr(img1_path, img2_path):
    x, y = load_images(img1_path, img2_path)
    psnr = piqa.PSNR()
    l = psnr(x, y)
    return l.item()

def compute_ssim(img1_path, img2_path):
    x, y = load_images(img1_path, img2_path)
    x = x.requires_grad_(True).cuda()
    y = y.cuda()

    ssim = piqa.SSIM().cuda()
    l = 1 - ssim(x, y)
    l.backward()
    return l.item()

def compute_fid(img1_path, img2_path):
    x, y = load_images(img1_path, img2_path)
    x = x.requires_grad_(True).cuda()
    y = y.cuda()

    ssim = piqa.SSIM().cuda()
    l = 1 - ssim(x, y)
    l.backward()
    return l.item()

def load_images(img1_path, img2_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    img_x = Image.open(img1_path).convert("RGB")
    img_y = Image.open(img2_path).convert("RGB")

    x = transform(img_x).unsqueeze(0)
    y = transform(img_y).unsqueeze(0)

    return x, y

def main():
    parser = argparse.ArgumentParser(description="Evaluation to generative model")
    parser.add_argument("--function", type=str, required=True, default='all', help="Function to select")
    parser.add_argument("--img1_path", type=str, required=True, default='/workspace/images/face.jpg', help="Path to first image")
    parser.add_argument("--img2_path", type=str, required=True, default='/workspace/images/face.jpg', help="Path to second image")

    args = parser.parse_args()

    if args.function == 'all':
        psnr_result = compute_psnr(args.img1_path, args.img2_path)
        ssim_result = compute_ssim(args.img1_path, args.img2_path)
        print(f"PSNR: {psnr_result}")
        print(f"SSIM: {ssim_result}")
    else:
        if args.function == "psnr":
            result = compute_psnr(args.img1_path, args.img2_path)
            print(f"PSNR: {result}")
        elif args.function == "ssim":
            result = compute_ssim(args.img1_path, args.img2_path)
            print(f"SSIM: {result}")
        else:
            print('Please provide a evaluation function name')

if __name__ == "__main__":
    main()
