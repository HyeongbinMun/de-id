import torch.nn.functional as F


def ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    mu1 = F.avg_pool2d(img1, 3, 1)
    mu2 = F.avg_pool2d(img2, 3, 1)
    mu1_mu2 = mu1*mu2
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.avg_pool2d(img1*img1, 3, 1) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2*img2, 3, 1) - mu2_sq
    sigma12 = F.avg_pool2d(img1*img2, 3, 1) - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().clamp(0, 1)