from PIL import Image
import torchvision.transforms as transforms

def remove_alpha_channel(tensor):
    return tensor[:, :3, :, :]  # RGBA에서 RGB로 변경

class MSELossScheduler:
    def __init__(self, total_steps,  weight, mode='constant'):
        self.total_steps = total_steps
        self.weight = weight
        self.current_step = 0
        self.mode = mode
        assert self.mode in ['constant', 'linear'], "Mode should be either 'constant' or 'linear'"

    def step(self):
        """
        Call this function whenever you take an optimization step to update the current step.
        """
        self.current_step += 1

    def get_loss_weight(self):
        """
        Returns the current MSE loss weight based on the current step and mode.
        """
        if self.mode == 'constant':
            return self.weight
        elif self.mode == 'linear':
            return 1.0 - (self.current_step / self.total_steps)

def subtract_noise(scheduler, noisy_samples, noise, timesteps):
    # Make sure alphas_cumprod and timestep have same device and dtype as noisy_samples
    alphas_cumprod = scheduler.alphas_cumprod.to(device=noisy_samples.device, dtype=noisy_samples.dtype)
    timesteps = timesteps.to(noisy_samples.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(noisy_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(noisy_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    # Reconstruct the original samples by reversing the noise addition process
    original_samples = (noisy_samples - sqrt_one_minus_alpha_prod * noise) / sqrt_alpha_prod
    return original_samples

def save_decoder_output_to_image(decoder_output, save_path):
    # decoder_output.sample는 [1, C, H, W] 형태의 텐서를 가정합니다.
    # 여기에서 첫 번째 차원(배치 차원)을 제거합니다.
    image_tensor = decoder_output.sample.squeeze(0)

    # 텐서가 여전히 4차원이면, 코드가 올바르지 않게 작성되었음을 의미합니다.
    if image_tensor.ndim == 4:
        raise ValueError("Image tensor should be 3-dimensional after squeezing the batch dimension.")

    # 이미지 텐서가 [C, H, W] 형태임을 확실히 하고 [-1, 1] 범위로 정규화된 것으로 가정합니다.
    # 텐서를 [0, 255] 범위로 변환합니다.
    image_tensor = image_tensor.clamp(min=-1, max=1)
    image_tensor = (image_tensor + 1) / 2 * 255  # Scale to [0, 255]
    image_tensor = image_tensor.byte()

    # 텐서를 PIL 이미지로 변환합니다.
    image = transforms.ToPILImage()(image_tensor)

    # 이미지를 디스크에 저장합니다.
    image.save(save_path)

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
def get_timesteps(infer_scheduler, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = infer_scheduler.timesteps[t_start * infer_scheduler.order :]

    return timesteps, num_inference_steps - t_start