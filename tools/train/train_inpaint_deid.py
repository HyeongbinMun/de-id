"""
Face De-Identification via Inpainting + LoRA
=============================================
SD Inpainting UNet(9ch)에 LoRA를 적용하여 얼굴 영역만 재생성.
배경은 원본 보존 → black-box collapse 구조적 방지.

Loss 구성:
  1. L_diff   = MSE(noise_pred, target)              → 인페인팅 품질
  2. L_id     = max(cos_sim(face) - margin, 0)       → 얼굴 신원 변환
  3. L_feat   = 1 - cos_sim(full_img)                → CBIR 특징 보존
  4. L_lpips  = LPIPS(face_orig, face_gen)            → 지각적 품질

  L_total = λ_diff * L_diff
          + warmup * α_t * (λ_id*L_id + λ_feat*L_feat + λ_lpips*L_lpips)

핵심: 인페인팅이므로 마스크 밖 영역은 원본 그대로 → feature 유사도 자연 보존.
"""

import argparse
import datetime
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import lpips
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

import wandb
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import numpy as np

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model.icd.sscd.sscd.models.model import Model as SSCDResNet
from model.vcd.vcd.models.frame import MobileNet_AVG


# =============================================================================
#  LoRA Implementation (train_new_deid.py에서 재사용)
# =============================================================================

class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_feat = original.in_features
        out_feat = original.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, in_feat))
        self.lora_B = nn.Parameter(torch.zeros(out_feat, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        original.weight.requires_grad_(False)
        if original.bias is not None:
            original.bias.requires_grad_(False)

    def forward(self, x, *args, **kwargs):
        base = self.original(x)
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base + lora

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    @property
    def in_features(self):
        return self.original.in_features

    @property
    def out_features(self):
        return self.original.out_features


def inject_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    target_suffixes: Optional[List[str]] = None,
) -> Dict[str, LoRALinear]:
    if target_suffixes is None:
        target_suffixes = ["to_q", "to_k", "to_v", "to_out.0"]

    named = dict(model.named_modules())
    injected: Dict[str, LoRALinear] = {}

    for full_name, module in list(named.items()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(full_name.endswith(s) for s in target_suffixes):
            continue

        lora = LoRALinear(module, rank=rank, alpha=alpha)

        parts = full_name.rsplit(".", 1)
        if len(parts) == 2:
            parent = named[parts[0]]
            attr = parts[1]
        else:
            parent = model
            attr = parts[0]

        if isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent[int(attr)] = lora
        else:
            setattr(parent, attr, lora)

        injected[full_name] = lora

    return injected


def collect_lora_params(model: nn.Module) -> List[nn.Parameter]:
    params = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            params.extend([m.lora_A, m.lora_B])
    return params


def save_lora_state_dict(model: nn.Module) -> dict:
    sd = {}
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            sd[f"{name}.lora_A"] = m.lora_A.data.clone()
            sd[f"{name}.lora_B"] = m.lora_B.data.clone()
            sd[f"{name}.rank"] = torch.tensor(m.rank)
            sd[f"{name}.alpha"] = torch.tensor(m.alpha)
    return sd


# =============================================================================
#  Dataset
# =============================================================================

class InpaintDeIdDataset(Dataset):
    """인페인팅 비식별화 데이터셋.

    images/ 와 masks/ 디렉토리에서 동일 파일명으로 매칭.
    labels/ 는 선택적으로 YOLO bbox를 읽어 얼굴 crop 좌표 제공.
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, data_root: str, split: str = "train", size: int = 512):
        self.data_root = Path(data_root)
        self.size = size

        img_dir = self.data_root / "images" / split
        mask_dir = self.data_root / "masks" / split
        label_dir = self.data_root / "labels" / split

        if not img_dir.exists():
            raise ValueError(f"Image directory not found: {img_dir}")
        if not mask_dir.exists():
            raise ValueError(f"Mask directory not found: {mask_dir}")

        img_files = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in self.EXTENSIONS and p.is_file()
        )

        self.samples = []
        for img_path in img_files:
            stem = img_path.stem
            mask_candidates = [mask_dir / f"{stem}{ext}" for ext in self.EXTENSIONS]
            mask_path = next((p for p in mask_candidates if p.exists()), None)
            if mask_path is None:
                continue

            label_path = label_dir / f"{stem}.txt" if label_dir.exists() else None
            if label_path is not None and not label_path.exists():
                label_path = None

            self.samples.append((img_path, mask_path, label_path))

        if not self.samples:
            raise ValueError(f"No image-mask pairs found in {img_dir}")

        self.img_transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
        ])

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _parse_yolo_label(label_path: Path, img_w: int, img_h: int):
        """YOLO label → (x1, y1, x2, y2) pixel coordinates (first box)."""
        with open(label_path, "r") as f:
            line = f.readline().strip()
        if not line:
            return None
        parts = line.split()
        if len(parts) < 5:
            return None
        _, cx, cy, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

    def __getitem__(self, idx):
        img_path, mask_path, label_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.img_transform(image)
        mask = self.img_transform(mask)

        bbox = None
        if label_path is not None:
            bbox = self._parse_yolo_label(label_path, self.size, self.size)

        img_np = np.array(image).astype(np.float32)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1) / 127.5 - 1.0  # [-1, 1]

        mask_np = np.array(mask).astype(np.float32) / 255.0
        mask_np[mask_np < 0.5] = 0.0
        mask_np[mask_np >= 0.5] = 1.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # (1, H, W)

        masked_image = img_tensor * (mask_tensor < 0.5)  # face region zeroed

        result = {
            "pixel_values": img_tensor,
            "mask": mask_tensor,
            "masked_image": masked_image,
        }
        if bbox is not None:
            result["bbox"] = torch.tensor(bbox, dtype=torch.long)

        return result


# =============================================================================
#  Face Identity Extractor
# =============================================================================

class FaceIdentityExtractor(nn.Module):
    def __init__(
        self,
        model_path: Optional[str] = None,
        backbone: str = "resnet50",
        embedding_dim: int = 512,
    ):
        super().__init__()
        self.input_size = 160

        import torchvision.models as tv_models

        if backbone == "resnet50":
            weights = tv_models.ResNet50_Weights.DEFAULT if model_path is None else None
            resnet = tv_models.resnet50(weights=weights)
            feat_dim = 2048
        elif backbone == "resnet18":
            weights = tv_models.ResNet18_Weights.DEFAULT if model_path is None else None
            resnet = tv_models.resnet18(weights=weights)
            feat_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(feat_dim, embedding_dim)

        if model_path and os.path.exists(model_path):
            sd = torch.load(model_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            cleaned = {}
            for k, v in sd.items():
                cleaned[k.replace("module.", "").replace("model.", "")] = v
            self.load_state_dict(cleaned, strict=False)

        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.input_size, self.input_size),
                          mode="bilinear", align_corners=False)
        x = self._normalize(x)
        feat = self.features(x).flatten(1)
        emb = self.projection(feat)
        return F.normalize(emb, p=2, dim=1)


# =============================================================================
#  Lightning Module
# =============================================================================

class FaceInpaintDeIdLitModule(pl.LightningModule):
    """인페인팅 기반 얼굴 비식별화 LoRA 학습 모듈.

    SD Inpainting UNet (9ch input: noisy_latents + mask + masked_latents)에
    LoRA를 주입하여, 얼굴 영역만 재생성하면서 신원을 변환한다.

    배경은 원본 그대로 유지되므로 black-box collapse가 구조적으로 불가능하다.
    """

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(vars(args) if isinstance(args, argparse.Namespace) else args)
        self.args = args

        # ---- SD Inpainting 컴포넌트 ----
        ckpt = getattr(args, "single_file_ckpt", None)
        if ckpt:
            from diffusers import StableDiffusionInpaintPipeline
            print(f"[Model] Loading from single file: {ckpt}")
            pipe = StableDiffusionInpaintPipeline.from_single_file(
                ckpt, torch_dtype=torch.float32,
            )
            self.tokenizer = pipe.tokenizer
            self.text_encoder = pipe.text_encoder
            self.vae = pipe.vae
            self.unet = pipe.unet
            self.noise_scheduler = pipe.scheduler
            del pipe
            self.noise_scheduler = DDPMScheduler.from_config(self.noise_scheduler.config)
            self.infer_scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="tokenizer"
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder"
            )
            self.vae = AutoencoderKL.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="vae"
            )
            self.unet = UNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="unet"
            )
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="scheduler"
            )
            self.infer_scheduler = DDIMScheduler.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="scheduler"
            )

        # 외부 VAE 오버라이드 (Realistic Vision 등은 sd-vae-ft-mse 권장)
        vae_path = getattr(args, "vae_path", None)
        if vae_path:
            print(f"[VAE] Loading external VAE: {vae_path}")
            self.vae = AutoencoderKL.from_pretrained(vae_path)

        print(f"[Model] UNet in_channels={self.unet.config.in_channels}")

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        # ---- LoRA 주입 ----
        self.lora_layers = inject_lora(
            self.unet, rank=args.lora_rank, alpha=args.lora_alpha,
        )
        self.lora_params = collect_lora_params(self.unet)

        n_lora = sum(p.numel() for p in self.lora_params)
        n_unet = sum(p.numel() for p in self.unet.parameters())
        print(f"[LoRA] Trainable: {n_lora:,} / Total UNet: {n_unet:,} "
              f"({100 * n_lora / n_unet:.2f}%)")

        # ---- Face Identity 모델 ----
        self.face_id_model = FaceIdentityExtractor(
            model_path=getattr(args, "face_id_model_path", None),
            backbone=getattr(args, "face_id_backbone", "resnet50"),
            embedding_dim=getattr(args, "face_id_embedding_dim", 512),
        )
        self.face_id_model.eval()
        self.face_id_model.requires_grad_(False)

        # ---- Feature 모델 (SSCD) ----
        self.feature_model = self._build_feature_model(args)
        self.feature_model.eval()
        self.feature_model.requires_grad_(False)

        # ---- LPIPS ----
        self.lpips_model = lpips.LPIPS(net="vgg")
        self.lpips_model.eval()
        self.lpips_model.requires_grad_(False)

        # ---- Loss 유틸리티 ----
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

        # ---- 프롬프트 임베딩 캐싱 ----
        prompt = getattr(args, "prompt", "a photo of a face")
        self.register_buffer("prompt_embeds", self._encode_prompt(prompt), persistent=False)

        # ---- Loss 가중치 ----
        self.lambda_diff = args.lambda_diffusion
        self.lambda_id = args.lambda_identity
        self.lambda_feat = args.lambda_feature
        self.lambda_lpips = args.lambda_lpips

        # ---- Anti-collapse ----
        self.id_margin = args.id_margin
        self.timestep_threshold = args.timestep_threshold
        self.aux_warmup_steps = args.aux_warmup_steps

        # ---- Gradient checkpointing ----
        if getattr(args, "gradient_checkpointing", False):
            self.unet.enable_gradient_checkpointing()
            self.vae.enable_gradient_checkpointing()

        self.num_sample_inference_steps = getattr(args, "num_sample_inference_steps", 30)

    # -----------------------------------------------------------------
    #  Helpers
    # -----------------------------------------------------------------

    def _build_feature_model(self, args) -> nn.Module:
        if args.feature_model_name == "ResNet50":
            model = SSCDResNet(backbone="TV_RESNET50", dims=512, pool_param=3)
            model.load_state_dict(
                torch.load(args.feature_model_path, map_location="cpu")
            )
        else:
            model = MobileNet_AVG()
            sd = torch.load(args.feature_model_path, map_location="cpu")
            sd = {k.replace("base.", ""): v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
        return model

    @torch.no_grad()
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return self.text_encoder(tokens.input_ids)[0]

    def _predict_x0(self, noisy_latents, noise_pred, timesteps):
        alpha_cumprod = self.noise_scheduler.alphas_cumprod.to(
            device=noisy_latents.device, dtype=noisy_latents.dtype
        )
        alpha_t = alpha_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_alpha = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_t)
        return (noisy_latents - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha

    def _composite(self, pred_img, orig_img, mask):
        """생성된 얼굴을 원본 배경과 합성. mask=1인 영역은 생성, 나머지는 원본."""
        if mask.shape[2:] != pred_img.shape[2:]:
            mask = F.interpolate(mask, size=pred_img.shape[2:], mode="nearest")
        return pred_img * mask + orig_img * (1.0 - mask)

    def _crop_face(self, img_01, bbox):
        """bbox (x1,y1,x2,y2)로 얼굴 영역 crop. bbox 없으면 전체 이미지 반환."""
        if bbox is None:
            return img_01
        x1, y1, x2, y2 = bbox
        h, w = x2 - x1, y2 - y1
        if h < 8 or w < 8:
            return img_01
        return img_01[:, :, y1:y2, x1:x2]

    def _aux_warmup_factor(self) -> float:
        if self.aux_warmup_steps <= 0:
            return 1.0
        return min(1.0, self.global_step / self.aux_warmup_steps)

    # -----------------------------------------------------------------
    #  Training Step
    # -----------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]       # (B, 3, H, W) [-1, 1]
        masks = batch["mask"]                       # (B, 1, H, W) {0, 1}
        masked_images = batch["masked_image"]       # (B, 3, H, W) [-1, 1]
        bsz = pixel_values.shape[0]

        encoder_hidden_states = self.prompt_embeds.expand(bsz, -1, -1)

        # 1) VAE encode
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

            masked_latents = self.vae.encode(masked_images).latent_dist.sample()
            masked_latents = masked_latents * self.vae.config.scaling_factor

        # Mask → latent 해상도
        latent_mask = F.interpolate(
            masks, size=latents.shape[2:], mode="nearest"
        )

        # 2) Forward diffusion
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=self.device, dtype=torch.long,
        )
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # 3) 9ch input: [noisy_latents, mask, masked_latents]
        latent_model_input = torch.cat([noisy_latents, latent_mask, masked_latents], dim=1)

        # 4) UNet noise prediction
        noise_pred = self.unet(latent_model_input, timesteps, encoder_hidden_states).sample

        # ===== Diffusion Loss (항상 적용) =====
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type: {self.noise_scheduler.config.prediction_type}")

        diffusion_loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

        # ===== Auxiliary Losses (t < threshold인 샘플만) =====
        warmup = self._aux_warmup_factor()
        valid_mask = (timesteps < self.timestep_threshold)
        n_valid = valid_mask.sum().item()
        zero = torch.tensor(0.0, device=self.device)

        if n_valid > 0 and warmup > 0:
            pred_x0_latents = self._predict_x0(noisy_latents, noise_pred, timesteps)
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]

            id_cosines = []
            feat_cosines = []
            lpips_vals = []

            for idx in valid_indices:
                i = idx.item()
                lat_i = pred_x0_latents[i : i + 1]
                pred_img_i = self.vae.decode(
                    lat_i / self.vae.config.scaling_factor
                ).sample.clamp(-1, 1)

                # Composite: 생성 얼굴 + 원본 배경
                comp_i = self._composite(pred_img_i, pixel_values[i : i + 1], masks[i : i + 1])

                # [0, 1] 변환
                comp_01 = (comp_i + 1.0) * 0.5
                orig_01 = (pixel_values[i : i + 1] + 1.0) * 0.5

                # Identity: 얼굴 영역 crop으로 비교
                bbox = batch["bbox"][i].tolist() if "bbox" in batch else None
                face_orig = self._crop_face(orig_01, bbox)
                face_gen = self._crop_face(comp_01, bbox)

                with torch.no_grad():
                    orig_id = self.face_id_model(face_orig)
                pred_id = self.face_id_model(face_gen)
                id_cosines.append(self.cos_sim(orig_id, pred_id))

                # Feature: 전체 composite 이미지 비교
                with torch.no_grad():
                    orig_feat = self.feature_model(orig_01)
                pred_feat = self.feature_model(comp_01)
                feat_cosines.append(self.cos_sim(orig_feat, pred_feat))

                # LPIPS: 얼굴 영역 비교 ([-1,1] 범위)
                face_orig_11 = self._crop_face(pixel_values[i : i + 1], bbox)
                face_gen_11 = self._crop_face(comp_i, bbox)
                if face_orig_11.shape[2] >= 16 and face_orig_11.shape[3] >= 16:
                    lpips_vals.append(self.lpips_model(face_orig_11, face_gen_11).squeeze())
                else:
                    lpips_vals.append(zero)

            id_cosine = torch.cat(id_cosines)
            feat_cosine = torch.cat(feat_cosines)

            identity_loss = torch.clamp(id_cosine - self.id_margin, min=0).mean()
            feature_loss = (1.0 - feat_cosine).mean()
            lpips_loss = torch.stack(lpips_vals).mean()

            alpha_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
            timestep_weight = alpha_cumprod[timesteps[valid_mask]].mean()
        else:
            id_cosine = zero
            feat_cosine = zero
            identity_loss = zero
            feature_loss = zero
            lpips_loss = zero
            timestep_weight = zero

        # ===== Total Loss =====
        aux_scale = warmup * timestep_weight
        total_loss = (
            self.lambda_diff * diffusion_loss
            + aux_scale * (
                self.lambda_id * identity_loss
                + self.lambda_feat * feature_loss
                + self.lambda_lpips * lpips_loss
            )
        )

        # ===== Logging =====
        self.log("train/loss_total", total_loss, prog_bar=True, sync_dist=True)
        self.log("train/loss_diffusion", diffusion_loss, prog_bar=True, sync_dist=True)
        self.log("train/loss_identity", identity_loss, prog_bar=True, sync_dist=True)
        self.log("train/loss_feature", feature_loss, sync_dist=True)
        self.log("train/loss_lpips", lpips_loss, sync_dist=True)

        id_val = id_cosine.mean() if id_cosine.dim() > 0 else id_cosine
        feat_val = feat_cosine.mean() if feat_cosine.dim() > 0 else feat_cosine
        self.log("train/id_cosine_sim", id_val, sync_dist=True)
        self.log("train/feat_cosine_sim", feat_val, sync_dist=True)
        self.log("train/aux_warmup", warmup, sync_dist=True)
        self.log("train/aux_scale", aux_scale, sync_dist=True)
        self.log("train/n_valid", float(n_valid), sync_dist=True)

        opt = self.optimizers()
        self.log("train/lr", opt.param_groups[0]["lr"], sync_dist=True)

        if (self.global_step + 1) % self.args.log_every_n_steps == 0:
            total_norm = 0.0
            for p in self.lora_params:
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            self.log("train/grad_norm", math.sqrt(total_norm), sync_dist=True)

        if (self.global_step + 1) % self.args.sample_every_n_steps == 0:
            self._save_samples(pixel_values[:4], masks[:4], masked_images[:4])

        return total_loss

    # -----------------------------------------------------------------
    #  Validation Step
    # -----------------------------------------------------------------

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        masks = batch["mask"]
        masked_images = batch["masked_image"]
        bsz = pixel_values.shape[0]

        encoder_hidden_states = self.prompt_embeds.expand(bsz, -1, -1)

        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        masked_latents = self.vae.encode(masked_images).latent_dist.sample()
        masked_latents = masked_latents * self.vae.config.scaling_factor

        latent_mask = F.interpolate(masks, size=latents.shape[2:], mode="nearest")

        # low timestep으로 x0 품질 평가
        timesteps = torch.randint(
            0, self.timestep_threshold,
            (bsz,), device=self.device, dtype=torch.long,
        )
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        latent_model_input = torch.cat([noisy_latents, latent_mask, masked_latents], dim=1)
        noise_pred = self.unet(latent_model_input, timesteps, encoder_hidden_states).sample

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type: {self.noise_scheduler.config.prediction_type}")

        diffusion_loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

        pred_x0_latents = self._predict_x0(noisy_latents, noise_pred, timesteps)

        id_cosines = []
        feat_cosines = []
        lpips_vals = []
        comp_for_log = []

        for i in range(bsz):
            pred_img_i = self.vae.decode(
                pred_x0_latents[i : i + 1] / self.vae.config.scaling_factor
            ).sample.clamp(-1, 1)

            comp_i = self._composite(pred_img_i, pixel_values[i : i + 1], masks[i : i + 1])
            comp_01 = (comp_i + 1.0) * 0.5
            orig_01 = (pixel_values[i : i + 1] + 1.0) * 0.5

            bbox = batch["bbox"][i].tolist() if "bbox" in batch else None

            id_cosines.append(self.cos_sim(
                self.face_id_model(self._crop_face(orig_01, bbox)),
                self.face_id_model(self._crop_face(comp_01, bbox)),
            ))
            feat_cosines.append(self.cos_sim(
                self.feature_model(orig_01),
                self.feature_model(comp_01),
            ))
            lpips_vals.append(
                self.lpips_model(pixel_values[i : i + 1], comp_i).squeeze()
            )

            if batch_idx == 0 and i < 4:
                comp_for_log.append(comp_01)

        id_cosine = torch.cat(id_cosines)
        feat_cosine = torch.cat(feat_cosines)

        identity_loss = torch.clamp(id_cosine - self.id_margin, min=0).mean()
        feature_loss = (1.0 - feat_cosine).mean()
        lpips_loss = torch.stack(lpips_vals).mean()

        alpha_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        timestep_weight = alpha_cumprod[timesteps].mean()

        total_loss = (
            self.lambda_diff * diffusion_loss
            + timestep_weight * (
                self.lambda_id * identity_loss
                + self.lambda_feat * feature_loss
                + self.lambda_lpips * lpips_loss
            )
        )

        self.log("val/loss_total", total_loss, prog_bar=True, sync_dist=True)
        self.log("val/loss_diffusion", diffusion_loss, sync_dist=True)
        self.log("val/loss_identity", identity_loss, sync_dist=True)
        self.log("val/loss_feature", feature_loss, sync_dist=True)
        self.log("val/loss_lpips", lpips_loss, sync_dist=True)
        self.log("val/id_cosine_sim", id_cosine.mean(), prog_bar=True, sync_dist=True)
        self.log("val/feat_cosine_sim", feat_cosine.mean(), prog_bar=True, sync_dist=True)

        if batch_idx == 0 and self.global_rank == 0 and comp_for_log:
            nn = len(comp_for_log)
            orig_01 = (pixel_values[:nn] + 1.0) * 0.5
            masked_01 = (masked_images[:nn] + 1.0) * 0.5
            masks_01 = masks[:nn].float()
            self._log_wandb_images(
                orig_01, torch.cat(comp_for_log),
                masks_01=masks_01, masked_01=masked_01,
                prefix="val/x0_pred",
                caption_suffix="(x0 prediction)",
            )

        return total_loss

    # -----------------------------------------------------------------
    #  Sample & Image logging
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _log_wandb_images(self, orig_01, gen_01, masks_01=None, masked_01=None,
                          prefix="train", caption_suffix=""):
        """WandB에 이미지 로깅: 개별 이미지 + 비교 그리드."""
        if self.logger is None or not isinstance(self.logger, WandbLogger):
            return

        n = min(4, orig_01.shape[0])
        step = self.global_step
        log_dict = {"global_step": step}

        rows = [orig_01[:n]]
        row_labels = ["Original"]

        if masks_01 is not None:
            mask_rgb = masks_01[:n].repeat(1, 3, 1, 1) if masks_01.shape[1] == 1 else masks_01[:n]
            rows.append(mask_rgb)
            row_labels.append("Mask")

        if masked_01 is not None:
            rows.append(masked_01[:n])
            row_labels.append("Masked")

        rows.append(gen_01[:n])
        row_labels.append("Generated")

        grid = make_grid(torch.cat(rows, dim=0), nrow=n, padding=4, pad_value=1.0)
        grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        caption = " | ".join(row_labels) + f" (step {step})"
        if caption_suffix:
            caption += f" {caption_suffix}"
        log_dict[f"{prefix}/grid"] = wandb.Image(grid_np, caption=caption)

        for i in range(n):
            orig_np = (orig_01[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gen_np = (gen_01[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            log_dict[f"{prefix}/sample_{i}_orig"] = wandb.Image(orig_np)
            log_dict[f"{prefix}/sample_{i}_gen"] = wandb.Image(gen_np)

        self.logger.experiment.log(log_dict)

    @torch.no_grad()
    def _run_ddim_inference(self, masked_images, masks, n=4):
        """DDIM 풀 인퍼런스로 인페인팅 결과 생성."""
        masked_lat = self.vae.encode(masked_images[:n]).latent_dist.sample()
        masked_lat = masked_lat * self.vae.config.scaling_factor
        lat_mask = F.interpolate(masks[:n], size=masked_lat.shape[2:], mode="nearest")

        latents = torch.randn(
            n, 4, masked_images.shape[2] // 8, masked_images.shape[3] // 8,
            device=self.device, dtype=masked_images.dtype,
        ) * self.infer_scheduler.init_noise_sigma

        enc_states = self.prompt_embeds.expand(n, -1, -1)
        self.infer_scheduler.set_timesteps(self.num_sample_inference_steps, device=self.device)

        for t in self.infer_scheduler.timesteps:
            model_input = self.infer_scheduler.scale_model_input(latents, t)
            model_input = torch.cat([model_input, lat_mask, masked_lat], dim=1)
            noise_pred = self.unet(model_input, t, enc_states).sample
            latents = self.infer_scheduler.step(noise_pred, t, latents).prev_sample

        decoded = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        return decoded.clamp(-1, 1)

    @torch.no_grad()
    def _save_samples(self, orig_images, masks, masked_images, max_samples=4,
                      prefix="train/inpaint"):
        if self.global_rank != 0:
            return

        save_dir = Path(self.args.output_dir) / "samples" / f"step_{self.global_step}"
        save_dir.mkdir(parents=True, exist_ok=True)

        n = min(max_samples, orig_images.shape[0])
        orig_01 = (orig_images[:n] + 1.0) * 0.5
        masked_01 = (masked_images[:n] + 1.0) * 0.5
        masks_01 = masks[:n].float()

        save_image(orig_01, save_dir / "original.png", nrow=n)
        save_image(masks_01, save_dir / "masks.png", nrow=n)

        decoded = self._run_ddim_inference(masked_images, masks, n)
        composite = self._composite(decoded, orig_images[:n], masks[:n])
        comp_01 = (composite + 1.0) * 0.5
        save_image(comp_01, save_dir / "inpainted_composite.png", nrow=n)

        self._log_wandb_images(
            orig_01, comp_01,
            masks_01=masks_01,
            masked_01=masked_01,
            prefix=prefix,
        )

    # -----------------------------------------------------------------
    #  Lifecycle hooks
    # -----------------------------------------------------------------

    def on_train_start(self):
        """학습 시작 시 초기 샘플을 WandB에 로깅하여 데이터 파이프라인 확인."""
        if self.global_rank != 0:
            return
        dl = self.trainer.train_dataloader
        batch = next(iter(dl))
        batch = {k: v.to(self.device) for k, v in batch.items()}
        n = min(4, batch["pixel_values"].shape[0])
        orig_01 = (batch["pixel_values"][:n] + 1.0) * 0.5
        masked_01 = (batch["masked_image"][:n] + 1.0) * 0.5
        masks_01 = batch["mask"][:n].float()
        self._log_wandb_images(
            orig_01, masked_01,
            masks_01=masks_01,
            prefix="data/initial",
            caption_suffix="(before training)",
        )

    def on_validation_epoch_end(self):
        """Validation epoch 끝에 DDIM 풀 인퍼런스 샘플 로깅."""
        if self.global_rank != 0:
            return
        try:
            dl = self.trainer.val_dataloaders
            if isinstance(dl, list):
                dl = dl[0]
            batch = next(iter(dl))
            batch = {k: v.to(self.device) for k, v in batch.items()}
        except StopIteration:
            return

        n = min(4, batch["pixel_values"].shape[0])
        self._save_samples(
            batch["pixel_values"], batch["mask"], batch["masked_image"],
            max_samples=n, prefix="val/inpaint",
        )

    def on_train_epoch_end(self):
        if self.global_rank != 0:
            return
        if (self.current_epoch + 1) % self.args.save_every_n_epochs == 0:
            path = Path(self.args.output_dir) / f"lora_epoch_{self.current_epoch:03d}.pt"
            torch.save(save_lora_state_dict(self.unet), path)
            print(f"[Saved] LoRA weights → {path}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.lora_params,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )

        sched = getattr(self.args, "lr_scheduler", "constant")
        if sched == "constant":
            return optimizer

        total_steps = self.trainer.estimated_stepping_batches
        warmup = self.args.lr_warmup_steps

        def cosine_with_warmup(step):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, total_steps - warmup))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# =============================================================================
#  Arguments
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Face De-Identification via Inpainting + LoRA"
    )

    # -- 모델 --
    p.add_argument("--pretrained_model_name_or_path", type=str,
                   default="runwayml/stable-diffusion-inpainting",
                   help="diffusers 포맷 모델 경로 (single_file_ckpt 미지정 시 사용)")
    p.add_argument("--single_file_ckpt", type=str, default=None,
                   help="safetensors 체크포인트 경로 (지정 시 from_single_file로 로딩)")
    p.add_argument("--vae_path", type=str, default=None,
                   help="외부 VAE 경로 (예: stabilityai/sd-vae-ft-mse)")
    p.add_argument("--prompt", type=str,
                   default="a highly detailed photo of a person face, sharp focus, natural skin texture")

    # -- LoRA --
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)

    # -- Face Identity 모델 --
    p.add_argument("--face_id_model_path", type=str, default=None)
    p.add_argument("--face_id_backbone", type=str, default="resnet50",
                   choices=["resnet50", "resnet18"])
    p.add_argument("--face_id_embedding_dim", type=int, default=512)

    # -- Feature 모델 --
    p.add_argument("--feature_model_name", type=str, default="ResNet50",
                   choices=["ResNet50", "MobileNet_AVG"])
    p.add_argument("--feature_model_path", type=str,
                   default="/workspace/model/weight/sscd_disc_mixup.torchvision.pt")

    # -- 데이터 --
    p.add_argument("--data_root", type=str, required=True,
                   help="데이터셋 루트 (images/, masks/, labels/ 포함)")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=2)
    p.add_argument("--val_batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)

    # -- Loss 가중치 --
    p.add_argument("--lambda_diffusion", type=float, default=1.0)
    p.add_argument("--lambda_identity", type=float, default=0.3)
    p.add_argument("--lambda_feature", type=float, default=0.5)
    p.add_argument("--lambda_lpips", type=float, default=0.5)

    # -- Anti-collapse --
    p.add_argument("--id_margin", type=float, default=0.3)
    p.add_argument("--timestep_threshold", type=int, default=250)
    p.add_argument("--aux_warmup_steps", type=int, default=2000)

    # -- 학습 --
    p.add_argument("--num_train_epochs", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument(
        "--lr_scheduler", type=str, default="constant",
        choices=["constant", "cosine"],
        help="constant: 고정 LR | cosine: warmup 후 코사인 감쇠",
    )
    p.add_argument("--lr_warmup_steps", type=int, default=0,
                   help="lr_scheduler=cosine일 때만 사용")
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_weight_decay", type=float, default=1e-2)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # -- 출력 / 로깅 --
    p.add_argument("--output_dir", type=str, default="./output/inpaint_deid_lora")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed_precision", type=str, default="fp16",
                   choices=["no", "fp16", "bf16"])
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--wandb_project", type=str, default="face-deid-inpaint")
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--log_every_n_steps", type=int, default=10)
    p.add_argument("--save_every_n_epochs", type=int, default=5)
    p.add_argument("--sample_every_n_steps", type=int, default=100)
    p.add_argument("--num_sample_inference_steps", type=int, default=30)
    p.add_argument("--val_check_interval", type=float, default=1.0)

    return p.parse_args()


# =============================================================================
#  Main
# =============================================================================

def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model = FaceInpaintDeIdLitModule(args)

    # ---- Data ----
    train_dataset = InpaintDeIdDataset(
        data_root=args.data_root, split="train", size=args.resolution
    )
    val_dataset = InpaintDeIdDataset(
        data_root=args.data_root, split="val", size=args.resolution
    )

    def collate_fn(batch):
        result = {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "mask": torch.stack([b["mask"] for b in batch]),
            "masked_image": torch.stack([b["masked_image"] for b in batch]),
        }
        if "bbox" in batch[0]:
            result["bbox"] = torch.stack([b["bbox"] for b in batch])
        return result

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )

    print(f"[Data] Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")
    print(f"[Data] Train batches: {len(train_loader):,} | Val batches: {len(val_loader):,}")

    # ---- WandB ----
    n_lora = sum(p.numel() for p in model.lora_params)
    n_unet = sum(p.numel() for p in model.unet.parameters())

    wandb_config = {
        **vars(args),
        "architecture": "SD-Inpainting-LoRA",
        "unet_input_channels": 9,
        "dataset_train": len(train_dataset),
        "dataset_val": len(val_dataset),
        "lora_trainable_params": n_lora,
        "unet_total_params": n_unet,
        "lora_param_ratio": f"{100 * n_lora / n_unet:.2f}%",
        "effective_batch_size": args.train_batch_size * args.gradient_accumulation_steps * args.num_gpus,
        "anti_collapse": {
            "id_margin": args.id_margin,
            "timestep_threshold": args.timestep_threshold,
            "aux_warmup_steps": args.aux_warmup_steps,
        },
    }

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_name or f"inpaint_deid_r{args.lora_rank}",
        save_dir=args.output_dir,
        config=wandb_config,
    )

    # ---- Callbacks ----
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="inpaint-deid-{epoch:03d}-{val/loss_total:.4f}",
            save_top_k=3,
            monitor="val/loss_total",
            mode="min",
            every_n_epochs=args.save_every_n_epochs,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    precision_map = {"no": 32, "fp16": "16-mixed", "bf16": "bf16-mixed"}

    if args.num_gpus > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
            timeout=datetime.timedelta(minutes=60),
        )
    else:
        strategy = "auto"

    # ---- Trainer ----
    trainer = pl.Trainer(
        max_epochs=args.num_train_epochs,
        accelerator="gpu",
        devices=args.num_gpus,
        strategy=strategy,
        precision=precision_map.get(args.mixed_precision, 32),
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=args.max_grad_norm,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        default_root_dir=args.output_dir,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if trainer.global_rank == 0:
        final_path = Path(args.output_dir) / "final_lora_weights.pt"
        torch.save(save_lora_state_dict(model.unet), final_path)
        print(f"[Done] Final LoRA weights → {final_path}")


if __name__ == "__main__":
    main()
