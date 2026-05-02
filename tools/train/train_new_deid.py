"""Face De-Identification LoRA Training (v2 — Anti-Collapse)

SD UNet LoRA 학습. 5개 loss + 3개 안전장치로 black-box collapse 방지.
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

from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model.icd.sscd.sscd.models.model import Model as SSCDResNet
from model.vcd.vcd.models.frame import MobileNet_AVG


# ── LoRA ─────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(rank, original.in_features))
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        original.weight.requires_grad_(False)
        if original.bias is not None:
            original.bias.requires_grad_(False)

    def forward(self, x, *args, **kwargs):
        return self.original(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling

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


def inject_lora(model, rank=4, alpha=1.0, target_suffixes=None):
    if target_suffixes is None:
        target_suffixes = ["to_q", "to_k", "to_v", "to_out.0"]
    named = dict(model.named_modules())
    injected = {}
    for full_name, module in list(named.items()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(full_name.endswith(s) for s in target_suffixes):
            continue
        lora = LoRALinear(module, rank=rank, alpha=alpha)
        parts = full_name.rsplit(".", 1)
        parent = named[parts[0]] if len(parts) == 2 else model
        attr = parts[1] if len(parts) == 2 else parts[0]
        if isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent[int(attr)] = lora
        else:
            setattr(parent, attr, lora)
        injected[full_name] = lora
    return injected


def collect_lora_params(model):
    params = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            params.extend([m.lora_A, m.lora_B])
    return params


def save_lora_state_dict(model):
    sd = {}
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            sd[f"{name}.lora_A"] = m.lora_A.data.clone()
            sd[f"{name}.lora_B"] = m.lora_B.data.clone()
            sd[f"{name}.rank"] = torch.tensor(m.rank)
            sd[f"{name}.alpha"] = torch.tensor(m.alpha)
    return sd


def load_lora_state_dict(model, state_dict):
    loaded = 0
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            a, b = f"{name}.lora_A", f"{name}.lora_B"
            if a in state_dict and b in state_dict:
                m.lora_A.data.copy_(state_dict[a])
                m.lora_B.data.copy_(state_dict[b])
                loaded += 1
    return loaded


# ── Dataset ──────────────────────────────────────────────────────────────────

class FFHQDeIdDataset(Dataset):
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, data_dir: str, size: int = 512):
        self.image_paths = sorted(
            p for p in Path(data_dir).rglob("*")
            if p.suffix.lower() in self.EXTENSIONS and p.is_file()
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {data_dir}")
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return {"pixel_values": self.transform(Image.open(self.image_paths[idx]).convert("RGB"))}


# ── Face Identity Extractor ──────────────────────────────────────────────────

class FaceIdentityExtractor(nn.Module):
    def __init__(self, model_path=None, backbone="resnet50", embedding_dim=512):
        super().__init__()
        self.input_size = 160
        import torchvision.models as tv

        if backbone == "resnet50":
            resnet = tv.resnet50(weights=tv.ResNet50_Weights.DEFAULT if not model_path else None)
            feat_dim = 2048
        else:
            resnet = tv.resnet18(weights=tv.ResNet18_Weights.DEFAULT if not model_path else None)
            feat_dim = 512

        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(feat_dim, embedding_dim)

        if model_path and os.path.exists(model_path):
            sd = torch.load(model_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            sd = {k.replace("module.", "").replace("model.", ""): v for k, v in sd.items()}
            self.load_state_dict(sd, strict=False)

        self._normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def forward(self, x):
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return F.normalize(self.projection(self.features(self._normalize(x)).flatten(1)), p=2, dim=1)


# ── Lightning Module ─────────────────────────────────────────────────────────

class FaceDeIdLitModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(vars(args) if isinstance(args, argparse.Namespace) else args)
        self.args = args

        self.tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        self.lora_layers = inject_lora(self.unet, rank=args.lora_rank, alpha=args.lora_alpha)
        self.lora_params = collect_lora_params(self.unet)
        n_lora = sum(p.numel() for p in self.lora_params)
        n_unet = sum(p.numel() for p in self.unet.parameters())
        print(f"[LoRA] Trainable: {n_lora:,} / UNet: {n_unet:,} ({100*n_lora/n_unet:.2f}%)")

        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.infer_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

        self.face_id_model = FaceIdentityExtractor(
            getattr(args, "face_id_model_path", None),
            getattr(args, "face_id_backbone", "resnet50"),
            getattr(args, "face_id_embedding_dim", 512),
        )
        self.face_id_model.eval().requires_grad_(False)

        self.feature_model = self._build_feature_model(args)
        self.feature_model.eval().requires_grad_(False)

        self.lpips_model = lpips.LPIPS(net="vgg")
        self.lpips_model.eval().requires_grad_(False)

        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.register_buffer("prompt_embeds", self._encode_prompt(getattr(args, "prompt", "a photo of a face")), persistent=False)

        self.lambda_diff = args.lambda_diffusion
        self.lambda_id = args.lambda_identity
        self.lambda_feat = args.lambda_feature
        self.lambda_lpips = args.lambda_lpips
        self.lambda_recon = args.lambda_recon
        self.id_margin = args.id_margin
        self.timestep_threshold = args.timestep_threshold
        self.aux_warmup_steps = args.aux_warmup_steps
        self.num_sample_inference_steps = getattr(args, "num_sample_inference_steps", 30)

        if getattr(args, "gradient_checkpointing", False):
            self.unet.enable_gradient_checkpointing()
            self.vae.enable_gradient_checkpointing()

    def _build_feature_model(self, args):
        if args.feature_model_name == "ResNet50":
            model = SSCDResNet(backbone="TV_RESNET50", dims=512, pool_param=3)
            model.load_state_dict(torch.load(args.feature_model_path, map_location="cpu"))
        else:
            model = MobileNet_AVG()
            sd = torch.load(args.feature_model_path, map_location="cpu")
            model.load_state_dict({k.replace("base.", ""): v for k, v in sd.items()}, strict=False)
        return model

    @torch.no_grad()
    def _encode_prompt(self, prompt):
        tokens = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                                truncation=True, return_tensors="pt")
        return self.text_encoder(tokens.input_ids)[0]

    def _predict_x0(self, noisy_latents, noise_pred, timesteps):
        """x0 = (x_t - sqrt(1-α_t)·ε) / sqrt(α_t)"""
        ac = self.noise_scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        a = ac[timesteps].view(-1, 1, 1, 1)
        return (noisy_latents - torch.sqrt(1.0 - a) * noise_pred) / torch.sqrt(a)

    def _aux_warmup_factor(self):
        if self.aux_warmup_steps <= 0:
            return 1.0
        return min(1.0, self.global_step / self.aux_warmup_steps)

    def training_step(self, batch, batch_idx):
        pv = batch["pixel_values"]
        bsz = pv.shape[0]
        enc_h = self.prompt_embeds.expand(bsz, -1, -1)

        with torch.no_grad():
            latents = self.vae.encode(pv).latent_dist.sample() * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        ts = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device, dtype=torch.long)
        noisy = self.noise_scheduler.add_noise(latents, noise, ts)
        noise_pred = self.unet(noisy, ts, enc_h).sample

        target = noise if self.noise_scheduler.config.prediction_type == "epsilon" else self.noise_scheduler.get_velocity(latents, noise, ts)
        diff_loss = F.mse_loss(noise_pred.float(), target.float())

        warmup = self._aux_warmup_factor()
        valid = (ts < self.timestep_threshold)
        n_valid = valid.sum().item()
        zero = torch.tensor(0.0, device=self.device)

        if n_valid > 0 and warmup > 0:
            x0_lat = self._predict_x0(noisy, noise_pred, ts)
            ids, feats, lps, recs = [], [], [], []

            for idx in valid.nonzero(as_tuple=True)[0]:
                i = idx.item()
                pred_i = self.vae.decode(x0_lat[i:i+1] / self.vae.config.scaling_factor).sample.clamp(-1, 1)
                p01, o01 = (pred_i + 1) * 0.5, (pv[i:i+1] + 1) * 0.5

                with torch.no_grad():
                    oid = self.face_id_model(o01)
                    ofeat = self.feature_model(o01)
                ids.append(self.cos_sim(oid, self.face_id_model(p01)))
                feats.append(self.cos_sim(ofeat, self.feature_model(p01)))
                lps.append(self.lpips_model(pv[i:i+1], pred_i).squeeze())
                recs.append(F.l1_loss(pred_i, pv[i:i+1]))

            id_cos = torch.cat(ids)
            feat_cos = torch.cat(feats)
            id_loss = torch.clamp(id_cos - self.id_margin, min=0).mean()
            feat_loss = (1.0 - feat_cos).mean()
            lpips_loss = torch.stack(lps).mean()
            recon_loss = torch.stack(recs).mean()
            tw = self.noise_scheduler.alphas_cumprod.to(self.device)[ts[valid]].mean()
        else:
            id_cos, feat_cos = zero, zero
            id_loss = feat_loss = lpips_loss = recon_loss = tw = zero

        aux = warmup * tw
        total = self.lambda_diff * diff_loss + aux * (
            self.lambda_id * id_loss + self.lambda_feat * feat_loss
            + self.lambda_lpips * lpips_loss + self.lambda_recon * recon_loss
        )

        self.log("train/loss_total", total, prog_bar=True, sync_dist=True)
        self.log("train/loss_diffusion", diff_loss, prog_bar=True, sync_dist=True)
        self.log("train/loss_identity", id_loss, prog_bar=True, sync_dist=True)
        self.log("train/loss_feature", feat_loss, sync_dist=True)
        self.log("train/loss_lpips", lpips_loss, sync_dist=True)
        self.log("train/loss_recon", recon_loss, sync_dist=True)
        self.log("train/id_cosine_sim", id_cos.mean() if id_cos.dim() > 0 else id_cos, sync_dist=True)
        self.log("train/feat_cosine_sim", feat_cos.mean() if feat_cos.dim() > 0 else feat_cos, sync_dist=True)
        self.log("train/aux_scale", aux, sync_dist=True)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], sync_dist=True)

        if (self.global_step + 1) % self.args.sample_every_n_steps == 0:
            self._save_samples(pv[:4])

        return total

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pv = batch["pixel_values"]
        bsz = pv.shape[0]
        enc_h = self.prompt_embeds.expand(bsz, -1, -1)

        latents = self.vae.encode(pv).latent_dist.sample() * self.vae.config.scaling_factor
        ts = torch.randint(0, self.timestep_threshold, (bsz,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(latents)
        noisy = self.noise_scheduler.add_noise(latents, noise, ts)
        noise_pred = self.unet(noisy, ts, enc_h).sample

        target = noise if self.noise_scheduler.config.prediction_type == "epsilon" else self.noise_scheduler.get_velocity(latents, noise, ts)
        diff_loss = F.mse_loss(noise_pred.float(), target.float())

        x0_lat = self._predict_x0(noisy, noise_pred, ts)
        ids, feats, lps, recs, log_imgs = [], [], [], [], []

        for i in range(bsz):
            pred_i = self.vae.decode(x0_lat[i:i+1] / self.vae.config.scaling_factor).sample.clamp(-1, 1)
            p01, o01 = (pred_i + 1) * 0.5, (pv[i:i+1] + 1) * 0.5
            ids.append(self.cos_sim(self.face_id_model(o01), self.face_id_model(p01)))
            feats.append(self.cos_sim(self.feature_model(o01), self.feature_model(p01)))
            lps.append(self.lpips_model(pv[i:i+1], pred_i).squeeze())
            recs.append(F.l1_loss(pred_i, pv[i:i+1]))
            if batch_idx == 0 and i < 4:
                log_imgs.append(p01)

        id_cos, feat_cos = torch.cat(ids), torch.cat(feats)
        id_loss = torch.clamp(id_cos - self.id_margin, min=0).mean()
        feat_loss = (1.0 - feat_cos).mean()
        lpips_loss = torch.stack(lps).mean()
        recon_loss = torch.stack(recs).mean()
        tw = self.noise_scheduler.alphas_cumprod.to(self.device)[ts].mean()

        total = self.lambda_diff * diff_loss + tw * (
            self.lambda_id * id_loss + self.lambda_feat * feat_loss
            + self.lambda_lpips * lpips_loss + self.lambda_recon * recon_loss
        )

        self.log("val/loss_total", total, prog_bar=True, sync_dist=True)
        self.log("val/loss_diffusion", diff_loss, sync_dist=True)
        self.log("val/loss_identity", id_loss, sync_dist=True)
        self.log("val/loss_feature", feat_loss, sync_dist=True)
        self.log("val/id_cosine_sim", id_cos.mean(), prog_bar=True, sync_dist=True)
        self.log("val/feat_cosine_sim", feat_cos.mean(), prog_bar=True, sync_dist=True)

        if batch_idx == 0 and self.global_rank == 0 and log_imgs:
            self._log_wandb_images((pv[:len(log_imgs)] + 1) * 0.5, torch.cat(log_imgs), "val")

        return total

    @torch.no_grad()
    def _log_wandb_images(self, orig_01, pred_01, prefix="train"):
        if not isinstance(self.logger, WandbLogger):
            return
        n = min(4, orig_01.shape[0])
        grid = make_grid(torch.cat([orig_01[:n], pred_01[:n]]), nrow=n, padding=2)
        self.logger.experiment.log({
            f"{prefix}/comparison": wandb.Image(grid.permute(1, 2, 0).cpu().numpy(),
                caption=f"Top: Original / Bottom: Generated (step {self.global_step})"),
            "global_step": self.global_step,
        })

    @torch.no_grad()
    def _save_samples(self, orig_images, max_samples=4):
        if self.global_rank != 0:
            return
        save_dir = Path(self.args.output_dir) / "samples" / f"step_{self.global_step}"
        save_dir.mkdir(parents=True, exist_ok=True)

        n = min(max_samples, orig_images.shape[0])
        save_image((orig_images[:n] + 1) * 0.5, save_dir / "original.png", nrow=n)

        self.infer_scheduler.set_timesteps(self.num_sample_inference_steps, device=self.device)
        lat = torch.randn(n, 4, orig_images.shape[2]//8, orig_images.shape[3]//8,
                          device=self.device, dtype=orig_images.dtype) * self.infer_scheduler.init_noise_sigma
        enc = self.prompt_embeds.expand(n, -1, -1)
        for t in self.infer_scheduler.timesteps:
            lat = self.infer_scheduler.step(
                self.unet(self.infer_scheduler.scale_model_input(lat, t), t, enc).sample, t, lat
            ).prev_sample
        dec = ((self.vae.decode(lat / self.vae.config.scaling_factor).sample + 1) * 0.5).clamp(0, 1)
        save_image(dec, save_dir / "ddim_generated.png", nrow=n)
        self._log_wandb_images((orig_images[:n] + 1) * 0.5, dec, "train/ddim")

    def on_train_epoch_end(self):
        if self.global_rank != 0:
            return
        if (self.current_epoch + 1) % self.args.save_every_n_epochs == 0:
            p = Path(self.args.output_dir) / f"lora_epoch_{self.current_epoch:03d}.pt"
            torch.save(save_lora_state_dict(self.unet), p)
            print(f"[Saved] LoRA → {p}")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.lora_params, lr=self.args.learning_rate,
                                betas=(self.args.adam_beta1, self.args.adam_beta2),
                                weight_decay=self.args.adam_weight_decay, eps=self.args.adam_epsilon)
        total_steps = self.trainer.estimated_stepping_batches
        w = self.args.lr_warmup_steps

        def lr_fn(step):
            if step < w:
                return step / max(1, w)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (step - w) / max(1, total_steps - w))))

        return {"optimizer": opt, "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lr_fn), "interval": "step"}}


# ── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--prompt", type=str, default="a photo of a face")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--face_id_model_path", type=str, default=None)
    p.add_argument("--face_id_backbone", type=str, default="resnet50", choices=["resnet50", "resnet18"])
    p.add_argument("--face_id_embedding_dim", type=int, default=512)
    p.add_argument("--feature_model_name", type=str, default="ResNet50", choices=["ResNet50", "MobileNet_AVG"])
    p.add_argument("--feature_model_path", type=str, default="/workspace/model/weight/sscd_disc_mixup.torchvision.pt")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--val_batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.05)
    p.add_argument("--lambda_diffusion", type=float, default=1.0)
    p.add_argument("--lambda_identity", type=float, default=0.3)
    p.add_argument("--lambda_feature", type=float, default=0.5)
    p.add_argument("--lambda_lpips", type=float, default=1.0)
    p.add_argument("--lambda_recon", type=float, default=0.5)
    p.add_argument("--id_margin", type=float, default=0.3)
    p.add_argument("--timestep_threshold", type=int, default=250)
    p.add_argument("--aux_warmup_steps", type=int, default=2000)
    p.add_argument("--num_train_epochs", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_warmup_steps", type=int, default=500)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_weight_decay", type=float, default=1e-2)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--output_dir", type=str, default="./output/deid_lora")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--wandb_project", type=str, default="face-deid-lora")
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--log_every_n_steps", type=int, default=10)
    p.add_argument("--save_every_n_epochs", type=int, default=5)
    p.add_argument("--sample_every_n_steps", type=int, default=500)
    p.add_argument("--num_sample_inference_steps", type=int, default=30)
    p.add_argument("--val_check_interval", type=float, default=1.0)
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model = FaceDeIdLitModule(args)

    full_ds = FFHQDeIdDataset(data_dir=args.data_dir, size=args.resolution)
    val_n = int(len(full_ds) * args.val_split)
    train_n = len(full_ds) - val_n
    train_ds, val_ds = random_split(full_ds, [train_n, val_n], generator=torch.Generator().manual_seed(args.seed))

    kw = dict(num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    train_dl = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False, **kw)

    print(f"[Data] Train: {train_n:,} | Val: {val_n:,} | Batches: {len(train_dl):,}/{len(val_dl):,}")

    n_lora = sum(p.numel() for p in model.lora_params)
    n_unet = sum(p.numel() for p in model.unet.parameters())

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_name or f"deid_lora_r{args.lora_rank}",
        save_dir=args.output_dir,
        config={**vars(args), "lora_params": n_lora, "unet_params": n_unet,
                "lora_ratio": f"{100*n_lora/n_unet:.2f}%",
                "effective_bs": args.train_batch_size * args.gradient_accumulation_steps * args.num_gpus},
    )

    callbacks = [
        ModelCheckpoint(dirpath=os.path.join(args.output_dir, "checkpoints"),
                        filename="deid-{epoch:03d}-{val/loss_total:.4f}",
                        save_top_k=3, monitor="val/loss_total", mode="min",
                        every_n_epochs=args.save_every_n_epochs),
        LearningRateMonitor(logging_interval="step"),
    ]

    prec_map = {"no": 32, "fp16": "16-mixed", "bf16": "bf16-mixed"}
    strategy = DDPStrategy(find_unused_parameters=False, timeout=datetime.timedelta(minutes=60)) if args.num_gpus > 1 else "auto"

    trainer = pl.Trainer(
        max_epochs=args.num_train_epochs, accelerator="gpu", devices=args.num_gpus,
        strategy=strategy, precision=prec_map.get(args.mixed_precision, 32),
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=args.max_grad_norm, logger=wandb_logger, callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps, val_check_interval=args.val_check_interval,
        default_root_dir=args.output_dir, enable_progress_bar=True,
    )

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    if trainer.global_rank == 0:
        final = Path(args.output_dir) / "final_lora_weights.pt"
        torch.save(save_lora_state_dict(model.unet), final)
        print(f"[Done] LoRA → {final}")


if __name__ == "__main__":
    main()
