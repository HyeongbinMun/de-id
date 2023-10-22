import os
import shutil
import sys

import wandb
import datetime
import argparse
import itertools

from torchvision import transforms
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utility import logging
from utility.params import load_params_yml
from utility.argment.stylegan2 import StyleGAN2Argments
from utility.model.train import requires_grad
from utility.model.model_file import load_checkpoint, save_checkpoint
from utility.model.region import crop_face, overlay_faces_on_image, save_gan_concat_text_image
from config.models import model_classes

from model.deid.dataset.dataset import GANFaceDetDataset
from model.deid.stylegan2.distributed import get_world_size, reduce_sum, reduce_loss_dict
from model.deid.stylegan2.model import Generator, Discriminator
from model.deid.stylegan2.non_leaking import AdaptiveAugment, augment
from model.deid.stylegan2.train import accumulate, mixing_noise, d_logistic_loss, d_r1_loss, g_nonsaturating_loss, g_path_regularize

from model.deid.stylegan2.utils.non_leaking import AdaptiveAugment, augment
from model.deid.stylegan2.utils.train import accumulate, reduce_loss_dict, mixing_noise, d_logistic_loss, d_r1_loss, g_nonsaturating_loss, g_path_regularize, get_world_size, reduce_sum
from model.deid.stylegan2.models.stylegan2 import Generator, Discriminator



def train(rank, params):
    # if params["model"]["stylegan2"]["resume"]:
    #     tmp_start_time_stamp = params["model"]["stylegan2"]["pretrained_model_dir"].split("/")[-3].split("_")
    #     start_timestamp = f"{tmp_start_time_stamp[0]}_{tmp_start_time_stamp[1]}"
    #     wandb_run_name = f"{start_timestamp}-{params['model']['stylegan2']['model_name']}"
    #     wandb.init(project=params['wandb']['project_name'], entity=params['wandb']['account'], name=wandb_run_name, resume=True, id=wandb_run_name)
    # else:
    #     start_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     wandb_run_name = f"{start_timestamp}-{params['model']['stylegan2']['model_name']}"
    #     wandb.init(project=params['wandb']['project_name'], entity=params['wandb']['account'], name=wandb_run_name)

    batch_size = params["batch_size"]
    start_epoch = 0
    end_epoch = params["epoch"]
    valid_epoch = params["valid_epoch"]
    decay_epoch = params["decay_epoch"]

    config = StyleGAN2Argments(params["model"]["stylegan2"])
    dataset_dir = params["dataset_dir"]

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(rank)

    image_orin_dir = os.path.join(dataset_dir, "orin")
    image_deid_dir = os.path.join(dataset_dir, "deid")
    label_dir = os.path.join(dataset_dir, "labels")

    train_image_real_dir, train_image_deid_dir, train_label_dir = os.path.join(image_orin_dir, "train"), os.path.join(image_deid_dir, "train"),os.path.join(label_dir, "train")
    valid_image_real_dir, valid_image_deid_dir, valid_label_dir = os.path.join(image_orin_dir, "valid"),os.path.join(image_deid_dir, "valid"), os.path.join(label_dir, "valid")
    test_image_real_dir, test_image_deid_dir, test_label_dir = os.path.join(image_orin_dir, "test"), os.path.join(image_deid_dir, "test"), os.path.join(label_dir, "test")

    save_dir = params["save_dir"]
    # model_dir = os.path.join(save_dir, start_timestamp + "_" + params["model"]["stylegan2"]["model_name"])
    # model_save_dir = os.path.join(model_dir, "weights")
    # generated_images_dir = os.path.join(model_dir, 'generated_images')
    # if not os.path.exists(model_save_dir):
    #     os.makedirs(model_save_dir)
    # if not os.path.exists(generated_images_dir):
    #     os.makedirs(generated_images_dir)

    # Data Loaders
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    train_dataset = GANFaceDetDataset(train_image_real_dir, train_image_deid_dir, train_label_dir, transform=transform)
    valid_dataset = GANFaceDetDataset(valid_image_real_dir, valid_image_deid_dir, valid_label_dir)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=1,
                              collate_fn=GANFaceDetDataset.collate_fn)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=1,
                              collate_fn=GANFaceDetDataset.collate_fn)

    if params["model"]["feature"]["model_name"] == "ResNet50":
        feature_model = model_classes["feature"][params["model"]["feature"]["model_name"]](backbone="TV_RESNET50", dims=512, pool_param=3).to(device)
        feature_model.load_state_dict(torch.load(params["model"]["feature"]["feature_weight_path"]))
    else:
        feature_model = model_classes["feature"][params["model"]["feature"]["model_name"]]().to(device)
        state_dict = torch.load(params["model"]["feature"]["feature_weight_path"], map_location=device)
        state_dict = {k.replace("base.", ""): v for k, v in state_dict.items()}
        feature_model.load_state_dict(state_dict, strict=False)
    feature_model.eval()

    # Model Initialization
    generator = Generator(config.input_size, config.latent, config.n_mlp, channel_multiplier=config.channel_multiplier).to(device)
    discriminator = Discriminator(config.input_size, channel_multiplier=config.channel_multiplier).to(device)
    g_ema = Generator(config.input_size, config.latent, config.n_mlp, channel_multiplier=config.channel_multiplier).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = config.g_reg_every / (config.g_reg_every + 1)
    d_reg_ratio = config.d_reg_every / (config.d_reg_every + 1)

    # Optimizers setup
    g_optim = torch.optim.Adam(
        generator.parameters(),
        lr=config.learning_rate * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.learning_rate * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    # Move models to GPU if available


    # Learning Rate Schedulers

    # Model Loading or Weight Initialization


    print(logging.i("Dataset Description"))
    print(logging.s(f"    training     : {len(train_dataset)}"))
    print(logging.s(f"    validation   : {len(valid_dataset)}"))
    print(logging.i("Parameters Description"))
    print(logging.s(f"    Device       : {params['device']}"))
    print(logging.s(f"    Batch size   : {params['batch_size']}"))
    print(logging.s(f"    Max epoch    : {params['epoch']}"))
    print(logging.s(f"    Learning rate: {config.learning_rate}"))
    print(logging.s(f"    Decay epoch: {decay_epoch}"))
    print(logging.s(f"    Valid epoch: {valid_epoch}"))
    print(logging.s(f"    Model:"))
    print(logging.s(f"    - Feature extractor: {params['model']['feature']['model_name']}"))
    print(logging.s(f"    - StyleGAN2        : {config.model_name}"))
    print(logging.s(f"       n_sample             : {config.n_sample}"))
    print(logging.s(f"       input_size           : {config.input_size}"))
    print(logging.s(f"       r1                   : {config.r1}"))
    print(logging.s(f"       path_regularize      : {config.path_regularize}"))
    print(logging.s(f"       path_batch_shrink    : {config.path_batch_shrink}"))
    print(logging.s(f"       d_reg_every          : {config.d_reg_every}"))
    print(logging.s(f"       g_reg_every          : {config.g_reg_every}"))
    print(logging.s(f"       mixing               : {config.mixing}"))
    print(logging.s(f"       learning_rate        : {config.learning_rate}"))
    print(logging.s(f"       channel_multiplier   : {config.channel_multiplier}"))
    print(logging.s(f"       augment              : {config.augment}"))
    print(logging.s(f"       augment_p            : {config.augment_p}"))
    print(logging.s(f"       ada_target           : {config.ada_target}"))
    print(logging.s(f"       ada_length           : {config.ada_length}"))
    # print(logging.s(f"Loss lambda:"))
    # print(logging.s(f"- Lambda Cycle         : {lambda_cycle}"))
    # print(logging.s(f"- Lambda Identity      : {lambda_identity}"))
    # print(logging.s(f"- Lambda full image    : {lambda_full_image}"))
    print(logging.s(f"Task Information"))
    # print(logging.s(f"- Task ID              : {start_timestamp + '_' + params['model']['stylegan2']['model_name']}"))
    # print(logging.s(f"- Model directory      : {model_dir}"))
    print(logging.s(f"- Resume               : {params['model']['stylegan2']['resume']}"))
    if params['model']['stylegan2']['resume']:
        pretrained_model_dir = params["model"]["stylegan2"]["pretrained_model_dir"]
        model_names = os.listdir(pretrained_model_dir)
        generator_orin2deid_model_name = sorted([item for item in model_names if "generator_orin2deid" in item])[-1]
        discriminator_orin_model_name = sorted([item for item in model_names if "discriminator_orin" in item])[-1]
        discriminator_deid_model_name = sorted([item for item in model_names if "discriminator_deid" in item])[-1]
        print(logging.s(f"- Pretrained model directory: {params['model']['stylegan2']['pretrained_model_dir']}"))
        print(logging.s(f"   Generator Orin2Deid: {generator_orin2deid_model_name}"))
        print(logging.s(f"   Discriminator Orin: {discriminator_orin_model_name}"))
        print(logging.s(f"   Discriminator Deid: {discriminator_deid_model_name}"))

    mean_path_length = 0
    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    g_module = generator
    d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = config.augment_p if config.augment_p > 0 else 0.0
    r_t_stat = 0

    if config.augment and config.augment_p == 0:
        ada_augment = AdaptiveAugment(config.ada_target, config.ada_length, 8, device)

    sample_z = torch.randn(config.n_sample, config.latent, device=device)

    terminal_size = shutil.get_terminal_size().columns
    print("Status   Dev   Epoch  GAN   D_orin D_deid feature")
    print("-" * terminal_size)

    for epoch in range(start_epoch, end_epoch):
        epoch_loss_generator = 0.0
        epoch_loss_feature = 0.0
        epoch_loss_discriminator_real_fake = 0.0
        epoch_loss_discriminator_orin_deid = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (images_real_path, image_deid_path, label_path, images_orin, images_deid, boxes_list) in progress_bar:
            images_orin = images_orin.clone().to(device)
            images_deid = images_deid.clone().to(device)

            batch_real_orin_faces = []
            batch_real_deid_faces = []
            batch_images_indices = []
            batch_faces_boxes = []

            for i, boxes in enumerate(boxes_list):
                real_orin_faces, faces_index, face_boxes = crop_face(images_orin, boxes, i)
                real_deid_faces, __ , __ = crop_face(images_deid, boxes, i)
                batch_real_orin_faces.extend(real_orin_faces)
                batch_real_deid_faces.extend(real_deid_faces)
                batch_images_indices.extend(faces_index)
                batch_faces_boxes.extend(face_boxes)

            if len(batch_real_orin_faces) > 0 and len(batch_real_deid_faces) > 0 :
                batch_real_orin_faces = [F.interpolate(face.unsqueeze(0), size=(config.input_size, config.input_size), mode='bilinear', align_corners=False).squeeze(0) for face in batch_real_orin_faces]
                batch_real_deid_faces = [F.interpolate(face.unsqueeze(0), size=(config.input_size, config.input_size), mode='bilinear', align_corners=False).squeeze(0) for face in batch_real_deid_faces]
                batch_real_orin_faces = torch.stack(batch_real_orin_faces)
                batch_real_deid_faces = torch.stack(batch_real_deid_faces)

                requires_grad(generator, False)
                requires_grad(discriminator, True)

                noise = mixing_noise(batch_real_orin_faces.shape[0], config.latent, config.mixing, device)
                batch_fake_deid_faces, _ = generator(noise)

                print(batch_real_orin_faces.shape, batch_real_orin_faces.shape)
                fake_pred = discriminator(batch_fake_deid_faces)
                real_pred = discriminator(batch_real_orin_faces)
                loss_discriminator = d_logistic_loss(real_pred, fake_pred)

                loss_dict["d"] = loss_discriminator
                loss_dict["real_score"] = real_pred.mean()
                loss_dict["fake_score"] = fake_pred.mean()

                discriminator.zero_grad()
                loss_discriminator.backward()
                d_optim.step()

                d_regularize = epoch % config.d_reg_every == 0

                if d_regularize:
                    batch_real_orin_faces.requires_grad = True

                    batch_real_orin_faces_aug = batch_real_orin_faces

                    real_pred = discriminator(batch_real_orin_faces_aug)
                    r1_loss = d_r1_loss(real_pred, batch_real_orin_faces)

                    discriminator.zero_grad()
                    (config.r1 / 2 * r1_loss * config.d_reg_every + 0 * real_pred[0]).backward()

                    d_optim.step()

                loss_dict["r1"] = r1_loss
                requires_grad(generator, True)
                requires_grad(discriminator, False)

                noise = mixing_noise(batch_real_orin_faces.shape[0], config.latent, config.mixing, device)
                batch_fake_deid_faces, _ = generator(noise)

                fake_pred = discriminator(batch_fake_deid_faces)
                g_loss = g_nonsaturating_loss(fake_pred)

                loss_dict["g"] = g_loss

                generator.zero_grad()
                g_loss.backward()
                g_optim.step()

                g_regularize = epoch % config.g_reg_every == 0

                if g_regularize:
                    path_batch_size = max(1, batch_real_orin_faces.shape[0] // config.path_batch_shrink)
                    noise = mixing_noise(path_batch_size, config.latent, config.mixing, device)
                    fake_img, latents = generator(noise, return_latents=True)

                    path_loss, mean_path_length, path_lengths = g_path_regularize(
                        fake_img, latents, mean_path_length
                    )

                    generator.zero_grad()
                    weighted_path_loss = config.path_regularize * config.g_reg_every * path_loss

                    if config.path_batch_shrink:
                        weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

                    weighted_path_loss.backward()

                    g_optim.step()

                    mean_path_length_avg = (
                            reduce_sum(mean_path_length).item() / get_world_size()
                    )

                loss_dict["path"] = path_loss
                loss_dict["path_length"] = path_lengths.mean()

                accumulate(g_ema, g_module, accum)

                loss_reduced = reduce_loss_dict(loss_dict)

                d_loss_val = loss_reduced["d"].mean().item()
                g_loss_val = loss_reduced["g"].mean().item()
                r1_val = loss_reduced["r1"].mean().item()
                path_loss_val = loss_reduced["path"].mean().item()
                real_score_val = loss_reduced["real_score"].mean().item()
                fake_score_val = loss_reduced["fake_score"].mean().item()
                path_length_val = loss_reduced["path_length"].mean().item()
                loss_feature = 0

                progress_bar.set_description(f"Train {epoch + 1:>3}/{end_epoch:>3} G:{g_loss_val:.4f} D:{d_loss_val:.4f} R:{r1_val:.4f} F:{loss_feature:.4f}")

            epoch_loss_generator /= len(train_loader)
            epoch_loss_feature /= len(train_loader)
            epoch_loss_discriminator_real_fake /= len(train_loader)
            epoch_loss_discriminator_orin_deid /= len(train_loader)

        wandb.log({"Generator Loss": epoch_loss_generator, "Epoch": epoch})
        wandb.log({"Discriminator Real Fake Loss": epoch_loss_discriminator_real_fake, "Epoch": epoch})
        wandb.log({"Discriminator Orin Deid Loss": epoch_loss_discriminator_orin_deid, "Epoch": epoch})
        wandb.log({"Feature Similarity Loss": epoch_loss_feature, "Epoch": epoch})
        progress_bar.set_description(f"Train  {rank:^5} {epoch + 1:>3}/{end_epoch:>3} {epoch_loss_generator:.4f} {epoch_loss_feature:.4f}")

        # lr_scheduler_generator_orin2deid.step()
        # lr_scheduler_discriminator_real_fake.step()
        # lr_scheduler_discriminator_orin_deid.step()
        #
        # save_checkpoint(epoch, generator_orin2deid, optimizer_generator_orin2deid, epoch_loss_generator, os.path.join(model_save_dir, f"epoch_generator_orin2deid_{epoch + 1:06d}.pth"))
        # save_checkpoint(epoch, discriminator_real_fake, optimizer_discriminator_real_fake, epoch_loss_discriminator_real_fake, os.path.join(model_save_dir, f"epoch_discriminator_orin_{epoch + 1:06d}.pth"))
        # save_checkpoint(epoch, discriminator_orin_deid, optimizer_discriminator_orin_deid, epoch_loss_discriminator_orin_deid, os.path.join(model_save_dir, f"epoch_discriminator_deid_{epoch + 1:06d}.pth"))
        #
        # if epoch_loss_generator < best_loss_generator:
        #     best_loss_generator = epoch_loss_generator
        #     save_checkpoint(epoch, generator_orin2deid, optimizer_generator_orin2deid, epoch_loss_generator, os.path.join(model_save_dir, f"epoch_generator_orin2deid_best_{epoch + 1:06d}.pth"))
        #
        # if epoch_loss_discriminator_real_fake < best_loss_discriminator_real_fake:
        #     best_loss_discriminator_real_fake = epoch_loss_discriminator_real_fake
        #     save_checkpoint(epoch, discriminator_real_fake, optimizer_discriminator_real_fake, epoch_loss_discriminator_real_fake, os.path.join(model_save_dir, f"epoch_discriminator_deid_best_{epoch + 1:06d}.pth"))
        #
        # if epoch_loss_discriminator_orin_deid < best_loss_discriminator_orin_deid:
        #     best_loss_discriminator_orin_deid = epoch_loss_discriminator_orin_deid
        #     save_checkpoint(epoch, discriminator_orin_deid, optimizer_discriminator_orin_deid, epoch_loss_discriminator_orin_deid, os.path.join(model_save_dir, f"epoch_discriminator_deid_best_{epoch + 1:06d}.pth"))
        #
        #
        # if (epoch + 1) % valid_epoch == 0:
        #     epoch_inverted_images_dir = os.path.join(generated_images_dir, f"epoch-{epoch + 1}")
        #     if not os.path.exists(epoch_inverted_images_dir):
        #         os.makedirs(epoch_inverted_images_dir)
        #     valid_loss_generator, valid_loss_feature, valid_loss_discriminator_real_fake, valid_loss_discriminator_orin_deid = validate(
        #         valid_loader,
        #         generator_orin2deid,
        #         discriminator_real_fake,
        #         discriminator_orin_deid,
        #         feature_model,
        #         criterion_gan_loss,
        #         criterion_identity_loss,
        #         criterion_full_image_feature_loss,
        #         device,
        #         params,
        #         epoch_inverted_images_dir
        #     )
        #     print(f"Generator validation Loss at epoch {epoch + 1}: {valid_loss_generator}")
        #     print(f"Feature validation Loss at epoch {epoch + 1}: {valid_loss_feature}")
        #     print(f"Discriminator Real Fake validation Loss at epoch {epoch + 1}: {valid_loss_discriminator_real_fake}")
        #     print(f"Discriminator Orin Deid  validation Loss at epoch {epoch + 1}: {valid_loss_discriminator_orin_deid}")
        #     wandb.log({"Train Generator Loss": epoch_loss_generator, "Epoch": epoch})
        #     wandb.log({"Train Discriminator Real Fake Loss": epoch_loss_discriminator_real_fake, "Epoch": epoch})
        #     wandb.log({"Train Discriminator Orin Deid Loss": epoch_loss_discriminator_orin_deid, "Epoch": epoch})




def validate(
        valid_loader,
        generator_orin2deid,
        discriminator_real_fake,
        discriminator_orin_deid,
        feature_model,
        criterion_gan_loss,
        criterion_identity_loss,
        criterion_full_image_feature_loss,
        device,
        params,
        epoch_inverted_images_dir):
    generator_orin2deid.eval()
    discriminator_real_fake.eval()
    discriminator_orin_deid.eval()

    lambda_identity = params["lambda"]["lambda_identity"]
    lambda_full_image = params["lambda"]["lambda_full_image"]

    total_loss_generator = 0.0
    total_loss_feature = 0.0
    total_loss_discriminator_realfake = 0.0
    total_loss_discriminator_orindeid = 0.0

    with torch.no_grad():
        progress_bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        for batch_idx, (images_real_path, image_deid_path, label_path, images_orin, images_deid, boxes_list) in progress_bar:
            images_orin = images_orin.clone().to(device)
            images_deid = images_deid.clone().to(device)

            image_size = params["model"]["stylegan2"]["input_size"]

            batch_real_orin_faces = []
            batch_real_deid_faces = []
            batch_images_indices = []
            batch_faces_boxes = []

            for i, boxes in enumerate(boxes_list):
                real_orin_faces, faces_index, face_boxes = crop_face(images_orin, boxes, i)
                real_deid_faces, __ , __ = crop_face(images_deid, boxes, i)
                batch_real_orin_faces.extend(real_orin_faces)
                batch_real_deid_faces.extend(real_deid_faces)
                batch_images_indices.extend(faces_index)
                batch_faces_boxes.extend(face_boxes)

            if len(batch_real_orin_faces) > 0 and len(batch_real_deid_faces) > 0 :
                batch_real_orin_faces = [F.interpolate(face.unsqueeze(0), size=(image_size, image_size), mode='bilinear', align_corners=False).squeeze(0) for face in batch_real_orin_faces]
                batch_real_deid_faces = [F.interpolate(face.unsqueeze(0), size=(image_size, image_size), mode='bilinear', align_corners=False).squeeze(0) for face in batch_real_deid_faces]
                batch_real_orin_faces = torch.stack(batch_real_orin_faces)
                batch_real_deid_faces = torch.stack(batch_real_deid_faces)



                save_concat_images(images_orin, images_deid, deid_full_image, images_real_path, epoch_inverted_images_dir)
            progress_bar.set_description(f"Valid ")
    valid_loss_generator = total_loss_generator / len(valid_loader)
    valid_loss_feature = total_loss_feature / len(valid_loader)
    valid_loss_discriminator_realfake = total_loss_discriminator_realfake / len(valid_loader)
    valid_loss_discriminator_orindeid = total_loss_discriminator_orindeid / len(valid_loader)

    return valid_loss_generator, valid_loss_feature, valid_loss_discriminator_realfake, valid_loss_discriminator_orindeid

def save_concat_images(images_orin, images_deid, fake_deid_full_images, images_real_path, epoch_inverted_images_dir):
    concat_images = torch.cat((images_orin, images_deid, fake_deid_full_images), dim=3)
    for idx, concat_image in enumerate(concat_images):
        image_name = images_real_path[idx].split("/")[-1]
        save_gan_concat_text_image(concat_image, ["real_orin", "fake_orin", "fake_deid"], os.path.join(epoch_inverted_images_dir, image_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, default="config/config_stylegan2.yml", help="model parameters file path")

    option = parser.parse_known_args()[0]
    params = load_params_yml(option.config)["train"]

    train(params["device"], params)
