import os
import shutil
import sys

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
from utility.model.region import crop_face, overlay_faces_on_image, save_gan_concat_text_image
from utility.model.model_file import save_checkpoint
from config.models import model_classes
from model.deid.dataset.dataset import GANFaceDetDataset
from model.deid.stylegan2.distributed import synchronize
from model.deid.stylegan2.model import Generator, Discriminator

from model.deid.stylegan2.non_leaking import AdaptiveAugment, augment
from model.deid.stylegan2.train import accumulate, reduce_loss_dict, mixing_noise, d_logistic_loss, d_r1_loss, \
    g_nonsaturating_loss, g_path_regularize, get_world_size, reduce_sum, data_sampler


def train(rank, params):
    start_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    batch_size = params["batch_size"]
    start_epoch = 0
    end_epoch = params["epoch"]
    valid_epoch = params["valid_epoch"]
    decay_epoch = params["decay_epoch"]
    lambda_generator = params["lambda"]["lambda_generator"]
    lambda_deid = params["lambda"]["lambda_deid"]
    lambda_full_image = params["lambda"]["lambda_full_image"]

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
    model_dir = os.path.join(save_dir, start_timestamp + "_" + params["model"]["stylegan2"]["model_name"])
    model_save_dir = os.path.join(model_dir, "weights")
    generated_images_dir = os.path.join(model_dir, 'generated_images')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(generated_images_dir):
        os.makedirs(generated_images_dir)

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

    # Model Initialization
    generator = Generator(config.input_size, config.latent, config.n_mlp, channel_multiplier=config.channel_multiplier).to(device)
    discriminator = Discriminator(config.input_size, channel_multiplier=config.channel_multiplier).to(device)
    generator_ema = Generator(config.input_size, config.latent, config.n_mlp, channel_multiplier=config.channel_multiplier).to(device)
    generator_ema.eval()
    accumulate(generator_ema, generator, 0)

    ratio_regularize_generator = config.g_reg_every / (config.g_reg_every + 1)
    ratio_regulaize_discriminator = config.d_reg_every / (config.d_reg_every + 1)

    # Optimizers setup
    optimizer_generator = torch.optim.Adam(
        generator.parameters(),
        lr=config.learning_rate * ratio_regularize_generator,
        betas=(0 ** ratio_regularize_generator, 0.99 ** ratio_regularize_generator),
    )
    optimizer_discriminator = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.learning_rate * ratio_regulaize_discriminator,
        betas=(0 ** ratio_regulaize_discriminator, 0.99 ** ratio_regulaize_discriminator),
    )

    if params["model"]["stylegan2"]["resume"]:
        state_dict = torch.load(params["model"]["stylegan2"]["pretrained_model_path"], map_location=lambda storage, loc: storage)
        state_dict = {k.replace("base.", ""): v for k, v in state_dict.items()}
        generator.load_state_dict(state_dict["g"], strict=False)
        discriminator.load_state_dict(state_dict["d"], strict=False)
        generator_ema.load_state_dict(state_dict["g_ema"], strict=False)
        optimizer_generator.load_state_dict(state_dict["g_optim"])
        optimizer_discriminator.load_state_dict(state_dict["d_optim"])

    if params["model"]["feature"]["model_name"] == "ResNet50":
        feature_model = model_classes["feature"][params["model"]["feature"]["model_name"]](backbone="TV_RESNET50", dims=512, pool_param=3).to(device)
        feature_model.load_state_dict(torch.load(params["model"]["feature"]["feature_weight_path"]))
    elif params["model"]["feature"]["model_name"] == "S2VC":
        feature_model = model_classes["feature"][params["model"]["feature"]["model_name"]]["RESNET"].get_model(512).to(device)
        state_dict = torch.load(params["model"]["feature"]["feature_weight_path"], map_location=device)
        state_dict = {k.replace("base.", ""): v for k, v in state_dict.items()}
        feature_model.load_state_dict(state_dict, strict=False)
    else:
        feature_model = model_classes["feature"][params["model"]["feature"]["model_name"]]().to(device)
        state_dict = torch.load(params["model"]["feature"]["feature_weight_path"], map_location=device)
        state_dict = {k.replace("base.", ""): v for k, v in state_dict.items()}
        feature_model.load_state_dict(state_dict, strict=False)
    feature_model.eval()

    criterion_identity_loss = torch.nn.L1Loss()
    criterion_identity_loss.cuda()
    criterion_full_image_feature_loss = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    criterion_full_image_feature_loss.cuda()

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
    print(logging.s(f"    - GAN model name : {config.model_name}"))
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
    print(logging.s(f"Loss lambda:"))
    print(logging.s(f"- Lambda Generator     : {lambda_generator}"))
    print(logging.s(f"- Lambda Face De-id    : {lambda_deid}"))
    print(logging.s(f"- Lambda full image    : {lambda_full_image}"))
    print(logging.s(f"Task Information"))
    print(logging.s(f"- Task ID              : {start_timestamp + '_' + params['model']['stylegan2']['model_name']}"))
    print(logging.s(f"- Model directory      : {model_dir}"))
    print(logging.s(f"- Resume               : {params['model']['stylegan2']['resume']}"))
    if params['model']['stylegan2']['resume']:
        pretrained_model_path = params["model"]["stylegan2"]["pretrained_model_path"]
        print(logging.s(f"   Pretrained model path: {pretrained_model_path}"))

    mean_path_length = 0
    r1_loss = torch.tensor(0.0, device=device)
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    accum = 0.5 ** (32 / (10 * 1000))

    terminal_size = shutil.get_terminal_size().columns
    print("Status   Dev   Epoch  G     D      r1     Path   mPath  Augment")
    print("-" * terminal_size)

    for epoch in range(start_epoch, end_epoch):
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (images_real_path, image_deid_path, label_path, images_orin, images_deid, boxes_list) in progress_bar:
            images_orin = images_orin.clone().to(device)
            images_deid = images_deid.clone().to(device)

            batch_real_orin_faces = []
            batch_real_deid_faces = []
            batch_images_indices = []
            batch_faces_boxes = []

            for i, boxes in enumerate(boxes_list):
                real_orin_faces, faces_index, face_boxes = crop_face(images_orin, boxes, i, slice=True)
                real_deid_faces, __ , __ = crop_face(images_deid, boxes, i, slice=True)
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
                fake_pred = discriminator(batch_fake_deid_faces)
                real_pred = discriminator(batch_real_orin_faces)

                deid_full_image = overlay_faces_on_image(images_orin, batch_fake_deid_faces, batch_faces_boxes, batch_images_indices)
                orin_features = feature_model(images_orin)
                deid_features = feature_model(deid_full_image)
                loss_feature = 1 - criterion_full_image_feature_loss(orin_features, deid_features).mean()

                loss_discriminator = d_logistic_loss(real_pred, fake_pred)

                loss_deid_face_matching = F.l1_loss(batch_real_deid_faces, batch_fake_deid_faces)

                loss_dict["d"] = loss_discriminator
                loss_dict["real_score"] = real_pred.mean()
                loss_dict["fake_score"] = fake_pred.mean()

                discriminator.zero_grad()
                loss_discriminator.backward()
                optimizer_discriminator.step()

                regularize_discriminator = epoch % config.d_reg_every == 0

                if regularize_discriminator:
                    batch_real_orin_faces.requires_grad = True
                    batch_real_orin_faces_aug = batch_real_orin_faces

                    real_pred = discriminator(batch_real_orin_faces_aug)
                    r1_loss = d_r1_loss(real_pred, batch_real_orin_faces)

                    discriminator.zero_grad()
                    (config.r1 / 2 * r1_loss * config.d_reg_every + 0 * real_pred[0]).backward()

                    optimizer_discriminator.step()

                loss_dict["r1"] = r1_loss
                requires_grad(generator, True)
                requires_grad(discriminator, False)

                noise = mixing_noise(batch_real_orin_faces.shape[0], config.latent, config.mixing, device)
                batch_fake_deid_faces, _ = generator(noise)

                fake_pred = discriminator(batch_fake_deid_faces)
                loss_generator = g_nonsaturating_loss(fake_pred)
                loss_generator = lambda_generator * loss_generator + lambda_deid * loss_deid_face_matching + lambda_full_image * loss_feature

                loss_dict["g"] = loss_generator

                generator.zero_grad()
                loss_generator.backward()
                optimizer_generator.step()

                ratio_regularize_generator = epoch % config.g_reg_every == 0

                if ratio_regularize_generator:
                    path_batch_size = max(1, batch_real_orin_faces.shape[0] // config.path_batch_shrink)
                    noise = mixing_noise(path_batch_size, config.latent, config.mixing, device)
                    fake_img, latents = generator(noise, return_latents=True)

                    path_loss, mean_path_length, path_lengths = g_path_regularize(fake_img, latents, mean_path_length)

                    generator.zero_grad()
                    loss_weighted_path = config.path_regularize * config.g_reg_every * path_loss

                    if config.path_batch_shrink:
                        loss_weighted_path += 0 * fake_img[0, 0, 0, 0]

                    loss_weighted_path.backward()
                    optimizer_generator.step()

                    mean_path_length_avg = (reduce_sum(mean_path_length).item() / get_world_size())

                loss_dict["path"] = path_loss
                loss_dict["path_length"] = path_lengths.mean()

                accumulate(generator_ema, generator, accum)

                loss_reduced = reduce_loss_dict(loss_dict)

                loss_discriminator_valid = loss_reduced["d"].mean().item()
                loss_generator_valid = loss_reduced["g"].mean().item()
                loss_r1_valid = loss_reduced["r1"].mean().item()
                path_loss_val = loss_reduced["path"].mean().item()
                real_score_val = loss_reduced["real_score"].mean().item()
                fake_score_val = loss_reduced["fake_score"].mean().item()
                path_length_val = loss_reduced["path_length"].mean().item()

                progress_bar.set_description(
                    (
                        f"Train  {rank:^5} {epoch + 1:>3}/{end_epoch:>3} "
                        f"{loss_discriminator_valid:.4f} {loss_generator_valid:.4f} {loss_r1_valid:.4f} "
                        f"{path_loss_val:.4f} {mean_path_length_avg:.4f} "
                    )
                )
                save_checkpoint(epoch, generator, optimizer_generator, loss_generator_valid, os.path.join(model_save_dir, f"epoch_generator_{epoch + 1:06d}.pth"))
                save_checkpoint(epoch, discriminator, optimizer_discriminator, loss_discriminator_valid, os.path.join(model_save_dir, f"epoch_discriminator_{epoch + 1:06d}.pth"))

        if (epoch + 1) % valid_epoch == 0:
            epoch_inverted_images_dir = os.path.join(generated_images_dir, f"epoch-{epoch + 1}")
            if not os.path.exists(epoch_inverted_images_dir):
                os.makedirs(epoch_inverted_images_dir)
            val_loss = validate(valid_loader, generator, discriminator, feature_model, criterion_full_image_feature_loss, device, config, epoch_inverted_images_dir)
            print(f"Validation Losses - D: {val_loss['d']:.4f}, G: {val_loss['g']:.4f}, F: {val_loss['f']:.4f}")


def validate(valid_loader, generator, discriminator, feature_model, criterion_full_image_feature_loss, device, config, epoch_inverted_images_dir):
    generator.eval()
    discriminator.eval()

    loss_dict_val = {"d": 0.0, "g": 0.0, "f": 0.0}
    total_images = 0
    total_samples = 0

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
                batch_real_orin_faces = torch.stack(batch_real_orin_faces)

                noise = mixing_noise(batch_real_orin_faces.shape[0], config.latent, config.mixing, device)
                batch_fake_deid_faces, _ = generator(noise)
                fake_pred = discriminator(batch_fake_deid_faces)
                real_pred = discriminator(batch_real_orin_faces)

                loss_discriminator = d_logistic_loss(real_pred, fake_pred)
                loss_generator = g_nonsaturating_loss(fake_pred)

                deid_full_image = overlay_faces_on_image(images_orin, batch_fake_deid_faces, batch_faces_boxes, batch_images_indices)
                orin_features = feature_model(images_orin)
                deid_features = feature_model(deid_full_image)
                loss_feature = 1 - criterion_full_image_feature_loss(orin_features, deid_features).mean()

                loss_dict_val["d"] += loss_discriminator.item() * batch_real_orin_faces.shape[0]
                loss_dict_val["g"] += loss_generator.item() * batch_real_orin_faces.shape[0]
                loss_dict_val["f"] += loss_feature
                total_images += images_orin.shape[0]
                total_samples += batch_real_orin_faces.shape[0]

                save_concat_images(images_orin, images_deid, deid_full_image, images_real_path, epoch_inverted_images_dir)

    loss_dict_val["d"] /= total_samples
    loss_dict_val["g"] /= total_samples
    loss_dict_val["f"] /= total_images

    return loss_dict_val

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
