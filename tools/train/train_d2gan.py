import os
import shutil
import sys

import wandb
import datetime
import argparse
import itertools
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utility import logging
from utility.params import load_params_yml
from utility.model.model_file import load_checkpoint, save_checkpoint
from utility.model.region import crop_face, overlay_faces_on_image, save_gan_concat_text_image
from config.models import model_classes
from model.deid.dataset.dataset import GANFaceDetDataset
from model.deid.cyclegan.utils.buffer import ReplayBuffer
from model.deid.cyclegan.utils.lambdas import LambdaLR
from model.deid.cyclegan.models.discriminator import Discriminator
from model.deid.cyclegan.models.generator import GeneratorResNet
from model.deid.cyclegan.models.layers import weights_init_normal



def train(rank, params):
    if params["model"]["cyclegan"]["resume"]:
        tmp_start_time_stamp = params["model"]["cyclegan"]["pretrained_model_dir"].split("/")[-3].split("_")
        start_timestamp = f"{tmp_start_time_stamp[0]}_{tmp_start_time_stamp[1]}"
        wandb_run_name = f"{start_timestamp}-{params['model']['cyclegan']['model_name']}"
        wandb.init(project=params['wandb']['project_name'], entity=params['wandb']['account'], name=wandb_run_name, resume=True, id=wandb_run_name)
    else:
        start_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb_run_name = f"{start_timestamp}-{params['model']['cyclegan']['model_name']}"
        wandb.init(project=params['wandb']['project_name'], entity=params['wandb']['account'], name=wandb_run_name)

    batch_size = params["batch_size"]
    start_epoch = 0
    end_epoch = params["epoch"]
    valid_epoch = params["valid_epoch"]
    decay_epoch = params["decay_epoch"]
    learning_rate = params["learning_rate"]
    b1 = params["model"]["cyclegan"]["b1"]
    b2 = params["model"]["cyclegan"]["b2"]
    image_size = params["model"]["cyclegan"]["input_size"]
    channels = params["model"]["cyclegan"]["channels"]
    residual_block_number = params["model"]["cyclegan"]["residual_block_number"]
    lambda_cycle = params["lambda"]["lambda_cycle"]
    lambda_identity = params["lambda"]["lambda_identity"]
    lambda_full_image = params["lambda"]["lambda_full_image"]
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
    model_dir = os.path.join(save_dir, start_timestamp + "_" + params["model"]["cyclegan"]["model_name"])
    model_save_dir = os.path.join(model_dir, "weights")
    generated_images_dir = os.path.join(model_dir, 'generated_images')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(generated_images_dir):
        os.makedirs(generated_images_dir)

    # Data Loaders
    train_dataset = GANFaceDetDataset(train_image_real_dir, train_image_deid_dir, train_label_dir)
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

    criterion_gan_loss = torch.nn.BCEWithLogitsLoss()
    criterion_identity_loss = torch.nn.L1Loss()
    criterion_full_image_feature_loss = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    image_shape = (channels, image_size, image_size)

    # Model Initialization
    generator_orin2deid = GeneratorResNet(image_shape, residual_block_number)
    discriminator_real_fake = Discriminator(image_shape)
    discriminator_orin_deid = Discriminator(image_shape)

    # Move models to GPU if available
    if cuda_available:
        generator_orin2deid = generator_orin2deid.cuda()
        discriminator_real_fake = discriminator_real_fake.cuda()
        discriminator_orin_deid = discriminator_orin_deid.cuda()
        criterion_gan_loss.cuda()
        criterion_identity_loss.cuda()
        criterion_full_image_feature_loss.cuda()

    # Optimizers setup
    optimizer_generator_orin2deid = torch.optim.Adam(itertools.chain(generator_orin2deid.parameters(), generator_orin2deid.parameters()), lr=learning_rate, betas=(b1, b2))
    optimizer_discriminator_real_fake = torch.optim.Adam(discriminator_real_fake.parameters(), lr=learning_rate, betas=(b1, b2))
    optimizer_discriminator_orin_deid = torch.optim.Adam(discriminator_orin_deid.parameters(), lr=learning_rate, betas=(b1, b2))

    # Learning Rate Schedulers
    lr_scheduler_generator_orin2deid = torch.optim.lr_scheduler.LambdaLR(optimizer_generator_orin2deid, lr_lambda=LambdaLR(end_epoch, start_epoch, decay_epoch).step)
    lr_scheduler_discriminator_real_fake = torch.optim.lr_scheduler.LambdaLR(optimizer_discriminator_real_fake, lr_lambda=LambdaLR(end_epoch, start_epoch, decay_epoch).step)
    lr_scheduler_discriminator_orin_deid = torch.optim.lr_scheduler.LambdaLR(optimizer_discriminator_orin_deid, lr_lambda=LambdaLR(end_epoch, start_epoch, decay_epoch).step)

    # Model Loading or Weight Initialization
    if params["model"]["cyclegan"]["resume"]:
        # Load pretrained models
        pretrained_model_dir = params["model"]["cyclegan"]["pretrained_model_dir"]
        model_names = os.listdir(pretrained_model_dir)
        generator_orin2deid_model_name = sorted([item for item in model_names if "generator_orin2deid" in item])[-1]
        discriminator_real_fake_model_name = sorted([item for item in model_names if "discriminator_real_fake" in item])[-1]
        discriminator_orin_deid_model_name = sorted([item for item in model_names if "discriminator_deid" in item])[-1]
        generator_orin2deid, optimizer_generator_orin2deid, start_epoch, best_loss_generator = load_checkpoint(generator_orin2deid, optimizer_generator_orin2deid, os.path.join(pretrained_model_dir, generator_orin2deid_model_name))
        discriminator_real_fake, optimizer_discriminator_real_fake, start_epoch, best_loss_discriminator_real_fake = load_checkpoint(discriminator_real_fake, discriminator_real_fake, os.path.join(pretrained_model_dir, discriminator_real_fake_model_name))
        discriminator_orin_deid, optimizer_discriminator_orin_deid, start_epoch, best_loss_discriminator_orin_deid = load_checkpoint(discriminator_orin_deid, optimizer_discriminator_orin_deid, os.path.join(pretrained_model_dir, discriminator_orin_deid_model_name))
    else:
        # Weight initialization
        generator_orin2deid.apply(weights_init_normal)
        discriminator_real_fake.apply(weights_init_normal)
        discriminator_orin_deid.apply(weights_init_normal)
        start_epoch = 0
        best_loss_generator = float('inf')
        best_loss_discriminator_real_fake = float('inf')
        best_loss_discriminator_orin_deid = float('inf')

    wandb.watch(generator_orin2deid)
    wandb.watch(discriminator_real_fake)
    wandb.watch(discriminator_orin_deid)

    print(logging.i("Dataset Description"))
    print(logging.s(f"    training     : {len(train_dataset)}"))
    print(logging.s(f"    validation   : {len(valid_dataset)}"))
    print(logging.i("Parameters Description"))
    print(logging.s(f"    Device       : {params['device']}"))
    print(logging.s(f"    Batch size   : {params['batch_size']}"))
    print(logging.s(f"    Max epoch    : {params['epoch']}"))
    print(logging.s(f"    Learning rate: {learning_rate}"))
    print(logging.s(f"    Decay epoch: {decay_epoch}"))
    print(logging.s(f"    Valid epoch: {valid_epoch}"))
    print(logging.s(f"    Model:"))
    print(logging.s(f"    - Feature extractor: {params['model']['feature']['model_name']}"))
    print(logging.s(f"    - CycleGAN         : {params['model']['cyclegan']['model_name']}"))
    print(logging.s(f"       b1                   : {b1}"))
    print(logging.s(f"       b2                   : {b2}"))
    print(logging.s(f"       input size           : {image_size}"))
    print(logging.s(f"       channels             : {channels}"))
    print(logging.s(f"       # of residual block  : {residual_block_number}"))
    print(logging.s(f"Loss lambda:"))
    print(logging.s(f"- Lambda Cycle         : {lambda_cycle}"))
    print(logging.s(f"- Lambda Identity      : {lambda_identity}"))
    print(logging.s(f"- Lambda full image    : {lambda_full_image}"))
    print(logging.s(f"Task Information"))
    print(logging.s(f"- Task ID              : {start_timestamp + '_' + params['model']['cyclegan']['model_name']}"))
    print(logging.s(f"- Model directory      : {model_dir}"))
    print(logging.s(f"- Resume               : {params['model']['cyclegan']['resume']}"))
    if params['model']['cyclegan']['resume']:
        pretrained_model_dir = params["model"]["cyclegan"]["pretrained_model_dir"]
        model_names = os.listdir(pretrained_model_dir)
        generator_orin2deid_model_name = sorted([item for item in model_names if "generator_orin2deid" in item])[-1]
        discriminator_orin_model_name = sorted([item for item in model_names if "discriminator_orin" in item])[-1]
        discriminator_deid_model_name = sorted([item for item in model_names if "discriminator_deid" in item])[-1]
        print(logging.s(f"- Pretrained model directory: {params['model']['cyclegan']['pretrained_model_dir']}"))
        print(logging.s(f"   Generator Orin2Deid: {generator_orin2deid_model_name}"))
        print(logging.s(f"   Discriminator Orin: {discriminator_orin_model_name}"))
        print(logging.s(f"   Discriminator Deid: {discriminator_deid_model_name}"))

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
                batch_real_orin_faces = [F.interpolate(face.unsqueeze(0), size=(image_size, image_size), mode='bilinear', align_corners=False).squeeze(0) for face in batch_real_orin_faces]
                batch_real_deid_faces = [F.interpolate(face.unsqueeze(0), size=(image_size, image_size), mode='bilinear', align_corners=False).squeeze(0) for face in batch_real_deid_faces]
                batch_real_orin_faces = torch.stack(batch_real_orin_faces)
                batch_real_deid_faces = torch.stack(batch_real_deid_faces)

                real_labels = torch.ones((batch_real_orin_faces.size(0), *discriminator_real_fake.output_shape)).to(device)
                random_noise = torch.randn(batch_real_orin_faces.size()).to(batch_real_orin_faces.device)
                batch_fake_deid_faces = generator_orin2deid(random_noise)
                # -----------------------------------------
                #  Train Discriminator for Real vs. Fake
                # -----------------------------------------
                optimizer_discriminator_real_fake.zero_grad()
                preds_real = discriminator_real_fake(batch_real_orin_faces)
                preds_fake = discriminator_real_fake(batch_fake_deid_faces.detach())
                loss_real = criterion_gan_loss(preds_real, torch.ones_like(preds_real))
                loss_fake = criterion_gan_loss(preds_fake, torch.zeros_like(preds_fake))
                loss_discriminator_real_fake = (loss_real + loss_fake) / 2
                loss_discriminator_real_fake.backward()
                optimizer_discriminator_real_fake.step()
                # -----------------------------------------------------
                #  Train Discriminator for Original vs. De-identified
                # -----------------------------------------------------
                optimizer_discriminator_orin_deid.zero_grad()
                preds_orin = discriminator_orin_deid(batch_real_deid_faces)
                preds_deid = discriminator_orin_deid(batch_fake_deid_faces.detach())
                loss_orin = criterion_gan_loss(preds_orin, torch.ones_like(preds_orin))
                loss_deid = criterion_gan_loss(preds_deid, torch.zeros_like(preds_deid))
                loss_discriminator_orin_deid = (loss_orin + loss_deid) / 2
                loss_discriminator_orin_deid.backward()
                optimizer_discriminator_orin_deid.step()
                # ------------------
                #  Train Generator
                # ------------------
                optimizer_generator_orin2deid.zero_grad()
                generator_orin2deid.train()
                preds_fake = discriminator_orin_deid(batch_fake_deid_faces.detach())
                loss_generator = criterion_gan_loss(preds_fake, real_labels)
                loss_identity = criterion_identity_loss(batch_fake_deid_faces.detach(), batch_real_deid_faces)

                # Full image feature similarity loss
                deid_full_image = overlay_faces_on_image(images_orin, batch_fake_deid_faces, batch_faces_boxes, batch_images_indices)
                orin_features = feature_model(images_orin)
                deid_features = feature_model(deid_full_image)
                loss_feature = 1 - criterion_full_image_feature_loss(orin_features, deid_features).mean()

                # Total loss
                loss_generator = loss_generator + lambda_identity * loss_identity + lambda_full_image * loss_feature
                loss_generator.backward()
                optimizer_generator_orin2deid.step()

                # Log and Update progress bar
                epoch_loss_generator += loss_generator.item()
                epoch_loss_feature += loss_feature.item()
                epoch_loss_discriminator_real_fake += loss_discriminator_real_fake.item()
                epoch_loss_discriminator_orin_deid += loss_discriminator_orin_deid.item()

                progress_bar.set_description(
                    f"Train {epoch + 1:>3}/{end_epoch:>3} G:{loss_generator:.4f} D_RF:{loss_discriminator_real_fake:.4f} D_OD:{loss_discriminator_orin_deid:.4f} F:{loss_feature:.4f}")

            epoch_loss_generator /= len(train_loader)
            epoch_loss_feature /= len(train_loader)
            epoch_loss_discriminator_real_fake /= len(train_loader)
            epoch_loss_discriminator_orin_deid /= len(train_loader)

        wandb.log({"Generator Loss": epoch_loss_generator, "Epoch": epoch})
        wandb.log({"Discriminator Real Fake Loss": epoch_loss_discriminator_real_fake, "Epoch": epoch})
        wandb.log({"Discriminator Orin Deid Loss": epoch_loss_discriminator_orin_deid, "Epoch": epoch})
        wandb.log({"Feature Similarity Loss": epoch_loss_feature, "Epoch": epoch})
        progress_bar.set_description(f"Train  {rank:^5} {epoch + 1:>3}/{end_epoch:>3} {epoch_loss_generator:.4f} {epoch_loss_feature:.4f}")

        lr_scheduler_generator_orin2deid.step()
        lr_scheduler_discriminator_real_fake.step()
        lr_scheduler_discriminator_orin_deid.step()

        save_checkpoint(epoch, generator_orin2deid, optimizer_generator_orin2deid, epoch_loss_generator, os.path.join(model_save_dir, f"epoch_generator_orin2deid_{epoch + 1:06d}.pth"))
        save_checkpoint(epoch, discriminator_real_fake, optimizer_discriminator_real_fake, epoch_loss_discriminator_real_fake, os.path.join(model_save_dir, f"epoch_discriminator_orin_{epoch + 1:06d}.pth"))
        save_checkpoint(epoch, discriminator_orin_deid, optimizer_discriminator_orin_deid, epoch_loss_discriminator_orin_deid, os.path.join(model_save_dir, f"epoch_discriminator_deid_{epoch + 1:06d}.pth"))

        if epoch_loss_generator < best_loss_generator:
            best_loss_generator = epoch_loss_generator
            save_checkpoint(epoch, generator_orin2deid, optimizer_generator_orin2deid, epoch_loss_generator, os.path.join(model_save_dir, f"epoch_generator_orin2deid_best_{epoch + 1:06d}.pth"))

        if epoch_loss_discriminator_real_fake < best_loss_discriminator_real_fake:
            best_loss_discriminator_real_fake = epoch_loss_discriminator_real_fake
            save_checkpoint(epoch, discriminator_real_fake, optimizer_discriminator_real_fake, epoch_loss_discriminator_real_fake, os.path.join(model_save_dir, f"epoch_discriminator_deid_best_{epoch + 1:06d}.pth"))

        if epoch_loss_discriminator_orin_deid < best_loss_discriminator_orin_deid:
            best_loss_discriminator_orin_deid = epoch_loss_discriminator_orin_deid
            save_checkpoint(epoch, discriminator_orin_deid, optimizer_discriminator_orin_deid, epoch_loss_discriminator_orin_deid, os.path.join(model_save_dir, f"epoch_discriminator_deid_best_{epoch + 1:06d}.pth"))


        if (epoch + 1) % valid_epoch == 0:
            epoch_inverted_images_dir = os.path.join(generated_images_dir, f"epoch-{epoch + 1}")
            if not os.path.exists(epoch_inverted_images_dir):
                os.makedirs(epoch_inverted_images_dir)
            valid_loss_generator, valid_loss_feature, valid_loss_discriminator_real_fake, valid_loss_discriminator_orin_deid = validate(
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
                epoch_inverted_images_dir
            )
            print(f"Generator validation Loss at epoch {epoch + 1}: {valid_loss_generator}")
            print(f"Feature validation Loss at epoch {epoch + 1}: {valid_loss_feature}")
            print(f"Discriminator Real Fake validation Loss at epoch {epoch + 1}: {valid_loss_discriminator_real_fake}")
            print(f"Discriminator Orin Deid  validation Loss at epoch {epoch + 1}: {valid_loss_discriminator_orin_deid}")
            wandb.log({"Train Generator Loss": epoch_loss_generator, "Epoch": epoch})
            wandb.log({"Train Discriminator Real Fake Loss": epoch_loss_discriminator_real_fake, "Epoch": epoch})
            wandb.log({"Train Discriminator Orin Deid Loss": epoch_loss_discriminator_orin_deid, "Epoch": epoch})




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

    valid_fake_deid_buffer = ReplayBuffer()

    with torch.no_grad():
        progress_bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        for batch_idx, (images_real_path, image_deid_path, label_path, images_orin, images_deid, boxes_list) in progress_bar:
            images_orin = images_orin.clone().to(device)
            images_deid = images_deid.clone().to(device)

            image_size = params["model"]["cyclegan"]["input_size"]

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

                real_labels = torch.ones((batch_real_orin_faces.size(0), *discriminator_real_fake.output_shape)).to(device)
                random_noise = torch.randn(batch_real_orin_faces.size()).to(batch_real_orin_faces.device)
                batch_fake_deid_faces = generator_orin2deid(random_noise)
                # -----------------------------------------
                #  Train Discriminator for Real vs. Fake
                # -----------------------------------------
                preds_real = discriminator_real_fake(batch_real_orin_faces)
                preds_fake = discriminator_real_fake(batch_fake_deid_faces.detach())
                loss_real = criterion_gan_loss(preds_real, torch.ones_like(preds_real))
                loss_fake = criterion_gan_loss(preds_fake, torch.zeros_like(preds_fake))
                loss_discriminator_real_fake = (loss_real + loss_fake) / 2
                total_loss_discriminator_realfake += loss_discriminator_real_fake
                # -----------------------------------------------------
                #  Train Discriminator for Original vs. De-identified
                # -----------------------------------------------------
                preds_orin = discriminator_orin_deid(batch_real_deid_faces)
                preds_deid = discriminator_orin_deid(batch_fake_deid_faces.detach())
                loss_orin = criterion_gan_loss(preds_orin, torch.ones_like(preds_orin))
                loss_deid = criterion_gan_loss(preds_deid, torch.zeros_like(preds_deid))
                loss_discriminator_orin_deid = (loss_orin + loss_deid) / 2
                total_loss_discriminator_orindeid += loss_discriminator_orin_deid
                # ------------------
                #  Train Generator
                # ------------------
                generator_orin2deid.train()
                preds_fake = discriminator_orin_deid(batch_fake_deid_faces.detach())
                loss_generator = criterion_gan_loss(preds_fake, real_labels)
                loss_identity = criterion_identity_loss(batch_fake_deid_faces.detach(), batch_real_deid_faces)

                # Full image feature similarity loss
                deid_full_image = overlay_faces_on_image(images_orin, batch_fake_deid_faces, batch_faces_boxes, batch_images_indices)
                orin_features = feature_model(images_orin)
                deid_features = feature_model(deid_full_image)
                loss_feature = 1 - criterion_full_image_feature_loss(orin_features, deid_features).mean()
                total_loss_feature += loss_feature

                # Total loss
                loss_generator = loss_generator + lambda_identity * loss_identity + lambda_full_image * loss_feature
                total_loss_generator += loss_generator

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
    parser.add_argument("--config", type=str, default="config/params_gan.yml", help="model parameters file path")

    option = parser.parse_known_args()[0]
    params = load_params_yml(option.config)["train"]

    train(params["device"], params)
