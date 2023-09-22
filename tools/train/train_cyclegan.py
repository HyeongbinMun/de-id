import os
import shutil
import sys
import wandb
import datetime
import argparse
import itertools
import torchvision

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.models import model_classes
from utility.params import load_params_yml
from utility.model.model_file import load_checkpoint, save_checkpoint
from model.deid.dataset.dataset import FaceDetDataset, CycleGANFaceDetDataset
from model.deid.cyclegan.utils.buffer import ReplayBuffer
from model.deid.cyclegan.utils.lambdas import LambdaLR
from model.deid.cyclegan.models.discriminator import Discriminator
from model.deid.cyclegan.models.generator import GeneratorResNet
from model.deid.cyclegan.models.layers import weights_init_normal


def crop_face(images, boxes, index):
    faces = []
    image_indices = []
    valid_boxes = []

    for b, box in enumerate(boxes):
        image_w, image_h = images.shape[3], images.shape[2]
        _, x_center, y_center, w, h = box
        x1 = int((x_center - w / 2) * image_w)
        y1 = int((y_center - h / 2) * image_h)
        x2 = int((x_center + w / 2) * image_w)
        y2 = int((y_center + h / 2) * image_h)

        if (x2 - x1) > 10 and (y2 - y1) > 10 and x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
            face = images[index, :, y1:y2, x1:x2].clone()
            faces.append(face)
            image_indices.append(index)
            valid_boxes.append(box)

    return faces, image_indices, valid_boxes


def overlay_faces_on_image(original_images, fake_faces, valid_boxes, indices):
    overlaid_images = original_images.clone()

    for face, box, idx in zip(fake_faces, valid_boxes, indices):
        image_w, image_h = original_images.shape[3], original_images.shape[2]
        _, x_center, y_center, w, h = box
        x1 = int((x_center - w / 2) * image_w)
        y1 = int((y_center - h / 2) * image_h)
        x2 = int((x_center + w / 2) * image_w)
        y2 = int((y_center + h / 2) * image_h)

        face_resized = F.interpolate(face.unsqueeze(0), size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False).squeeze(0)
        h_face, w_face = face_resized.shape[1], face_resized.shape[2]

        y1 = max(0, y1)
        x1 = max(0, x1)
        y2 = min(image_h, y1 + h_face)
        x2 = min(image_w, x1 + w_face)

        face_resized = face_resized[:, 0:y2 - y1, 0:x2 - x1]
        overlaid_images[idx, :, y1:y2, x1:x2] = face_resized

    return overlaid_images



def train(rank, params):
    if params["model"]["cyclegan"]["resume"]:
        tmp_start_time_stamp = params["model"]["cyclegan"]["pretrained_model_path"].split("/")[-3].split("_")
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
    learning_rate = params["learning_rate"]
    b1 = params["b1"]
    b2 = params["b2"]
    decay_epoch = params["decay_epoch"]
    image_size = params["image_size"]
    channels = params["channels"]
    residual_block_number = params["residual_block_number"]
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
    train_dataset = CycleGANFaceDetDataset(train_image_real_dir, train_image_deid_dir, train_label_dir)
    valid_dataset = CycleGANFaceDetDataset(valid_image_real_dir, valid_image_deid_dir, valid_label_dir)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=1,
                              collate_fn=CycleGANFaceDetDataset.collate_fn)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=1,
                              collate_fn=CycleGANFaceDetDataset.collate_fn)

    if params["model"]["feature"]["model_name"] == "ResNet50":
        feature_model = model_classes["feature"][params["model"]["feature"]["model_name"]](backbone="TV_RESNET50", dims=512, pool_param=3).to(device)
        feature_model.load_state_dict(torch.load(params["model"]["feature"]["feature_weight_path"]))
    else:
        feature_model = model_classes["feature"][params["model"]["feature"]["model_name"]]().to(device)
        state_dict = torch.load(params["model"]["feature"]["feature_weight_path"], map_location=device)
        state_dict = {k.replace("base.", ""): v for k, v in state_dict.items()}
        feature_model.load_state_dict(state_dict, strict=False)
    feature_model.eval()

    criterion_gan_loss = torch.nn.MSELoss()
    criterion_cycle_loss = torch.nn.L1Loss()
    criterion_identity_loss = torch.nn.L1Loss()
    criterion_full_image_feature_loss = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    image_shape = (channels, image_size, image_size)

    # Model Initialization
    generator_orin2deid = GeneratorResNet(image_shape, residual_block_number)
    generator_deid2orin = GeneratorResNet(image_shape, residual_block_number)
    discriminator_orin = Discriminator(image_shape)
    discriminator_deid = Discriminator(image_shape)

    # Move models to GPU if available
    if cuda_available:
        generator_orin2deid = generator_orin2deid.cuda()
        generator_deid2orin = generator_deid2orin.cuda()
        discriminator_orin = discriminator_orin.cuda()
        discriminator_deid = discriminator_deid.cuda()
        criterion_gan_loss.cuda()
        criterion_cycle_loss.cuda()
        criterion_identity_loss.cuda()
        criterion_full_image_feature_loss.cuda()

    # Optimizers setup
    optimizer_generator_orin2deid = torch.optim.Adam(itertools.chain(generator_orin2deid.parameters(), generator_deid2orin.parameters()), lr=learning_rate, betas=(b1, b2))
    optimizer_generator_deid2orin = torch.optim.Adam(itertools.chain(generator_deid2orin.parameters(), generator_orin2deid.parameters()), lr=learning_rate, betas=(b1, b2))
    optimizer_discriminator_orin = torch.optim.Adam(discriminator_orin.parameters(), lr=learning_rate, betas=(b1, b2))
    optimizer_discriminator_deid = torch.optim.Adam(discriminator_deid.parameters(), lr=learning_rate, betas=(b1, b2))

    # Learning Rate Schedulers
    lr_scheduler_generator_orin2deid = torch.optim.lr_scheduler.LambdaLR(optimizer_generator_orin2deid, lr_lambda=LambdaLR(end_epoch, start_epoch, decay_epoch).step)
    lr_scheduler_generator_deid2orin = torch.optim.lr_scheduler.LambdaLR(optimizer_generator_deid2orin, lr_lambda=LambdaLR(end_epoch, start_epoch, decay_epoch).step)
    lr_scheduler_discriminator_orin = torch.optim.lr_scheduler.LambdaLR(optimizer_discriminator_orin, lr_lambda=LambdaLR(end_epoch, start_epoch, decay_epoch).step)
    lr_scheduler_discriminator_deid = torch.optim.lr_scheduler.LambdaLR(optimizer_discriminator_deid, lr_lambda=LambdaLR(end_epoch, start_epoch, decay_epoch).step)

    # Buffer for Generated Samples
    fake_orin_buffer = ReplayBuffer()
    fake_deid_buffer = ReplayBuffer()

    # Model Loading or Weight Initialization
    if params["model"]["cyclegan"]["resume"]:
        # Load pretrained models
        pretrained_model_dir = params["model"]["cyclegan"]["pretrained_model_dir"]
        model_names = os.listdir(pretrained_model_dir)
        generator_orin2deid_model_names = sorted([item for item in model_names if "generator_orin2deid" in item]).reverse()
        generator_deid2orin_model_names = sorted([item for item in model_names if "generator_deid2orin" in item]).reverse()
        discriminator_orin_model_names = sorted([item for item in model_names if "discriminator_orin" in item]).reverse()
        discriminator_deid_model_names = sorted([item for item in model_names if "discriminator_deid" in item]).reverse()
        generator_orin2deid, optimizer_generator_orin2deid, start_epoch, best_loss_generator = load_checkpoint(generator_orin2deid, optimizer_generator_orin2deid, generator_orin2deid_model_names[0])
        generator_deid2orin, optimizer_generator_deid2orin, start_epoch, best_loss_generator = load_checkpoint(generator_orin2deid, optimizer_generator_deid2orin, generator_deid2orin_model_names[0])
        discriminator_orin, optimizer_discriminator_orin, start_epoch, best_loss_discriminator_orin = load_checkpoint(discriminator_orin, optimizer_discriminator_orin, discriminator_orin_model_names[0])
        discriminator_deid, optimizer_discriminator_deid, start_epoch, best_loss_discriminator_deid = load_checkpoint(discriminator_deid, optimizer_discriminator_deid, discriminator_deid_model_names[0])
    else:
        # Weight initialization
        generator_orin2deid.apply(weights_init_normal)
        generator_deid2orin.apply(weights_init_normal)
        discriminator_orin.apply(weights_init_normal)
        discriminator_deid.apply(weights_init_normal)
        start_epoch = 0
        best_loss_generator = float('inf')
        best_loss_discriminator_orin = float('inf')
        best_loss_discriminator_deid = float('inf')


    wandb.watch(generator_orin2deid)
    wandb.watch(generator_deid2orin)
    wandb.watch(discriminator_orin)
    wandb.watch(discriminator_deid)

    terminal_size = shutil.get_terminal_size().columns
    print("Status   Dev   Epoch  GAN   D_orin D_deid feature")
    print("-" * terminal_size)

    for epoch in range(start_epoch, end_epoch):
        epoch_loss_generator = 0.0
        epoch_loss_generator_orin2deid = 0.0
        epoch_loss_generator_deid2orin = 0.0
        epoch_loss_discriminator_orin = 0.0
        epoch_loss_discriminator_deid = 0.0
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

            batch_real_orin_faces = [F.interpolate(face.unsqueeze(0), size=(image_size, image_size), mode='bilinear', align_corners=False).squeeze(0) for face in batch_real_orin_faces]
            batch_real_deid_faces = [F.interpolate(face.unsqueeze(0), size=(image_size, image_size), mode='bilinear', align_corners=False).squeeze(0) for face in batch_real_deid_faces]
            batch_real_orin_faces = torch.stack(batch_real_orin_faces)
            batch_real_deid_faces = torch.stack(batch_real_deid_faces)

            valid = torch.ones((batch_real_orin_faces.size(0), *discriminator_orin.output_shape)).to(device)
            fake = torch.zeros((batch_real_orin_faces.size(0), *discriminator_orin.output_shape)).to(device)

            # ------------------
            #  Train Generators
            # ------------------
            generator_orin2deid.train()
            generator_deid2orin.train()

            optimizer_generator_orin2deid.zero_grad()
            optimizer_discriminator_deid.zero_grad()

            batch_fake_deid_faces = generator_orin2deid(batch_real_orin_faces)
            batch_fake_orin_faces = generator_deid2orin(batch_real_deid_faces)

            # Identity loss
            loss_identity_real = criterion_identity_loss(batch_fake_orin_faces, batch_real_orin_faces)
            loss_identity_deid = criterion_identity_loss(batch_fake_deid_faces, batch_real_deid_faces)
            loss_identity = (loss_identity_real + loss_identity_deid) / 2

            # GAN loss
            loss_gan_orin2deid = criterion_gan_loss(discriminator_deid(batch_fake_deid_faces), valid)  # MSE loss
            loss_gan_deid2orin = criterion_gan_loss(discriminator_orin(batch_fake_orin_faces), valid)
            loss_gan = (loss_gan_orin2deid + loss_gan_deid2orin) / 2

            # Cycle loss
            batch_reco_orin_faces = generator_deid2orin(batch_fake_deid_faces)
            batch_reco_deid_faces = generator_orin2deid(batch_fake_deid_faces)
            loss_cycle_orin = criterion_cycle_loss(batch_reco_orin_faces, batch_real_orin_faces)
            loss_cycle_deid = criterion_cycle_loss(batch_reco_deid_faces, batch_real_deid_faces)
            loss_cycle = (loss_cycle_orin + loss_cycle_deid) / 2

            # Full image feature similarity loss
            deid_full_image = overlay_faces_on_image(images_orin, batch_fake_deid_faces, batch_faces_boxes, batch_images_indices)
            orin_features = feature_model(images_orin)
            deid_features = feature_model(deid_full_image)
            loss_feature = 1 - criterion_full_image_feature_loss(orin_features, deid_features).mean()

            # Total loss
            loss_gerator = loss_gan + lambda_cycle * loss_cycle + lambda_identity * loss_identity + lambda_full_image * loss_feature
            loss_gerator.backward()
            optimizer_generator_orin2deid.step()
            optimizer_generator_deid2orin.step()

            # ----------------------------
            #  Train Discriminator Origin
            # ----------------------------
            optimizer_discriminator_orin.zero_grad()

            loss_real = criterion_gan_loss(discriminator_orin(batch_real_orin_faces), valid)
            batch_fake_orin = fake_orin_buffer.push_and_pop(batch_fake_orin_faces)
            loss_fake = criterion_gan_loss(discriminator_orin(batch_fake_orin.detach()), fake)
            loss_discriminator_orin = (loss_real + loss_fake) / 2

            loss_discriminator_orin.backward()
            optimizer_discriminator_orin.step()

            # ----------------------------
            #  Train Discriminator De-id
            # ----------------------------
            optimizer_discriminator_deid.zero_grad()

            loss_real = criterion_gan_loss(discriminator_deid(batch_real_deid_faces), valid)
            batch_fake_deid = fake_deid_buffer.push_and_pop(batch_fake_deid_faces)
            loss_fake = criterion_gan_loss(discriminator_deid(batch_fake_deid.detach()), fake)
            loss_discriminator_deid = (loss_real + loss_fake) / 2

            loss_discriminator_deid.backward()
            optimizer_discriminator_deid.step()

            epoch_loss_generator += loss_gerator.item()
            epoch_loss_generator_orin2deid += loss_gan_orin2deid.item()
            epoch_loss_generator_deid2orin += loss_gan_deid2orin.item()
            epoch_loss_discriminator_orin += loss_discriminator_orin.item()
            epoch_loss_discriminator_deid += loss_discriminator_deid.item()

            progress_bar.set_description(f"Train  {rank:^5} {epoch+1:>3}/{end_epoch:>3} {loss_gerator:.4f} {loss_discriminator_orin:.4f} {loss_discriminator_deid:.4f} {loss_feature:.4f})")

            epoch_loss_generator /= len(train_loader)
            epoch_loss_discriminator_orin /= len(train_loader)
            epoch_loss_discriminator_deid /= len(train_loader)

            if epoch_loss_generator < best_loss_generator:
                best_loss_generator = epoch_loss_generator
                save_checkpoint(epoch, generator_orin2deid, optimizer_generator_orin2deid, epoch_loss_generator_orin2deid, os.path.join(model_save_dir, f"epoch_generator_orin2deid_{epoch + 1:06d}.pth"))
                save_checkpoint(epoch, generator_deid2orin, optimizer_generator_deid2orin, epoch_loss_generator_deid2orin, os.path.join(model_save_dir, f"epoch_generator_deid2orin_{epoch + 1:06d}.pth"))

            if epoch_loss_discriminator_orin < best_loss_discriminator_orin:
                best_loss_discriminator_orin = epoch_loss_discriminator_orin
                save_checkpoint(epoch, discriminator_orin, optimizer_discriminator_orin, epoch_loss_discriminator_orin, os.path.join(model_save_dir, f"epoch_discriminator_orin_{epoch + 1:06d}.pth"))

            if epoch_loss_discriminator_deid < best_loss_discriminator_deid:
                best_loss_discriminator_deid = epoch_loss_discriminator_deid
                save_checkpoint(epoch, discriminator_deid, optimizer_discriminator_deid, epoch_loss_discriminator_deid, os.path.join(model_save_dir, f"epoch_discriminator_deid_{epoch + 1:06d}.pth"))

            if (epoch + 1) % valid_epoch == 0:
                epoch_inverted_images_dir = os.path.join(generated_images_dir, f"epoch-{epoch + 1}")
                if not os.path.exists(epoch_inverted_images_dir):
                    os.makedirs(epoch_inverted_images_dir)
                valid_loss_generator, valid_loss_feature, valid_loss_discriminator_orin, valid_loss_discriminator_deid = validate(
                    valid_loader,
                    generator_orin2deid,
                    generator_deid2orin,
                    discriminator_orin,
                    discriminator_deid,
                    feature_model,
                    criterion_gan_loss,
                    criterion_cycle_loss,
                    criterion_identity_loss,
                    criterion_full_image_feature_loss,
                    device,
                    params,
                    epoch_inverted_images_dir
                )
                print(f"Generator validation Loss at epoch {epoch + 1}: {valid_loss_generator}")
                print(f"Feature validation Loss at epoch {epoch + 1}: {valid_loss_feature}")
                print(f"Discriminator Origin validation Loss at epoch {epoch + 1}: {valid_loss_discriminator_orin}")
                print(f"Discriminator De-id  validation Loss at epoch {epoch + 1}: {valid_loss_discriminator_deid}")
            wandb.log({"Train Generator Loss": epoch_loss_generator, "Epoch": epoch})
            wandb.log({"Train Discriminator Orin Loss": epoch_loss_discriminator_orin, "Epoch": epoch})
            wandb.log({"Train Discriminator Deid Loss": epoch_loss_discriminator_deid, "Epoch": epoch})

        lr_scheduler_generator_orin2deid.step()
        lr_scheduler_generator_deid2orin.step()
        lr_scheduler_discriminator_orin.step()
        lr_scheduler_discriminator_deid.step()


def validate(
        valid_loader,
        generator_orin2deid,
        generator_deid2orin,
        discriminator_orin,
        discriminator_deid,
        feature_model,
        criterion_gan_loss,
        criterion_cycle_loss,
        criterion_identity_loss,
        criterion_full_image_feature_loss,
        device,
        params,
        epoch_inverted_images_dir):
    generator_orin2deid.eval()
    generator_deid2orin.eval()
    discriminator_orin.eval()
    discriminator_deid.eval()

    lambda_cycle = params["lambda"]["lambda_cycle"]
    lambda_identity = params["lambda"]["lambda_identity"]
    lambda_full_image = params["lambda"]["lambda_full_image"]

    total_loss_generator = 0.0
    total_loss_feature = 0.0
    total_loss_discriminator_orin = 0.0
    total_loss_discriminator_deid = 0.0

    valid_fake_orin_buffer = ReplayBuffer()
    valid_fake_deid_buffer = ReplayBuffer()

    with torch.no_grad():
        for images_real_path, image_deid_path, label_path, images_orin, images_deid, boxes_list in valid_loader:
            images_orin = images_orin.clone().to(device)
            images_deid = images_deid.clone().to(device)

            image_size = params["image_size"]

            batch_real_orin_faces = []
            batch_real_deid_faces = []
            batch_images_indices = []
            batch_faces_boxes = []

            for i, boxes in enumerate(boxes_list):
                real_orin_faces, faces_index, face_boxes = crop_face(images_orin, boxes, i)
                real_deid_faces, __, __ = crop_face(images_deid, boxes, i)
                batch_real_orin_faces.extend(real_orin_faces)
                batch_real_deid_faces.extend(real_deid_faces)
                batch_images_indices.extend(faces_index)
                batch_faces_boxes.extend(face_boxes)

            batch_real_orin_faces = [F.interpolate(face.unsqueeze(0), size=(image_size, image_size), mode='bilinear', align_corners=False).squeeze(0) for face in batch_real_orin_faces]
            batch_real_deid_faces = [F.interpolate(face.unsqueeze(0), size=(image_size, image_size), mode='bilinear', align_corners=False).squeeze(0) for face in batch_real_deid_faces]
            batch_real_orin_faces = torch.stack(batch_real_orin_faces)
            batch_real_deid_faces = torch.stack(batch_real_deid_faces)

            valid = torch.ones((batch_real_orin_faces.size(0), *discriminator_orin.output_shape)).to(device)
            fake = torch.zeros((batch_real_orin_faces.size(0), *discriminator_orin.output_shape)).to(device)

            batch_fake_deid_faces = generator_orin2deid(batch_real_orin_faces)
            batch_fake_orin_faces = generator_deid2orin(batch_real_deid_faces)

            loss_identity_real = criterion_identity_loss(batch_fake_orin_faces, batch_real_orin_faces)
            loss_identity_deid = criterion_identity_loss(batch_fake_deid_faces, batch_real_deid_faces)
            loss_identity = (loss_identity_real + loss_identity_deid) / 2

            loss_gan_orin2deid = criterion_gan_loss(discriminator_deid(batch_fake_deid_faces), valid)  # MSE loss
            loss_gan_deid2orin = criterion_gan_loss(discriminator_orin(batch_fake_orin_faces), valid)
            loss_gan = (loss_gan_orin2deid + loss_gan_deid2orin) / 2

            batch_reco_orin_faces = generator_deid2orin(batch_fake_deid_faces)
            batch_reco_deid_faces = generator_orin2deid(batch_fake_deid_faces)
            loss_cycle_orin = criterion_cycle_loss(batch_reco_orin_faces, batch_real_orin_faces)
            loss_cycle_deid = criterion_cycle_loss(batch_reco_deid_faces, batch_real_deid_faces)
            loss_cycle = (loss_cycle_orin + loss_cycle_deid) / 2

            deid_full_image = overlay_faces_on_image(images_orin, batch_fake_deid_faces, batch_faces_boxes, batch_images_indices)
            torchvision.utils.save_image(deid_full_image, os.path.join(epoch_inverted_images_dir, images_real_path), normalize=True)
            orin_features = feature_model(images_orin)
            deid_features = feature_model(deid_full_image)
            loss_feature = 1 - criterion_full_image_feature_loss(orin_features, deid_features).mean()
            total_loss_feature += loss_feature

            loss_gerator = loss_gan + lambda_cycle * loss_cycle + lambda_identity * loss_identity + lambda_full_image * loss_feature
            total_loss_generator += loss_gerator

            loss_real = criterion_gan_loss(discriminator_orin(batch_real_orin_faces), valid)
            batch_fake_orin = valid_fake_orin_buffer.push_and_pop(batch_fake_orin_faces)
            loss_fake = criterion_gan_loss(discriminator_orin(batch_fake_orin.detach()), fake)
            loss_discriminator_orin = (loss_real + loss_fake) / 2
            total_loss_discriminator_orin += loss_discriminator_orin

            loss_real = criterion_gan_loss(discriminator_deid(batch_real_deid_faces), valid)
            batch_fake_deid = valid_fake_deid_buffer.push_and_pop(batch_fake_deid_faces)
            loss_fake = criterion_gan_loss(discriminator_deid(batch_fake_deid.detach()), fake)
            loss_discriminator_deid = (loss_real + loss_fake) / 2
            total_loss_discriminator_deid += loss_discriminator_deid

    valid_loss_generator = total_loss_generator / len(valid_loader)
    valid_loss_feature = total_loss_feature / len(valid_loader)
    valid_loss_discriminator_orin = total_loss_discriminator_orin / len(valid_loader)
    valid_loss_discriminator_deid = total_loss_discriminator_deid / len(valid_loader)

    return valid_loss_generator, valid_loss_feature, valid_loss_discriminator_orin, valid_loss_discriminator_deid

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--params_path", type=str, default="config/params_cyclegan.yml", help="model parameters file path")

    option = parser.parse_known_args()[0]
    params = load_params_yml(option.params_path)["train"]

    train(params["device"], params)
