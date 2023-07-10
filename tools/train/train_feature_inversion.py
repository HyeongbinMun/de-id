import math
import os
import sys
import random

import torch
import wandb
import argparse
import datetime
import shutil
from torch import nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utility.params import load_params_yml
from utility import logging
from utility.image.ssim import ssim
from utility.image.file import save_tensort2image
from utility.image.color_histogram import calculate_histogram_similarity
from utility.model.model_file import save_checkpoint, load_checkpoint
from model.deid.feature_inversion.dataset.dataset import DNADBDataset
from config.models import model_classes


def validate(valid_loader, feature_model, inversion_model, loss_fns, device, params, inverted_images_dir):
    feature_model.eval()
    inversion_model.eval()

    l1_loss_fn, mse_loss_fn, cosine_similarity, region_sim_loss_fn = loss_fns

    cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
    lrs = params["lambda"]["region_mse"]     # lambda region image mse
    lfm = params["lambda"]["feature_mse"]    # lambda MobileNet_AVG feature mse
    lfc = params["lambda"]["feature_cosine"] # lambda MobileNet_AVG cosine similarity

    valid_loss = 0.0

    with torch.no_grad():
        progress_bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        for batch_idx, (image_path, label_path, images, boxes_list) in progress_bar:
            images = images.clone().to(device)
            inverted_images = images.clone()

            region_losses = []
            for i, boxes in enumerate(boxes_list):
                for box in boxes:
                    image_w, image_h = images.shape[3], images.shape[2]
                    class_id, x_center, y_center, w, h = box
                    x1 = int((x_center - w / 2) * image_w)
                    y1 = int((y_center - h / 2) * image_h)
                    x2 = int((x_center + w / 2) * image_w)
                    y2 = int((y_center + h / 2) * image_h)
                    if (x2 - x1) != 0 and (y2 - y1) != 0:
                        region = images[i, :, y1:y2, x1:x2].clone()
                        if region.shape[1] > 10 and region.shape[2] > 10:
                            region_resized = F.interpolate(region.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
                            region_inverted = inversion_model(region_resized)
                            region_inverted_resized = F.interpolate(region_inverted, size=(int(region.shape[1]), int(region.shape[2])), mode='bilinear', align_corners=False)
                            sim_loss = mse_loss_fn(region_resized, region_inverted)
                            region_losses.append(sim_loss)
                            # sim_loss = 1 - sim_loss_fn(region.unsqueeze(0), region_inverted_resized)
                            # region_losses.append(sim_loss.item())
                            # sim_loss = region_sim_loss_fn(region.unsqueeze(0), region_inverted_resized.detach())
                            # region_losses.append(sim_loss)
                            inverted_images[i, :, y1:y2, x1:x2] = region_inverted_resized.squeeze(0)

            region_loss = torch.mean(torch.Tensor(region_losses)).item()
            resized_origin_images = F.interpolate(images.clone(), size=(224, 224))
            resized_inverted_images = F.interpolate(inverted_images.clone(), size=(224, 224))

            origin_features = feature_model(resized_origin_images)
            inverted_features = feature_model(resized_inverted_images)

            origin_features.requires_grad = True
            inverted_features.requires_grad = True


            feature_mse_loss = mse_loss_fn(origin_features, inverted_features)
            feature_cos_loss = cosine_similarity(origin_features, inverted_features).mean()

            lfeature_mse_loss = lfm * feature_mse_loss
            lfeature_cos_loss = lfc * feature_cos_loss
            lregion_ssim_loss = lrs * region_loss
            loss = (lfeature_mse_loss + lfeature_cos_loss + lregion_ssim_loss)

            selected_indices = random.sample(range(len(inverted_images)), 5)
            for i in selected_indices:
                save_tensort2image(inverted_images[i], os.path.join(inverted_images_dir, f'batch_{batch_idx}_image_{i}.png'))

            valid_loss += loss.item()
            progress_bar.set_description(f"Validating, Loss: {loss.item()}")

    valid_loss /= len(valid_loader)
    return valid_loss


def test(test_loader, feature_model, inversion_model, criterion, device, params, inverted_images_dir):
    return validate(test_loader, feature_model, inversion_model, criterion, device, params, inverted_images_dir)


def train(rank, world_size, params):
    # torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    image_dir = os.path.join(params["dataset_dir"], "images")
    label_dir = os.path.join(params["dataset_dir"], "labels")

    train_image_dir, train_label_dir = os.path.join(image_dir, "train"), os.path.join(label_dir, "train")
    valid_image_dir, valid_label_dir = os.path.join(image_dir, "valid"), os.path.join(label_dir, "valid")
    test_image_dir, test_label_dir = os.path.join(image_dir, "test"), os.path.join(label_dir, "test")

    train_dataset = DNADBDataset(train_image_dir, train_label_dir)
    valid_dataset = DNADBDataset(valid_image_dir, valid_label_dir)
    test_dataset = DNADBDataset(test_image_dir, test_label_dir)
    # train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    # valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank)
    # test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset,
                             batch_size=params["batch_size"],
                             shuffle=True,
                             num_workers=1,
                             # sampler=train_sampler,
                             collate_fn=DNADBDataset.collate_fn)
    valid_loader = DataLoader(valid_dataset,
                                   batch_size=params["batch_size"],
                                   shuffle=True,
                                   num_workers=1,
                                   # sampler=valid_sampler,
                                   collate_fn=DNADBDataset.collate_fn)
    test_loader = DataLoader(test_dataset,
                                   batch_size=params["batch_size"],
                                   shuffle=True,
                                   num_workers=1,
                                   # sampler=test_sampler,
                                   collate_fn=DNADBDataset.collate_fn)

    print(logging.i("Dataset Description"))
    print(logging.s(f"    training     : {len(train_dataset)}"))
    print(logging.s(f"    validation   : {len(valid_dataset)}"))
    print(logging.s(f"    test         : {len(test_dataset)}"))
    print(logging.i("Parameters Description"))
    print(logging.s(f"    Device       : {params['device']}"))
    print(logging.s(f"    Batch size   : {params['batch_size']}"))
    print(logging.s(f"    Max epoch    : {params['epoch']}"))
    print(logging.s(f"    Learning rate: {params['epoch']}"))
    print(logging.s(f"    Valid epoch: {params['valid_epoch']}"))
    print(logging.s(f"    Model:"))
    print(logging.s(f"    - Feature extractor: {params['model']['feature']['model_name']}"))
    print(logging.s(f"    - Feature inverter : {params['model']['inverter']}"))
    print(logging.s(f"Loss lambda:"))
    print(logging.s(f"- Feature MSE              : {params['lambda']['feature_mse']}"))
    print(logging.s(f"- Feature cosine similarity: {params['lambda']['feature_cosine']}"))
    print(logging.s(f"- MSE of each face regions : {params['lambda']['region_mse']}"))

    save_dir = params["save_dir"]
    start_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(save_dir, start_timestamp + "_" + params["model"]["inverter"])
    model_save_dir = os.path.join(model_dir, "weights")
    inverted_images_dir = os.path.join(model_dir, 'inverted_images')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(inverted_images_dir):
        os.makedirs(inverted_images_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(rank)

    feature_model = model_classes["feature"][params["model"]["feature"]["model_name"]]().to(device)
    state_dict = torch.load(params["model"]["feature"]["feature_weight_path"], map_location=device)
    state_dict = {k.replace("base.", ""): v for k, v in state_dict.items()}
    feature_model.load_state_dict(state_dict, strict=False)
    # feature_model = DistributedDataParallel(feature_model, device_ids=[rank], find_unused_parameters=True)
    feature_model.eval()

    inversion_model = model_classes["deid"][params["model"]["inverter"]]().to(device)
    # inversion_model = DistributedDataParallel(inversion_model, device_ids=[rank], find_unused_parameters=True)
    optimizer = optim.Adam(inversion_model.parameters(), lr=params["learning_rate"])
    if params["resume"]:
        inversion_model, optimizer, start_epoch, best_loss = load_checkpoint(inversion_model, optimizer, params["resume_from_checkpoint"])
    else:
        start_epoch = 0
        best_loss = float('inf')
    num_epochs = params["epoch"]
    wandb.watch(inversion_model)

    l1_loss_fn = nn.L1Loss()
    mse_loss_fn = nn.MSELoss()
    # region_sim_loss_fn = ssim
    region_sim_loss_fn = calculate_histogram_similarity
    cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    lfm = params["lambda"]["feature_mse"]    # lambda MobileNet_AVG feature mse
    lfc = params["lambda"]["feature_cosine"] # lambda MobileNet_AVG cosine similarity
    lrs = params["lambda"]["region_mse"]     # lambda region image mse

    torch.autograd.set_detect_anomaly(True)

    terminal_size = shutil.get_terminal_size().columns
    print("Status   Dev   Epoch  Loss ( mse  / cos  /region)")
    print("-" * terminal_size)
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (image_path, label_path, images, boxes_list) in progress_bar:
            images = images.clone().to(device)
            inverted_images = images.clone()
            region_losses = []
            for i, boxes in enumerate(boxes_list):
                for box in boxes:
                    image_w, image_h = images.shape[3], images.shape[2]
                    class_id, x_center, y_center, w, h = box
                    x1 = int((x_center - w / 2) * image_w)
                    y1 = int((y_center - h / 2) * image_h)
                    x2 = int((x_center + w / 2) * image_w)
                    y2 = int((y_center + h / 2) * image_h)
                    if (x2 - x1) != 0 and (y2 - y1) != 0:
                        region = images[i, :, y1:y2, x1:x2].clone()
                        if region.shape[1] > 10 and region.shape[2] > 10:
                            region_resized = F.interpolate(region.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
                            region_inverted = inversion_model(region_resized)
                            region_inverted_resized = F.interpolate(region_inverted, size=(int(region.shape[1]), int(region.shape[2])), mode='bilinear', align_corners=False)
                            sim_loss = mse_loss_fn(region_resized, region_inverted)
                            region_losses.append(sim_loss)
                            inverted_images[i, :, y1:y2, x1:x2] = region_inverted_resized.squeeze(0)

            region_loss = torch.mean(torch.Tensor(region_losses)).item()
            resized_origin_images = F.interpolate(images.clone(), size=(224, 224))
            resized_inverted_images = F.interpolate(inverted_images.clone(), size=(224, 224))

            with torch.no_grad():
                origin_features = feature_model(resized_origin_images)
                inverted_features = feature_model(resized_inverted_images)

            origin_features.requires_grad = True
            inverted_features.requires_grad = True

            optimizer.zero_grad()

            feature_mse_loss = mse_loss_fn(origin_features, inverted_features)
            feature_cos_loss = cosine_similarity(origin_features, inverted_features).mean()

            lfeature_mse_loss = lfm * feature_mse_loss
            lfeature_cos_loss = lfc * feature_cos_loss
            lregion_ssim_loss = lrs * region_loss

            loss = lfeature_mse_loss + lfeature_cos_loss + lregion_ssim_loss
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
            else:
                print(f"Skipping batch {batch_idx} due to invalid loss value: {loss.item()}")

            epoch_loss += loss.item()
            progress_bar.set_description(f"Train  {rank:^5} {epoch+1:>3}/{num_epochs:>3} {loss:.4f}({feature_mse_loss:.4f}/{feature_cos_loss:.4f}/{region_loss:.4f})")

        epoch_loss /= len(train_loader)
        save_checkpoint(epoch, inversion_model, optimizer, epoch_loss, os.path.join(model_save_dir, f"epoch_{epoch+1:06d}.pth"))

        wandb.log({"Train Loss": epoch_loss, "Epoch": epoch})
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(epoch, inversion_model, optimizer, epoch_loss, os.path.join(model_save_dir, f"epoch_{epoch+1:06d}.pth"))

        if (epoch + 1) % params["valid_epoch"] == 0:
            epoch_inverted_images_dir = os.path.join(inverted_images_dir, f"epoch-{epoch + 1}")
            if not os.path.exists(epoch_inverted_images_dir):
                os.makedirs(epoch_inverted_images_dir)
            valid_loss = validate(valid_loader, feature_model, inversion_model, (l1_loss_fn, mse_loss_fn, cosine_similarity, region_sim_loss_fn), device, params, epoch_inverted_images_dir)
            print(f"Validation Loss at epoch {epoch + 1}: {valid_loss}")
            wandb.log({"Validation Loss": valid_loss, "Epoch": epoch})

    test_inverted_images_dir = os.path.join(inverted_images_dir, f"test")
    if not os.path.exists(test_inverted_images_dir):
        os.makedirs(test_inverted_images_dir)
    test_loss = test(test_loader, feature_model, inversion_model, (l1_loss_fn, mse_loss_fn, cosine_similarity), device, params, test_inverted_images_dir)
    print(f"Test Loss: {test_loss}")


def main(params):
    train(params["device"], 1, params)
    # world_size = torch.cuda.device_count()
    # mp.spawn(train, args=(world_size, params), nprocs=world_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--params_path", type=str, default="config/params_inversion_resnet50.yml", help="dataset directory path")

    option = parser.parse_known_args()[0]
    params = load_params_yml(option.params_path)["train"]

    os.environ['MASTER_ADDR'] = params["master_addr"]
    os.environ['MASTER_PORT'] = params["master_port"]
    start_timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    wandb_run_name = f"{start_timestamp}-{params['model']['inverter']}"
    wandb.init(project=params['wandb']['project_name'], entity=params['wandb']['account'], name=wandb_run_name)
    main(params)
