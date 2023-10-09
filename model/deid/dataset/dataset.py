import os
import torch
import PIL.Image as Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FaceDetDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        self.image_files = sorted(os.listdir(image_dir))
        self.txt_files = sorted(os.listdir(label_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.txt_files[idx])

        image = Image.open(image_path).convert('RGB')
        image_width, image_height = image.size
        if self.transform == None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        image = self.transform(image)

        with open(label_path, 'r') as f:
            boxes = []
            for line in f:
                class_id, x_center, y_center, w, h = map(float, line.split())
                boxes.append([class_id, x_center, y_center, w, h])
            boxes = torch.tensor(boxes)

        return image_path, label_path, image, boxes

    @staticmethod
    def collate_fn(batch):
        image_paths = [item[0] for item in batch]
        label_paths = [item[1] for item in batch]
        images = [item[2] for item in batch]
        boxes_list = [item[3] for item in batch]

        images = torch.stack(images, dim=0)

        return image_paths, label_paths, images, boxes_list


class GANFaceDetDataset(Dataset):
    def __init__(self, image_orin_dir, image_deid_dir, label_dir, transform=None):
        self.image_orin_dir = image_orin_dir
        self.image_deid_dir = image_deid_dir
        self.label_dir = label_dir
        self.transform = transform

        self.image_orin_files = sorted(os.listdir(image_orin_dir))
        self.image_deid_files = sorted(os.listdir(image_deid_dir))
        self.txt_files = sorted(os.listdir(label_dir))

    def __len__(self):
        return len(self.image_orin_files)

    def __getitem__(self, idx):
        image_orin_path = os.path.join(self.image_orin_dir, self.image_orin_files[idx])
        image_deid_path = os.path.join(self.image_deid_dir, self.image_deid_files[idx])
        label_path = os.path.join(self.label_dir, self.txt_files[idx])

        image_orin = Image.open(image_orin_path).convert('RGB')
        if self.transform == None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        image_orin = self.transform(image_orin)

        image_deid = Image.open(image_deid_path).convert('RGB')
        if self.transform == None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        image_deid = self.transform(image_deid)

        with open(label_path, 'r') as f:
            boxes_list = []
            for line in f:
                class_id, x_center, y_center, w, h = map(float, line.split())
                boxes_list.append([class_id, x_center, y_center, w, h])
            boxes_list = torch.tensor(boxes_list)

        return image_orin_path, image_deid_path, label_path, image_orin, image_deid, boxes_list

    @staticmethod
    def collate_fn(batch):
        image_real_paths = [item[0] for item in batch]
        image_deid_paths = [item[1] for item in batch]
        label_paths = [item[2] for item in batch]
        images_orin = [item[3] for item in batch]
        images_deid = [item[4] for item in batch]
        boxes_list = [item[5] for item in batch]
        images_orin = torch.stack(images_orin, dim=0)
        images_deid = torch.stack(images_deid, dim=0)

        return image_real_paths, image_deid_paths, label_paths, images_orin, images_deid, boxes_list