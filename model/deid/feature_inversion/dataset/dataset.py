import os
import torch
import PIL.Image as Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DNADBDataset(Dataset):
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