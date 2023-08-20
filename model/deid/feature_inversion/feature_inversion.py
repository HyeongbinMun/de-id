import os
import sys
import torch
import numpy
import torch.nn.functional as F
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.models import model_classes

class FeatureInversion:
    def __init__(self, params):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(params["device"])
        checkpoint = torch.load(params["model_path"])
        self.inversion_model = model_classes["deid"][params["model_name"]]().to(self.device)
        self.inversion_model.load_state_dict(checkpoint['model_state_dict'])
        self.transform = transforms.Compose([transforms.ToTensor()])

    def inference(self, image, boxes):
        tensor_image = self.transform(image)
        tensor_image = tensor_image.clone().to(self.device)
        inverted_image = tensor_image.clone()

        regions = []
        for i, box in enumerate(boxes):
            image_w, image_h = tensor_image.shape[2], tensor_image.shape[1]
            class_id, x_center, y_center, w, h = box
            x1 = int((x_center - w / 2) * image_w)
            y1 = int((y_center - h / 2) * image_h)
            x2 = int((x_center + w / 2) * image_w)
            y2 = int((y_center + h / 2) * image_h)
            if (x2 - x1) != 0 and (y2 - y1) != 0:
                region = tensor_image[:, y1:y2, x1:x2].clone()
                region_resized = F.interpolate(region.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
                region_inverted = self.inversion_model(region_resized)
                region_inverted_resized = F.interpolate(region_inverted, size=(int(region.shape[1]), int(region.shape[2])), mode='bilinear', align_corners=False)
                inverted_image[:, y1:y2, x1:x2] = region_inverted_resized.squeeze(0)

                region_inverted_resized = region_inverted_resized.squeeze(0).cpu().detach().numpy()
                region_inverted_resized = numpy.transpose(region_inverted_resized, (1, 2, 0))
                region_inverted_resized = numpy.clip(region_inverted_resized * 255.0, 0, 255).astype(numpy.uint8)
                regions.append(region_inverted_resized)

        inverted_image = inverted_image.cpu().detach().numpy()
        inverted_image = numpy.transpose(inverted_image, (1, 2, 0))
        inverted_image = numpy.clip(inverted_image * 255.0, 0, 255).astype(numpy.uint8)

        return inverted_image, regions