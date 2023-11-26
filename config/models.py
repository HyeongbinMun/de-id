# from model.det.face.yolov5.yolov5_face import YOLOv5Face
from model.det.face.yolov7.yolov7_face import YOLOv7Face
from model.vcd.models.mobilenetv2_avg import MobileNet_AVG
from model.icd.sscd.sscd.models.model import Model as ResNet
# from model.deid.feature_inversion.models.vae import VAE
from model.deid.feature_inversion.models.mobilenetv2_inversion import MobileNetV2Inverter
from model.deid.feature_inversion.models.resnet50_inversion import ResNet50Inverter
from model.deid.feature_inversion.models.mobile_unet import MobileUNetInverter
from model.deid.feature_inversion.models.resnet50_unet import ResNet50UNetInverter
from model.deid.feature_inversion.models.mobilenetv2_unet import MobileNetV2UNetInverter
from model.deid.feature_inversion.models.mobilenetv3_unet import MobileNetV3SmallUNetInverter
from model.deid.feature_inversion.models.mobilenetv3_unet import MobileNetV3LargeUNetInverter

model_classes = {
    "det": {
        "face": {
            # "yolov5": YOLOv5Face,
            "yolov7": YOLOv7Face,
        }
    },
    "feature": {
        "MobileNet_AVG": MobileNet_AVG,
        "ResNet50": ResNet
    },
    "deid": {
        "cyclegan": "",
        # "VAE": VAE,
        "MobileUNetInverter": MobileUNetInverter,
        "ResNet50UNetInverter": ResNet50UNetInverter,
        "MobileNetV2UNetInverter": MobileNetV2UNetInverter,
        "MobileNetV3SmallUNetInverter": MobileNetV3SmallUNetInverter,
        "MobileNetV3LargeUNetInverter": MobileNetV3LargeUNetInverter,
        "MobileNetV2Inverter": MobileNetV2Inverter,
        "ResNet50Inverter": ResNet50Inverter,
    },
}