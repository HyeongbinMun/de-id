
# from model.det.face.yolov5.yolov5_face import YOLOv5Face
from model.det.face.yolov7.yolov7_face import YOLOv7Face
from model.vcd.s2vs.model.feature_extractor import FeatureExtractor
from model.vcd.vcd.models.frame import MobileNet_AVG
from model.icd.sscd.sscd.models.model import Model as ResNet
from model.deid.feature_inversion.models.mobilenetv2_inversion import MobileNetV2Inverter
from model.deid.feature_inversion.models.resnet50_inversion import ResNet50Inverter
from model.deid.feature_inversion.models.mobile_unet import MobileUNetInverter
from model.deid.feature_inversion.models.resnet50_unet import ResNet50UNetInverter
from model.deid.feature_inversion.models.mobilenetv2_unet import MobileNetV2UNetInverter
from model.deid.feature_inversion.models.mobilenetv3_unet import MobileNetV3SmallUNetInverter
from model.deid.feature_inversion.models.mobilenetv3_unet import MobileNetV3LargeUNetInverter
from model.deid.gan.models.generator import GeneratorResNet

model_classes = {
    "det": {
        "face": {
            # "yolov5": YOLOv5Face,

            "yolov7": YOLOv7Face,
        }
    },
    "feature": {
        "MobileNet_AVG": MobileNet_AVG,
        "ResNet50": ResNet,
        "S2VC": FeatureExtractor,
    },
    "deid": {
        "ResNet50Inverter": ResNet50Inverter,
        "ResNet50UNetInverter": ResNet50UNetInverter,
        "MobileNetV2Inverter": MobileNetV2Inverter,
        "MobileUNetInverter": MobileUNetInverter,
        "MobileNetV2UNetInverter": MobileNetV2UNetInverter,
        "MobileNetV3SmallUNetInverter": MobileNetV3SmallUNetInverter,
        "MobileNetV3LargeUNetInverter": MobileNetV3LargeUNetInverter,
        "ResNetD2GAN": GeneratorResNet,
    },
}