from model.det.face.yolov5.yolov5_face import YOLOv5Face
from model.det.face.yolov7.yolov7_face import YOLOv7Face
from model.vcd.models.mobilenetv2_avg import MobileNet_AVG
from model.deid.feature_inversion.models.resnet50_unet import ResNet50UNetInverter
from model.deid.feature_inversion.models.mobilenetv2_unet import MobileNetV2UNetInverter
from model.deid.feature_inversion.models.mobilenetv3_unet import MobileNetV3SmallUNetInverter
from model.deid.feature_inversion.models.mobilenetv3_unet import MobileNetV3LargeUNetInverter

model_classes = {
    "det": {
        "face": {
            "yolov5": YOLOv5Face,
            "yolov7": YOLOv7Face,
        }
    },
    "feature": {
        "MobileNet_AVG": MobileNet_AVG
    },
    "deid": {
        "cyclegan": "",
        "ResNet50UNetInverter": ResNet50UNetInverter,
        "MobileNetV2UNetInverter": MobileNetV2UNetInverter,
        "MobileNetV3SmallUNetInverter": MobileNetV3SmallUNetInverter,
        "MobileNetV3LargeUNetInverter": MobileNetV3LargeUNetInverter,
    },
}