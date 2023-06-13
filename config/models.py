from model.det.face.yolov5.yolov5_face import YOLOv5Face
from model.det.face.yolov7.yolov7_face import YOLOv7Face

models = {
    "det": {
        "yolov5-face": YOLOv5Face,
        "yolov7-face": YOLOv7Face,
    },
    "deid": {
        "cyclegan": "",
        "feature_inversion": ""
    }
}