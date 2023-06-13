from numpy import ndarray

class Detection:
    def __init__(self, params):
        self.params = params
        self.img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']
        self.vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']

    def detect(self, image:ndarray):
        """
        :param image: a image to inference.
                      - format: ndarray
        :return:      infer result
                      - format: [cls, cls_idx, score, x, y, w, h])
        """
        return None

    def detect_batch(self, images:list):
        """
        :param images: images to inference.
                       - format: list of ndarray
        :return:       infer result
                       - format: [[cls, cls_idx, score, x, y, w, h], [cls, cls_idx, score, x, y, w, h]...]
        """
        result = []
        for i, image in enumerate(images):
            result.append(self.detect(image))

        return result