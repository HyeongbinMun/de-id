DIR="/workspace/models/weights"

# Check if the directory exists
if [ ! -d "$DIR" ]; then
  # If the directory does not exist, create it
  mkdir -p "$DIR"
  echo "Directory $DIR created."
else
  echo "Directory $DIR already exists."
fi

echo "======================================="
echo "Download start(yolov5m-face)"
wget ftp://mldisk.sogang.ac.kr/models/yolov5-face/yolov5m-face.pt -O /workspace/models/weights/yolov5m-face.pt
echo "Download successfully complete(yolov5m-face)"
echo "======================================="