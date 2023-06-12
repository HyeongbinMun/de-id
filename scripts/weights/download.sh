DIR="/workspace/model/weights"

# Check if the directory exists
if [ ! -d "$DIR" ]; then
  # If the directory does not exist, create it
  mkdir -p "$DIR"
  echo "Directory $DIR created."
else
  echo "Directory $DIR already exists."
fi

echo "======================================="
echo "Download start(yolov5-face)"
wget -q ftp://mldisk.sogang.ac.kr/models/yolov5-face/yolov5m-face.pt -O /workspace/model/weights/yolov5m-face.pt \
&& echo "Download successful(yolov5-face)" \
|| echo -e "\e[31mDownload faile(yolov5-face)\e[0m"
echo "======================================="
echo "Download start(yolov7-face)"
wget -q ftp://mldisk.sogang.ac.kr/models/yolov7-face/yolov7-w6-face.pt -O /workspace/model/weights/yolov7-w6-face.pt \
&& echo "Download successful(yolov7-face)" \
|| echo -e "\e[31mDownload faile(yolov7-face)\e[0m"
echo "======================================="