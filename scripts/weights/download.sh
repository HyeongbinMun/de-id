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
wget -q ftp://mldisk2.sogang.ac.kr/models/yolov5-face/yolov5m-face.pt -O /workspace/model/weights/yolov5m-face.pt \
&& echo "Download successful(yolov5-face)" \
|| echo -e "\e[31mDownload failed(yolov5-face)\e[0m"
echo "======================================="
echo "Download start(yolov7-face)"
wget -q ftp://mldisk2.sogang.ac.kr/models/yolov7-face/yolov7-w6-face.pt -O /workspace/model/weights/yolov7-w6-face.pt \
&& echo "Download successful(yolov7-face)" \
|| echo -e "\e[31mDownload failed(yolov7-face)\e[0m"
echo "======================================="
echo "Download start(feature inversion mobileunet DNA)"
wget -q ftp://mldisk2.sogang.ac.kr/etri/dna/feature_inversion_mobileunet_dna.pth -O /workspace/model/weights/feature_inversion_mobileunet_dna.pth \
&& echo "Download successful(feature inversion mobileunet DNA)" \
|| echo -e "\e[31mDownload failed(feature inversion mobileunet DNA)\e[0m"
echo "======================================="
echo "Download start(feature inversion resnet50 DNA)"
wget -q ftp://mldisk2.sogang.ac.kr/etri/dna/feature_inversion_resnet50_dna.pth -O /workspace/model/weights/feature_inversion_resnet50_dna.pth \
&& echo "Download successful(feature inversion resnet50 DNA)" \
|| echo -e "\e[31mDownload failed(feature inversion resnet50 DNA)\e[0m"
echo "======================================="
echo "Download start(feature inversion D2GAN DNA)"
wget -q ftp://mldisk2.sogang.ac.kr/etri/dna/d2gan_dna.pth -O /workspace/model/weights/d2gan_dna.pth \
&& echo "Download successful(feature inversion D2GAN DNA)" \
|| echo -e "\e[31mDownload failed(feature inversion D2GAN DNA)\e[0m"
echo "======================================="