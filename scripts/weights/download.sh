DIR="/workspace/model/weights"

# Check if the directory exists
if [ ! -d "$DIR" ]; then
  # If the directory does not exist, create it
  mkdir -p "$DIR"
  echo "Directory $DIR created."
else
  echo "Directory $DIR already exists."
fi

declare -A FILES=(
    ["1TQsDFUfGOefd3zeHP-jxfESMeVUCZ6DR"]="mobilenet_avg_ep16_ckpt.pth"
    ["1MDi5_DEuzjPL40B3jfcQqRhCUJfP5Erc"]="sscd_disc_mixup.torchvision.pt"
    ["1WmLgFK-GYCQfgwEO1RTqbs1jhB10-ctv"]="yolov5m-face.pt"
    ["1JXVk4Wb2ivLvAj70WIDDxdiyxuuGmfn6"]="yolov7-w6-face.pt"
    ["1CDpugb6RNSnBn4gQRFQA3gaQsu8d3JbV"]="feature_inversion_mobileunet_dna.pth"
    ["1PQXVEZsdcj1f4LmuIo3AAeuX-seIvkbZ"]="feature_inversion_resnet50_dna.pth"
)

for FILE_ID in "${!FILES[@]}"; do
    OUTPUT_FILENAME="${FILES[$FILE_ID]}"
    echo "======================================="
    echo "Downloading ${OUTPUT_FILENAME}..."
    wget -q --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt \
    --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=${FILE_ID}" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILE_ID}" \
    -O $DIR/${OUTPUT_FILENAME} && rm -rf /tmp/cookies.txt && echo "Successfully Downloaded - ${OUTPUT_FILENAME}" \
|| echo -e "\e[31mDownload Failed - ${OUTPUT_FILENAME}\e[0m"

done
echo "======================================="