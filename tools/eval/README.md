# Image Copy Detection
## Dataset
### DISC 2021
* disc 2021 데이터셋은 아래와 같은 구조로 reformating 해야 SSCD를 이용하여 평가가 가능하다.
#### SSCD 평가 코드 실행을 위한 disc 2021 구조 변경 명령어
```shell
ln -s disc2021/train /dataset/training
ln -s disc2021/final_queries /dataset/val_query
ln -s disc2021/references /dataset/val_ref
ln -s disc2021/final_ground_truth.csv /dataset/val_groundtruth_matches.csv
```
#### SSCD 평가 코드 실행을 위한 disc 2021 구조
```shell
/dataset
├── training                      # disc2021/train/
│   ├── T000000.jpg
│   ├── T000001.jpg
│   ├── ......
│   └── T999999.jpg
├── val_query                     # disc2021/final_query/
│   ├── Q500000.jpg
│   ├── Q500001.jpg
│   ├── ......
│   └── Q999999.jpg
├── val_ref                       # disc2021/references/
│   ├── R000000.jpg
│   ├── R000001.jpg
│   ├── ......
│   └── R999999.jpg
└── val_ground_truth_matches.csv  # disc2021/final_ground_truth.csv
```
* ```training```: 1,000,000 장
* ```val_query```: 50,002 장
* ```val_ref``` : 1,000,000 장
## Evaluation
## A Self-Supervised Descriptor for Image Copy Detection(SSCD)
```shell
# In /workspace
mkdir ./output
python3 tools/eval_disc.py --disc_path /dataset/disc/ --gpus=2 \
  --output_path=./output \
  --size=288 \
  --preserve_aspect_ratio=true \
  --backbone=CV_RESNET50 \
  --dims=512 \
  --model_state=weights/sscd_disc_mixup.classy.pt
```