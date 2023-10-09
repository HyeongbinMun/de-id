# Dataset preprocessing
* 본 프로젝트의 학습 데이터셋을 구축하는 스크립트 모음이다.
* 기본적으로 학습 데이터셋의 구조는 yolo dataset의 형식을 따른다.
* yolo dataset 의 디렉토리 구조
```shell
final_dataset
├── images
│   ├── test/
│   ├── train/
│   └── valid/
└── labels
    ├── test/
    ├── train/
    └── valid/
```
* yolo 
```shell
0 0.5309 0.2759 0.1307 0.3444 # class_index x y w h
```
## Face Dataset 정제 순서
### 1. split_dataset.py
```shell
# in /workspace
python3 tools/preprocess/common/split_dataset.py --source_dir=./source --target_dir=./dataset --fold=1,1,4
```
* Target directory 하위에 valid, test, train directory 를 생성 하고 source directory 하위에 있는 image 를 argument 에 주어진 비율로 복사 한다.
#### Source directory 구조
```shell
./source
├── 0001.jpg
├── 0002.jpg
├── ......
└── 1000.jpg
```
#### Target directory 구조
```shell
./dataset
└── images
    ├── valid/
    │    ├── 0001.jpg
    │    ├── ....
    │    └── 0200.jpg
    ├── test/
    │    ├── 0201.jpg
    │    ├── ....
    │    └── 0400.jpg
    └── train/
         ├── 0401.jpg
         ├── ....
         └── 1200.jpg
```
### 2. remove_invalid_data.py
```shell
python3 tools/preprocess/common/remove_invalid_data.py --dataset_dir=./dataset
```
* split_dataset.py 를 이용해 생성한 directory 내에 있는 image 파일 중 PIL 로 열리지 않는 image 를 삭제 한다.
* Directory 구조는 유지된다.
### 3. resize_images.py
```shell
python3 tools/preprocess/common/resize_images.py --source_dir=./dataset --target_dir=./dataset_1080p --resolution=1920x1080
```
* Source directory 하위에 있는 모든 image 를 인자 값으로 주어준 해상도로 resize 한다.
* Source directory 는 [yolo dataset 구조](#target-directory-구조) 와 동일 해야 하며, target directory 는 source directory 와 같은 구조를 가지게 된다.
### 4. generate_face_label.py
```shell
python3 tools/preprocess/common/generate_face_label.py --config=/workspace/config/params_yolov7face.yml --dataset_dir=./dataset_1080p
```
* 인자 값으로 주어진 dataset direcory 하위에 있는 image 에서 얼굴을 감지하고, yolov7face 모델을 사용하여 labels directory 하위에 yolo 형식의 label을 생성한다.
#### Dataset 구조
```shell
./dataset_1080p
├── images
│   ├── valid/
│   │    ├── 0001.jpg
│   │    ├── ....
│   │    └── 0200.jpg
│   ├── test/
│   │    ├── 0201.jpg
│   │    ├── ....
│   │    └── 0400.jpg
│   └── train/
│        ├── 0401.jpg
│        ├── ....
│        └── 1200.jpg
└── labels
    ├── valid/
    │    ├── 0001.txt
    │    ├── ....
    │    └── 0200.txt
    ├── test/
    │    ├── 0201.txt
    │    ├── ....
    │    └── 0400.txt
    └── train/
         ├── 0401.txt
         ├── ....
         └── 1200.txt
```
### 5. sampling.py(Optional.)
```shell
python3 tools/preprocess/common/sampling.py --fraction=0.1 --source_dir=./dataset_1080p --target_dir=./dataset_1080p_0.1
```
* [4. generate_face_label.py](#4-generatefacelabelpy)에서 생성한 dataset을 기반으로 주어진 인자 값(--fraction) 비율로 subset dataset을 생성한다.
* Source directory 와 target directory 는 같은 구조를 가지며, 포함된 image 의 수만 fraction 에 따라 달라지게 된다.