# De-identification
* 본 저장소는 CNN 기반 face de-identification 통합 프레임워크입니다.
## Requirements
* Ubuntu 20.04
* [Nvidia-driver 515](https://velog.io/@jinhasong/nvidia-driver)
  * Nvidia-driver version은 CUDA 버전에 맞춰 설치 필요
* [CUDA 11.7 이상](https://developer.nvidia.com/cuda-11-7-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local)
* [Docker.ce](https://velog.io/@jinhasong/Docker-install)
* [Nvidia-docker2](https://velog.io/@jinhasong/Nvidia-docker2-install)
* [Docker-compose](https://velog.io/@jinhasong/Docker-compose-install)
## Build docker container using Docker-compose
* 본 프로젝트는 docker-compose 기반으로 빌드하도록 구성되어 있기 때문에 git clone이후 docker-compose를 이용해 빌드전 아래 옵션을 변경 후 빌드해야 합니다.
### Clone Github Repository
```shell
# in workspace
cd ${WOKRSPACE}
git clone http://${PERSONAL_TOKEN}@github.com/jinhasong/de-id.git
cd de-id
```
* 본 github 저장소를 clone하고 clone한 저장소의 디렉토리로 이동합니다.
* 해당 저장소는 private로 되어 있기 떄문에 본인 계정의 personal token을 발행해야 clone이 가능합니다.
### docker-compose.yml
```yaml
# vim docker-compose.yml
version: '2.3'

services:
  model:
    container_name: de-id_model
    build:
      context: ./
      dockerfile: docker/dockerfile_model
    runtime: nvidia
    restart: always
    ipc: "host"
    env_file:
      - "docker/dockerenv.env"
    volumes:
      - type: volume
        source: dataset
        target: /dataset
        volume:
          nocopy: true
    ports:
      - "8022:22"
      - "8000:8000"
      - "8001:8001"
      - "8888:8888"
    stdin_open: true
    tty: true
    environment:
      TZ: Asia/Seoul
volumes:
  dataset:
    driver: local
    driver_opts:
      type: none
      device: "/media/hdd" # 수정 필요
      o: bind
```
* 위는 [docker-compose.yml](docker-compose.yml)의 내용으로 빌드 전 line 34의 ```"/media/hdd"``` 부분을 충분한 공간을 가진 HDD나 SSD로 지정해야 합니다.
  * docker container를 빌드하면 ```/dataset```의 경로에 위치하게 되며 evaluation dataset이나 train dataset을 저장하는 용도로 사용합니다.
### Build docker container
```shell
# in workspace
docker-compose up -d
# After docker container build is complete.
docker attach de-id_model
```
* ```docker-compose up -d``` 명령어로 빌드 완료 후 ```docker attach de-id_model``` 명령어로 docker container에 접속한다.
## Evaluation
### Download DNA frame database
* DNA Video Dataset에서 추출한 frame dataset은 아래 링크에서 다운로드 가능합니다.
### DNA frame dataset 정보
* 아래는 DNA frame dataset의 정보입니다.
<table>
    <thead>
        <tr>
            <td colspan="3"><b>Dataset</b></td>
            <td><b>DNA</b></td>          
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="5">데이터 정보</td>
            <td rowspan="4">동영상</td>
            <td>동영상 수</td>
            <td>447</td>
        </tr>
        <tr>
            <td>동영상 최대 길이</td>
            <td>6,339초</td>
        </tr>
        <tr>
            <td>동영상 최소 길이</td>
            <td>20초</td>
        </tr>
        <tr>
            <td>동영상 평균 길이</td>
            <td>1,451.41초</td>
        </tr>
        <tr>
            <td colspan="2">프레임 수(1fps)</td>
            <td colspan="2">648,666</td>
        </tr>
        <tr>
            <td rowspan="13">정제 데이터 정보</td>
            <td colspan="2">얼굴 포함 이미지 수</td>
            <td>258,354</td>
        </tr>
        <tr>
            <td colspan="2">총 얼굴 bounding box 수</td>
            <td>309,946</td>
        </tr>
        <tr>
            <td rowspan="8">이미지에서 얼굴<br> 영역이 차지하는<br> 비율</td>
            <td>최대</td>
            <td>79.76%</td>
        </tr>
        <tr>
            <td>최소</td>
            <td>0%</td>
        </tr>
        <tr>
            <td>평균</td>
            <td>4.75%</td>
        </tr>
        <tr>
            <td>0~10%</td>
            <td>217,167</td>
        </tr>
        <tr>
            <td>10~30%</td>
            <td>36,440</td>
        </tr>
        <tr>
            <td>30~50%</td>
            <td>4,559</td>
        </tr>
        <tr>
            <td>50~70%</td>
            <td>172</td>
        </tr>
        <tr>
            <td>70%이상</td>
            <td>17</td>
        </tr>
    </tbody>
</table>

### How to Evaluation
#### De-identification Model using Feature Inversion
```shell
# in /workspace
python3 tools/evel/eval_deid_inversion.py \
    --config=config/config_inversion_mobileunet_dna.yml \
    --batch_size=16 \
    --dataset_dir=/dataset/dna_frame/ \
    --output_dir=/dataset/result/mobileunet \
    --save
```
* Arguments
  * ```config```: model config file
    * MobileNetv2 based feature inversion: ```config_inversion_mobileunet_dna.yml```
    * ResNet50 based feature inversion: ```config_inversion_resnet50unet_dna.yml```
  * dataset_dir: dataset이 저장된 directory path
  * output_dir: feature inversion 생성 결과를 저장할 directory
    * ```--save``` argument를 사용해야 저장
##### 결과 출력
```shell
	        10% 	30% 	50% 	70% 	70above	average
-----------------------------------------------------------
ssim    	0.0282	0.0260	0.0245	0.0243	0.0163	0.0280
psnr    	7.4593	7.1601	6.7994	6.0713	5.9969	7.4217
image sim	0.9995	0.9953	0.9878	0.9856	1.0000	0.9990
```
#### De-identification Model using D2GAN
```shell
# in /workspace
python3 tools/evel/eval_deid_d2gan.py \
    --config=config/config_d2gan_dna.yml \
    --batch_size=16 \
    --dataset_dir=/dataset/dna_frame/ \
    --output_dir=/dataset/result/d2gan \
    --save
```
* Arguments
  * ```config```: model config file
    * D2GAN config: ```config_d2gan_dna.yml```
  * dataset_dir: dataset이 저장된 directory path
  * output_dir: D2GAN 생성 결과를 저장할 directory
    * ```--save``` argument를 사용해야 저장
##### 결과 출력
```shell
	        10% 	30% 	50% 	70% 	70above	average
-----------------------------------------------------------
ssim    	0.0282	0.0260	0.0245	0.0243	0.0163	0.0280
psnr    	4.4593	4.1601	4.7994	4.0713	4.9969	4.4217
image sim	0.9995	0.9953	0.9878	0.9856	1.0000	0.9990
```