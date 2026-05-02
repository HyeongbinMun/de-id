# `train_inpaint_deid_v2.py` — Face De-Identification via Inpainting + LoRA (v2)

`SD 1.5` 기반 인페인팅 모델(권장: Realistic Vision V5.1 Inpainting)에
**LoRA**를 주입해 **얼굴 영역만 재생성**한다. 배경은 원본 그대로 유지되어
"black-box collapse"가 구조적으로 일어나기 어렵다.

대상 파일:

- 학습 스크립트: [`tools/train/train_inpaint_deid_v2.py`](../tools/train/train_inpaint_deid_v2.py)
- 실행 셸: [`scripts/train_inpaint_deid_v2.sh`](../scripts/train_inpaint_deid_v2.sh)

---

## 1. 학습 목적

논문 정의에 맞춰 **두 개의 상반된 목표**를 동시에 만족시키는 것이다.

1. **육안/얼굴 임베딩 기준 신원은 다른 사람으로 보이게 만든다**
   - 얼굴 임베딩 코사인 유사도(`face_id_model`)를 낮추는 방향으로 학습.
2. **CBIR(Image Copy Detection 등) 특징 벡터는 원본과 비슷하게 유지한다**
   - SSCD/MobileNet_AVG 임베딩 코사인 유사도(`feature_model`)를 1에 가깝게 유지.

추가로 인페인팅 품질을 위한 표준 **확산 손실**과, 선택적 **LPIPS**가 있다.
이미지 복원 품질은 학습에 직접 쓰지 않고 **PSNR/SSIM**만 모니터링한다.

---

## 2. 아키텍처 한 눈에

```
입력 이미지 ─┐
            ├─► VAE ─► z0 ─► 노이즈 추가 ─► z_t
마스크   ───┤                                      │
얼굴 마스킹 ─► VAE ─► z_masked                     │
                                                   ▼
            (z_t, mask, z_masked) ─► UNet(9ch) ─► 노이즈 예측
                                                   │
                                                   ▼
                       x0 추정 ─► VAE.decode ─► 생성 얼굴
                                                   │
                       Composite (mask로 합성) ◄────┘
                                                   │
                                                   ▼
                       원본 + 생성 얼굴 = 비식별화 합성 이미지
```

- **UNet 입력 채널**: 4(z_t) + 1(mask) + 4(z_masked) = **9ch** (인페인팅 전용)
- **LoRA 주입 대상**: UNet 어텐션의 `to_q, to_k, to_v, to_out.0`
- **나머지 가중치**: 텍스트 인코더, VAE, UNet 베이스 가중치는 모두 동결

---

## 3. 손실 함수

```
L_total = λ_diff * L_diff
        + aux_scale * (
              λ_id        * L_id
            + λ_feat      * L_feat
            + λ_lpips     * L_lpips        # --use_lpips_loss 일 때만
            + λ_clip_face * L_clip_face    # --use_clip_face_loss 일 때만
          )

aux_scale = warmup * timestep_weight * decay
```

| 기호 | 정의 | 비고 |
|------|------|------|
| `L_diff` | `MSE(noise_pred, target)` | `epsilon`/`v_prediction` 자동 처리 |
| `L_id` | `cos(face_id_orig, face_id_gen)` 평균 | `--identity_loss_mode` 로 모드 변경 |
| `L_feat` | `1 - cos(feat_orig, feat_gen)` | 전체 합성 이미지에 대해 SSCD/MobileNet_AVG |
| `L_lpips` | LPIPS(face_orig, face_gen) | 얼굴 crop, 기본은 OFF |
| `L_clip_face` | `1 - cos(CLIP(face_orig), CLIP(face_gen))` | 얼굴 crop의 의미적 속성(나이/성별/표정/액세서리 등) 보존 |

### 3.0 얼굴 의미 속성 보존 (`L_clip_face`)

기존 v2는 `face_id_model`을 떨어뜨리는 동시에 **나이·성별·표정·머리 같은
거시 속성도 같이 깨뜨려서** "애기 얼굴 → 기괴한 어른 남자"가 자주 나왔다.
이를 막기 위해 **얼굴 crop만** CLIP image encoder에 넣고 임베딩의 코사인
유사도를 보존한다.

- CLIP 임베딩은 정체성 식별보다 **고수준 의미(외형/스타일/속성)** 를 더 많이 담는다.
  → `L_id`는 신원만 떨어뜨리고, `L_clip_face`는 나머지 속성을 잡아주는 역할 분리.
- 동작은 `--use_clip_face_loss` 플래그로 켠다 (기본 ON, shell에서 `USE_CLIP_FACE_LOSS=0`로 끔).
- 입력은 원본/생성 face crop을 CLIP의 224×224로 bilinear 리사이즈 후
  CLIP `image_mean/std`로 정규화 → `pooler_output`을 L2-norm.
- 가중치 `--lambda_clip_face` (기본 0.3). 너무 크면 신원도 안 바뀌므로
  주의. `λ_id ≈ λ_clip_face` 정도부터 시작 권장.

### 3.1 신원 손실 모드 (`--identity_loss_mode`)

- `cosine` (기본): `cos(id_orig, id_gen).mean()`을 그대로 줄임 → **항상 gradient 존재**
- `margin`: `max(cos - id_margin, 0).mean()` → 충분히 떨어지면 gradient 0 (덜 공격적)

### 3.2 보조 손실 스케일 `aux_scale`

- `warmup = min(1, global_step / aux_warmup_steps)`
  - 학습 초반 보조 손실을 점진적으로 켜서 확산 학습이 우선되게 한다.
- `timestep_weight = mean(alpha_cumprod[timesteps])` (낮은 t일수록 큼)
  - 보조 손실은 **저잡음 영역에서만** 신뢰할 수 있다는 직관 반영.
  - `--timestep_threshold` (기본 250) 이상 timestep 샘플은 보조 손실 계산에서 제외.
- `decay = aux_decay_factor()` (선택)
  - `--aux_decay_start_step` 이후 `--aux_decay_end_step`까지 선형 감쇠 → `aux_decay_min`까지.
  - 후반에 보조 손실이 너무 강하게 밀어 붕괴를 일으키는 것을 완화.

### 3.3 검증 시 손실

검증 단계에서는 결정적 비교를 위해 `decay`를 적용하지 않는다(=학습 곡선이 덜 흔들림).

---

## 4. 데이터셋 형식 — `InpaintDeIdDataset`

기대하는 디렉토리 구조 (`--data_root`):

```
<data_root>/
├── images/{train,val,test}/
├── masks/{train,val,test}/      # 0/255 또는 0/1 마스크 (얼굴 영역=1)
├── labels/{train,val,test}/     # YOLO 형식 (선택). 없어도 동작
└── captions/{train,val,test}/   # per-image 텍스트 프롬프트 (선택). 없으면 fallback
```

- 이미지/마스크는 **stem(파일명)** 일치로 페어링.
- `captions/<split>/<stem>.txt` 가 존재하면 해당 텍스트를 그 이미지의 학습/추론 프롬프트로 사용. 없으면 `--prompt` (기본값) 로 fallback.
- YOLO 라벨이 있으면 첫 번째 박스 좌표를 `bbox` 텐서로 함께 반환 → 얼굴 crop 기반 손실/지표에 사용.
- 모든 이미지/마스크는 `--resolution` (기본 512) 로 Resize + CenterCrop.

배치 dict:

```python
{
  "pixel_values": Tensor[B, 3, H, W],   # [-1, 1]
  "mask":         Tensor[B, 1, H, W],   # {0, 1}
  "masked_image": Tensor[B, 3, H, W],   # 얼굴 영역 zero
  "prompts":      List[str],            # per-image 텍스트 프롬프트
  "bbox":         Tensor[B, 4]          # (선택) x1, y1, x2, y2
}
```

### 4.1 per-image 프롬프트 사전 생성 (LLaVA via Ollama)

이전 v2는 모든 이미지에 동일한 고정 프롬프트로 학습돼 텍스트 컨디셔닝이
**거의 의미가 없었다**. 새 파이프라인은 *학습 시작 전에* LLaVA 로 사람별
캡션을 만들어두고 학습 중에는 사전 인코딩된 임베딩을 룩업만 한다.

준비:

```bash
# (호스트) ollama 준비
ollama serve &
ollama pull llava:13b

# 캡션 생성 (resume 지원: 이미 .txt 가 있으면 skip)
python /workspace/tools/preprocess/common/caption_with_llava.py \
    --dataset_dir /dataset/deid/ffhq \
    --model llava:13b \
    --workers 4
```

결과:

```
/dataset/deid/ffhq/captions/{train,val}/<stem>.txt
```

각 .txt 에는 한 줄짜리 캡션(예: `"a photo of a 30s asian woman with long black hair, neutral expression, soft daylight"`)이 들어간다.

학습 측 동작 흐름:

1. `InpaintDeIdDataset` 가 sample 마다 `prompt` 문자열도 같이 들고 있는다.
2. `LightningModule.on_fit_start` 에서 train+val 의 모든 unique 프롬프트를
   한 번에 텍스트 인코더에 통과시켜 **CPU fp16** 으로 캐시한다.
   (FFHQ 70k 기준 약 8GB CPU RAM, GPU 메모리는 거의 안 늘어남.)
3. `training_step` / `validation_step` / DDIM 샘플링은 캐시에서 룩업해
   배치 단위로 GPU 로 옮겨 사용한다. 매 스텝마다 텍스트 인코더 forward 가
   다시 도는 일은 없다.

캡션이 일부만 존재해도 안전하다. 누락된 stem 은 `--prompt` 로 fallback.

---

## 5. 모델 로딩

`__init__` 분기:

1. `--single_file_ckpt <safetensors>` 지정
   - `StableDiffusionInpaintPipeline.from_single_file()` 으로 한 번에 로드
   - **Realistic Vision V5.1 Inpainting** 등 단일 파일 체크포인트 사용 시 권장
2. 미지정 시
   - `--pretrained_model_name_or_path` 로 diffusers 포맷 로드 (예: `runwayml/stable-diffusion-inpainting`)

`--vae_path` 로 외부 VAE 오버라이드 가능 (기본 권장 `stabilityai/sd-vae-ft-mse`).

LoRA 주입 후 학습 가능 파라미터는 보통 UNet 전체의 **0.3~0.5%** 수준.

---

## 6. WandB 로깅

기본 프로젝트 이름: `face-deid-inpaint-v2`, 런 이름: `inpaint_deid_v2_r{rank}` (변경 가능).

### 6.1 스칼라

| 키 | 의미 |
|----|------|
| `train/loss_total` | 총 손실 |
| `train/loss_diffusion` | 확산 MSE |
| `train/loss_identity` | 신원 손실 (`identity_loss_mode` 적용 후) |
| `train/loss_feature` | CBIR 특징 손실 |
| `train/loss_lpips` | LPIPS (use_lpips_loss=True일 때만 유효) |
| `train/loss_clip_face` | 얼굴 crop CLIP 임베딩 보존 손실 (use_clip_face_loss=True일 때) |
| `train/id_cosine_sim` | 얼굴 임베딩 cos sim (낮을수록 다른 사람) |
| `train/feat_cosine_sim` | CBIR 특징 cos sim (높을수록 보존) |
| `train/clip_face_cosine_sim` | 얼굴 crop CLIP cos sim (높을수록 속성 보존) |
| `train/metric_psnr / ssim` | 합성 이미지 vs 원본 PSNR/SSIM |
| `train/metric_face_psnr / face_ssim` | 얼굴 crop PSNR/SSIM (16×16 이상일 때) |
| `train/aux_warmup / aux_decay / aux_scale` | 보조 손실 가중치 진행 상태 |
| `train/n_valid` | 보조 손실 계산에 쓰인 샘플 수 (timestep < threshold) |
| `train/lr` | 현재 LR |
| `train/grad_norm` | LoRA 파라미터 grad norm |
| `val/...` | 위와 동일 항목의 검증 버전 |

### 6.2 이미지

| 키 | 시점 |
|----|------|
| `data/initial/grid` | `on_train_start` 직후 1회 — 데이터 파이프라인 확인용 |
| `train/inpaint/grid` | `--sample_every_n_steps` 마다 — DDIM 풀 인퍼런스 결과 |
| `val/x0_pred/grid` | 매 검증 epoch — `validation_step`의 x0 예측 합성 |
| `val/inpaint/grid` | 매 검증 epoch — DDIM 풀 인퍼런스 결과 |
| `*/sample_i_orig` / `*/sample_i_gen` | 위 그리드의 개별 이미지 (i = 0..3) |

그리드 행: **Original → Mask → Masked → Generated**.

---

## 7. 주요 인자 정리

### 7.1 모델/데이터

| 인자 | 기본 | 설명 |
|------|------|------|
| `--single_file_ckpt` | None | safetensors 단일 파일 (Realistic Vision 등) |
| `--pretrained_model_name_or_path` | `runwayml/stable-diffusion-inpainting` | diffusers 포맷 fallback |
| `--vae_path` | None | 외부 VAE (예: `stabilityai/sd-vae-ft-mse`) |
| `--prompt` | "a highly detailed photo of a person face, sharp focus, natural skin texture" | per-image 캡션이 없는 샘플의 fallback 프롬프트 |
| `--captions_subdir` | `captions` | `<data_root>/<captions_subdir>/<split>/<stem>.txt` 에서 per-image 프롬프트 로드 |
| `--data_root` | (필수) | `images/`, `masks/`, `labels/`, (선택) `captions/` 포함 |
| `--resolution` | 512 | Resize+CenterCrop 후 정사각 크기 |

### 7.2 LoRA

| 인자 | 기본 | 설명 |
|------|------|------|
| `--lora_rank` | 16 | LoRA 랭크 |
| `--lora_alpha` | 32 | scaling = alpha / rank |

### 7.3 손실 가중치

| 인자 | 기본 |
|------|------|
| `--lambda_diffusion` | 1.0 |
| `--lambda_identity` | 0.5 |
| `--lambda_feature` | 0.5 |
| `--lambda_lpips` | 0.0 (use_lpips_loss 활성화 시 의미) |
| `--use_lpips_loss` | off | LPIPS를 총 손실에 포함 |
| `--lambda_clip_face` | 0.3 (use_clip_face_loss 활성화 시 의미) |
| `--use_clip_face_loss` | off (shell에선 기본 ON) | 얼굴 속성 보존 손실 활성화 |
| `--clip_model_name` | `openai/clip-vit-base-patch32` | HF CLIP vision 모델 |
| `--identity_loss_mode` | `cosine` | `cosine` 또는 `margin` |
| `--id_margin` | 0.3 | margin 모드일 때 임계값 |

### 7.4 Anti-collapse

| 인자 | 기본 | 설명 |
|------|------|------|
| `--timestep_threshold` | 250 | 보조 손실은 t < 이 값에서만 |
| `--aux_warmup_steps` | 2000 | 보조 손실 스케일 0→1 선형 증가 |
| `--aux_decay_start_step` | 0 | 0이면 비활성 |
| `--aux_decay_end_step` | 0 | 감쇠 종료 스텝 (`<= start`이면 자동 +20000) |
| `--aux_decay_min` | 0.25 | 감쇠 후 곱해지는 최소 비율 |

### 7.5 최적화

| 인자 | 기본 | 설명 |
|------|------|------|
| `--learning_rate` | `5e-6` | 보수적 기본값 (충분히 안정 → 필요 시 `1e-5`~`3e-5`) |
| `--lr_scheduler` | `constant` | `constant` 또는 `cosine` |
| `--lr_warmup_steps` | 0 | `cosine`일 때만 의미 |
| `--max_grad_norm` | 1.0 | grad clip |
| `--gradient_checkpointing` | off | VRAM 절약 |
| `--gradient_accumulation_steps` | 1 | 효과적 배치 = bs * accum * num_gpus |
| `--max_steps` | -1 | `>=0`이면 해당 스텝에서 종료 (과학습 완화) |

### 7.6 출력/로깅

| 인자 | 기본 |
|------|------|
| `--output_dir` | `./output/inpaint_deid_lora` |
| `--mixed_precision` | `fp16` |
| `--num_gpus` | 1 |
| `--wandb_project / wandb_name` | `face-deid-inpaint-v2` / 자동 |
| `--save_every_n_epochs` | 5 |
| `--sample_every_n_steps` | 100 |
| `--num_sample_inference_steps` | 30 |
| `--val_check_interval` | 1.0 |

---

## 8. 실행 방법

### 8.1 사전 준비: per-image 캡션 생성

학습 전에 한 번만 실행한다 (FFHQ 70k @ llava:13b 기준 GPU 1장으로 ~수 시간 단위).

가장 쉬운 방법은 셸 래퍼 사용:

```bash
# 기본: /dataset/deid/ffhq, splits = train/val/test, llava:13b, workers=4
bash /workspace/scripts/caption_ffhq.sh

# 옵션 오버라이드
DATA_ROOT=/dataset/deid/ffhq \
SPLITS="train val test" \
MODEL=llava:13b WORKERS=8 \
AUTOSTART_OLLAMA=1 \
bash /workspace/scripts/caption_ffhq.sh
```

`AUTOSTART_OLLAMA=1` 이면 ollama 데몬이 안 떠있을 때 자동으로 띄우고 종료
시 정리한다. 이미 외부에서 `ollama serve` 가 떠있으면 자동 감지해 그대로
사용한다. 모델이 캐시에 없으면 자동 `pull`.

#### 컨테이너 안에 ollama 가 없는 경우

이 레포의 도커 이미지는 ollama 를 미포함이라 처음 실행 시
`bash: ollama: command not found` 가 난다. 세 가지 해결책:

1) 컨테이너 안에 자동 설치 + 자동 기동 (가장 간단):

```bash
INSTALL_OLLAMA=1 AUTOSTART_OLLAMA=1 \
bash /workspace/scripts/caption_ffhq.sh
```

내부적으로 `curl -fsSL https://ollama.com/install.sh | sh` 를 실행한다.
컨테이너에 systemd 가 없어도 binary 만 있으면 백그라운드로 직접 띄워준다.

2) 수동 설치:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llava:13b
bash /workspace/scripts/caption_ffhq.sh
```

3) 호스트 머신의 ollama 데몬을 컨테이너에서 사용:

```bash
# 호스트에서:
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# 컨테이너에서 (host.docker.internal 또는 호스트 게이트웨이 IP):
SERVER_URL=http://host.docker.internal:11434 \
bash /workspace/scripts/caption_ffhq.sh
```

`docker run` 시 `--add-host=host.docker.internal:host-gateway` 또는
`--network=host` 가 필요할 수 있다.

직접 호출도 가능:

```bash
ollama serve &
ollama pull llava:13b

python /workspace/tools/preprocess/common/caption_with_llava.py \
    --dataset_dir /dataset/deid/ffhq \
    --splits train val test \
    --model llava:13b \
    --workers 4
```

생성된 `captions/{train,val,test}/<stem>.txt` 가 학습 시 자동으로 사용된다.
캡션 디렉토리가 없으면 학습은 동작하지만 모든 샘플이 fallback 프롬프트로
학습되므로 (= 기존 v2 와 동일) 권장하지 않는다.

스크립트는 resume 안전. 이미 .txt 가 있는 stem 은 자동 skip 되므로 중간에
끊겨도 다시 같은 명령을 돌리면 이어서 진행된다. 실패한 항목은
`captions/errors.log` 에 누적 기록된다.

### 8.2 학습

기본 실행:

```bash
bash /workspace/scripts/train_inpaint_deid_v2.sh
```

자주 쓰는 변수 (셸에서 환경변수로 오버라이드):

```bash
DATA_ROOT=/dataset/deid/ffhq \
OUTPUT_DIR=/dataset/deid/models/inpaint_deid_lora_v2_run1 \
SINGLE_FILE_CKPT=/dataset/deid/models/Realistic_Vision_V5.1-inpainting.safetensors \
VAE_PATH=stabilityai/sd-vae-ft-mse \
BATCH_SIZE=4 GRAD_ACCUM=2 NUM_GPUS=2 \
LEARNING_RATE=5e-6 LR_SCHEDULER=constant \
IDENTITY_LOSS_MODE=cosine \
LAMBDA_IDENTITY=0.5 LAMBDA_FEATURE=0.5 \
TIMESTEP_THRESHOLD=250 AUX_WARMUP_STEPS=2000 \
bash /workspace/scripts/train_inpaint_deid_v2.sh
```

100k 스텝에서 종료 + 보조 손실 후반 감쇠:

```bash
MAX_STEPS=100000 \
AUX_DECAY_START=80000 AUX_DECAY_END=100000 AUX_DECAY_MIN=0.25 \
bash /workspace/scripts/train_inpaint_deid_v2.sh
```

LPIPS 손실까지 포함:

```bash
USE_LPIPS_LOSS=1 LAMBDA_LPIPS=0.3 \
bash /workspace/scripts/train_inpaint_deid_v2.sh
```

CLIP face 속성 보존(기본 ON, 가중치 조절·모델 변경):

```bash
USE_CLIP_FACE_LOSS=1 LAMBDA_CLIP_FACE=0.3 \
CLIP_MODEL_NAME=openai/clip-vit-large-patch14 \
bash /workspace/scripts/train_inpaint_deid_v2.sh
```

CLIP face 보존을 끄고 싶을 때:

```bash
USE_CLIP_FACE_LOSS=0 bash /workspace/scripts/train_inpaint_deid_v2.sh
```

---

## 9. 안정성 관련 알려진 이슈와 가드

### 9.1 SSIM 입력 크기

`piqa.SSIM`은 기본 11×11 커널이라 얼굴 crop이 너무 작으면 conv가 실패한다.
**16픽셀 미만 crop은 face PSNR/SSIM 계산을 건너뛴다**.

```python
fh, fw = face_gen.shape[2], face_gen.shape[3]
if fh >= 16 and fw >= 16:
    face_psnr_vals.append(...)
    face_ssim_vals.append(...)
```

### 9.2 DDP allreduce 동기화

DDP에서는 모든 랭크가 **같은 횟수의 collective**를 호출해야 한다.
조건부 `self.log(..., sync_dist=True)`가 한쪽 랭크만 호출되면 NCCL 데드락이 난다.
v2에서는 PSNR/SSIM/face metrics 모두 **항상 0 placeholder로라도 로깅**한다.

### 9.3 NCCL 타임아웃

`DDPStrategy(timeout=15분)` 으로 설정. 한 랭크가 비정상 종료되면 빠르게 죽고
디버깅이 가능하도록 함.

### 9.4 black-box collapse

후반 epoch에서 보조 손실이 강하게 작용하면 디코드가 단색 이미지로 붕괴할 수 있다.
완화책:

1. `--learning_rate` 낮춤 (`5e-6` 권장)
2. `--lr_scheduler constant` (cosine보다 안정)
3. `--timestep_threshold` 작게 / `--aux_warmup_steps` 늘림
4. `--aux_decay_*` 로 후반에 보조 손실 자동 감쇠
5. `--max_steps` 로 과학습 전에 종료

---

## 10. 평가 지침

학습 자체는 cos sim과 PSNR/SSIM을 가벼운 시그널로 보고,
**최종 비식별화 성능은 별도 벤치마크**로 평가한다.

- **얼굴 매칭**: 별도 face matching benchmark (예: ArcFace, AdaFace 기반 1:1/1:N)
- **CBIR 보존**: SSCD/MobileNet_AVG 임베딩으로 retrieval mAP/Recall@k
- **이미지 품질**: PSNR, SSIM (피험자가 "다른 사람으로 보이게" 만드는 것이 목적이므로
  PSNR/SSIM은 **너무 높지 않아야** 정상)

---

## 11. 출력 산출물

```
<output_dir>/
├── checkpoints/inpaint-deid-{epoch}-{val/loss_total}.ckpt
├── samples/step_<step>/
│   ├── original.png
│   ├── masks.png
│   └── inpainted_composite.png
├── final_lora_weights.pt          # 학습 종료 후 자동 저장
├── lora_epoch_<NNN>.pt             # save_every_n_epochs 마다 저장
└── wandb/                          # WandB 로컬 로그
```

`*.pt` LoRA 가중치는 `save_lora_state_dict()` 형식이며, 동일 LoRA 주입 로직으로
재현해 추론 시 합칠 수 있다.

---

## 12. 변경 이력 (v1 → v2)

| 항목 | v1 (`train_inpaint_deid.py`) | v2 |
|------|------|------|
| 화질 측정 | LPIPS 학습 + 표시 | **PSNR/SSIM 모니터링**, LPIPS는 옵션 |
| 신원 손실 | margin 고정 | `cosine` / `margin` 선택 |
| 보조 손실 감쇠 | 없음 | `aux_decay_*` 도입 |
| 기본 LR | `1e-4` | **`5e-6`** (constant) |
| WandB 이미지 | 단일 비교 | 4행 그리드 + 개별 이미지 |
| DDP 안전성 | 조건부 로그로 잠재적 행 | 모든 랭크 동일 횟수 보장 |
| 최대 스텝 | 없음 | `--max_steps` 추가 |
| 얼굴 속성 보존 | 없음 | **CLIP image encoder 기반 face crop 보존 손실 추가** (`L_clip_face`) |
