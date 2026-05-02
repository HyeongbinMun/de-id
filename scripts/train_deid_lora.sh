#!/bin/bash
set -euo pipefail

# ---- libcuda 심링크 자동 수정 (컨테이너 재시작 시 복구) ----
DRIVER_VER=$(cat /proc/driver/nvidia/version 2>/dev/null | head -1 | grep -oP '\d+\.\d+\.\d+' || true)
if [ -n "${DRIVER_VER}" ] && [ -f "/usr/lib/x86_64-linux-gnu/libcuda.so.${DRIVER_VER}" ]; then
    ln -sf "/usr/lib/x86_64-linux-gnu/libcuda.so.${DRIVER_VER}" /usr/lib/x86_64-linux-gnu/libcuda.so.1
    echo "[INFO] libcuda.so.1 → libcuda.so.${DRIVER_VER}"
fi

# ---- NCCL 안정성 설정 ----
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN

# ---- 주요 설정 (환경변수로 오버라이드 가능) ----
FFHQ_DIR="${FFHQ_DIR:-/dataset/deid/ffhq/images/train}"
OUTPUT_DIR="${OUTPUT_DIR:-/dataset/deid/models/deid_lora}"
SD_MODEL="${SD_MODEL:-stable-diffusion-v1-5/stable-diffusion-v1-5}"
FEATURE_MODEL_PATH="${FEATURE_MODEL_PATH:-/workspace/model/weights/sscd_disc_mixup.torchvision.pt}"
FACE_ID_MODEL_PATH="${FACE_ID_MODEL_PATH:-}"

NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LORA_RANK="${LORA_RANK:-16}"

# Loss 가중치
LAMBDA_IDENTITY="${LAMBDA_IDENTITY:-0.3}"
LAMBDA_FEATURE="${LAMBDA_FEATURE:-0.5}"
LAMBDA_LPIPS="${LAMBDA_LPIPS:-1.0}"
LAMBDA_RECON="${LAMBDA_RECON:-0.5}"

# Anti-collapse
ID_MARGIN="${ID_MARGIN:-0.3}"
TIMESTEP_THRESHOLD="${TIMESTEP_THRESHOLD:-250}"
AUX_WARMUP_STEPS="${AUX_WARMUP_STEPS:-2000}"

# ---- 검증 ----
if [ ! -d "${FFHQ_DIR}" ]; then
    echo "[ERROR] FFHQ 디렉토리 없음: ${FFHQ_DIR}" && exit 1
fi
mkdir -p "${OUTPUT_DIR}"

# ---- 실행 ----
ARGS=(
    --pretrained_model_name_or_path "${SD_MODEL}"
    --data_dir "${FFHQ_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --feature_model_path "${FEATURE_MODEL_PATH}"
    --lora_rank "${LORA_RANK}"
    --lambda_identity "${LAMBDA_IDENTITY}"
    --lambda_feature "${LAMBDA_FEATURE}"
    --lambda_lpips "${LAMBDA_LPIPS}"
    --lambda_recon "${LAMBDA_RECON}"
    --id_margin "${ID_MARGIN}"
    --timestep_threshold "${TIMESTEP_THRESHOLD}"
    --aux_warmup_steps "${AUX_WARMUP_STEPS}"
    --train_batch_size "${BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACCUM}"
    --num_train_epochs "${NUM_EPOCHS}"
    --learning_rate "${LEARNING_RATE}"
    --num_gpus "${NUM_GPUS}"
    --mixed_precision fp16
    --gradient_checkpointing
)

[ -n "${FACE_ID_MODEL_PATH}" ] && ARGS+=(--face_id_model_path "${FACE_ID_MODEL_PATH}")

python /workspace/tools/train/train_new_deid.py "${ARGS[@]}"
