#!/bin/bash
set -euo pipefail

# libcuda 심링크 자동 수정
DRIVER_VER=$(cat /proc/driver/nvidia/version 2>/dev/null | head -1 | grep -oP '\d+\.\d+\.\d+' || true)
[ -n "${DRIVER_VER}" ] && [ -f "/usr/lib/x86_64-linux-gnu/libcuda.so.${DRIVER_VER}" ] && \
    ln -sf "/usr/lib/x86_64-linux-gnu/libcuda.so.${DRIVER_VER}" /usr/lib/x86_64-linux-gnu/libcuda.so.1

export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_DEBUG=WARN

DATA_ROOT="${DATA_ROOT:-/dataset/deid/ffhq}"
OUTPUT_DIR="${OUTPUT_DIR:-/dataset/deid/models/inpaint_deid_lora}"
FEATURE_MODEL_PATH="${FEATURE_MODEL_PATH:-/workspace/model/weights/sscd_disc_mixup.torchvision.pt}"
FACE_ID_MODEL_PATH="${FACE_ID_MODEL_PATH:-}"

SINGLE_FILE_CKPT="${SINGLE_FILE_CKPT:-/dataset/deid/models/Realistic_Vision_V5.1-inpainting.safetensors}"
VAE_PATH="${VAE_PATH:-stabilityai/sd-vae-ft-mse}"
SD_MODEL="${SD_MODEL:-runwayml/stable-diffusion-inpainting}"

[ ! -d "${DATA_ROOT}/images/train" ] && echo "[ERROR] ${DATA_ROOT}/images/train 없음" && exit 1
[ ! -d "${DATA_ROOT}/masks/train" ] && echo "[ERROR] ${DATA_ROOT}/masks/train 없음" && exit 1
mkdir -p "${OUTPUT_DIR}"

ARGS=(
    --data_root "${DATA_ROOT}"
    --output_dir "${OUTPUT_DIR}"
    --feature_model_path "${FEATURE_MODEL_PATH}"
    --vae_path "${VAE_PATH}"
    --lora_rank "${LORA_RANK:-16}"
    --lambda_identity "${LAMBDA_IDENTITY:-0.3}"
    --lambda_feature "${LAMBDA_FEATURE:-0.5}"
    --lambda_lpips "${LAMBDA_LPIPS:-0.5}"
    --id_margin "${ID_MARGIN:-0.3}"
    --timestep_threshold "${TIMESTEP_THRESHOLD:-250}"
    --aux_warmup_steps "${AUX_WARMUP_STEPS:-2000}"
    --train_batch_size "${BATCH_SIZE:-4}"
    --gradient_accumulation_steps "${GRAD_ACCUM:-2}"
    --num_train_epochs "${NUM_EPOCHS:-50}"
    --learning_rate "${LEARNING_RATE:-5e-6}"
    --lr_scheduler "${LR_SCHEDULER:-constant}"
    --lr_warmup_steps "${LR_WARMUP_STEPS:-0}"
    --num_gpus "${NUM_GPUS:-2}"
    --mixed_precision fp16
    --gradient_checkpointing
)

if [ -f "${SINGLE_FILE_CKPT}" ]; then
    ARGS+=(--single_file_ckpt "${SINGLE_FILE_CKPT}")
else
    echo "[WARN] ${SINGLE_FILE_CKPT} 없음 → fallback: ${SD_MODEL}"
    ARGS+=(--pretrained_model_name_or_path "${SD_MODEL}")
fi

[ -n "${FACE_ID_MODEL_PATH}" ] && ARGS+=(--face_id_model_path "${FACE_ID_MODEL_PATH}")

python /workspace/tools/train/train_inpaint_deid.py "${ARGS[@]}"
