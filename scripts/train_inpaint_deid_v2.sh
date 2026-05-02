#!/bin/bash
set -euo pipefail

DRIVER_VER=$(cat /proc/driver/nvidia/version 2>/dev/null | head -1 | grep -oP '\d+\.\d+\.\d+' || true)
[ -n "${DRIVER_VER}" ] && [ -f "/usr/lib/x86_64-linux-gnu/libcuda.so.${DRIVER_VER}" ] && \
    ln -sf "/usr/lib/x86_64-linux-gnu/libcuda.so.${DRIVER_VER}" /usr/lib/x86_64-linux-gnu/libcuda.so.1

export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_DEBUG=WARN

DATA_ROOT="${DATA_ROOT:-/dataset/deid/ffhq}"
OUTPUT_DIR="${OUTPUT_DIR:-/dataset/deid/models/inpaint_deid_lora_v2}"
FEATURE_MODEL_PATH="${FEATURE_MODEL_PATH:-/workspace/model/weights/sscd_disc_mixup.torchvision.pt}"
FACE_ID_MODEL_PATH="${FACE_ID_MODEL_PATH:-}"

SINGLE_FILE_CKPT="${SINGLE_FILE_CKPT:-/dataset/deid/models/Realistic_Vision_V5.1-inpainting.safetensors}"
VAE_PATH="${VAE_PATH:-stabilityai/sd-vae-ft-mse}"
SD_MODEL="${SD_MODEL:-runwayml/stable-diffusion-inpainting}"

[ ! -d "${DATA_ROOT}/images/train" ] && echo "[ERROR] ${DATA_ROOT}/images/train 없음" && exit 1
[ ! -d "${DATA_ROOT}/masks/train" ] && echo "[ERROR] ${DATA_ROOT}/masks/train 없음" && exit 1
if [ ! -d "${DATA_ROOT}/captions/train" ]; then
    echo "[WARN] ${DATA_ROOT}/captions/train 없음 → 모든 학습 샘플이 fallback 프롬프트로 학습됩니다."
    echo "       per-person 프롬프트 적용을 위해 다음 명령으로 사전 캡션을 생성하세요:"
    echo "         python tools/preprocess/common/caption_with_llava.py \\"
    echo "             --dataset_dir ${DATA_ROOT} --model llava:13b --workers 4"
fi
mkdir -p "${OUTPUT_DIR}"

ARGS=(
    --data_root "${DATA_ROOT}"
    --output_dir "${OUTPUT_DIR}"
    --feature_model_path "${FEATURE_MODEL_PATH}"
    --vae_path "${VAE_PATH}"
    --captions_subdir "${CAPTIONS_SUBDIR:-captions}"
    --identity_loss_mode "${IDENTITY_LOSS_MODE:-cosine}"
    --lora_rank "${LORA_RANK:-16}"
    --lambda_identity "${LAMBDA_IDENTITY:-0.5}"
    --lambda_feature "${LAMBDA_FEATURE:-0.5}"
    --id_margin "${ID_MARGIN:-0.3}"
    --timestep_threshold "${TIMESTEP_THRESHOLD:-250}"
    --aux_warmup_steps "${AUX_WARMUP_STEPS:-2000}"
    --aux_decay_start_step "${AUX_DECAY_START:-0}"
    --aux_decay_end_step "${AUX_DECAY_END:-0}"
    --aux_decay_min "${AUX_DECAY_MIN:-0.25}"
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

[ -n "${MAX_STEPS:-}" ] && ARGS+=(--max_steps "${MAX_STEPS}")
[ "${USE_LPIPS_LOSS:-0}" = "1" ] && ARGS+=(--use_lpips_loss --lambda_lpips "${LAMBDA_LPIPS:-0.3}")
if [ "${USE_CLIP_FACE_LOSS:-1}" = "1" ]; then
    ARGS+=(
        --use_clip_face_loss
        --lambda_clip_face "${LAMBDA_CLIP_FACE:-0.3}"
        --clip_model_name "${CLIP_MODEL_NAME:-openai/clip-vit-base-patch32}"
    )
fi

if [ -f "${SINGLE_FILE_CKPT}" ]; then
    ARGS+=(--single_file_ckpt "${SINGLE_FILE_CKPT}")
else
    echo "[WARN] ${SINGLE_FILE_CKPT} 없음 → fallback: ${SD_MODEL}"
    ARGS+=(--pretrained_model_name_or_path "${SD_MODEL}")
fi

[ -n "${FACE_ID_MODEL_PATH}" ] && ARGS+=(--face_id_model_path "${FACE_ID_MODEL_PATH}")

python /workspace/tools/train/train_inpaint_deid_v2.py "${ARGS[@]}"
