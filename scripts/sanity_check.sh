#!/bin/bash
# 새로 빌드된 컨테이너 안에서 한 번 돌려서 환경이 정상인지 빠르게 검증.
#
# 검사 항목:
#   1) NVIDIA 드라이버 (nvidia-smi)
#   2) PyTorch CUDA 사용 가능 여부 + 버전
#   3) train_inpaint_deid_v2 의 핵심 import 가 모두 동작
#   4) ollama 바이너리 + GPU 인식 (선택: --skip-ollama 로 끄기)
#
# 사용:
#   bash scripts/sanity_check.sh
#   bash scripts/sanity_check.sh --skip-ollama
set -uo pipefail

SKIP_OLLAMA=0
for a in "$@"; do
    [ "$a" = "--skip-ollama" ] && SKIP_OLLAMA=1
done

PASS=()
FAIL=()
mark_pass() { PASS+=("$1"); echo "[PASS] $1"; }
mark_fail() { FAIL+=("$1"); echo "[FAIL] $1"; }

echo
echo "===================================================================="
echo " 1) NVIDIA driver"
echo "===================================================================="
if nvidia-smi -L >/dev/null 2>&1; then
    nvidia-smi -L | sed 's/^/  /'
    mark_pass "nvidia-smi"
else
    nvidia-smi 2>&1 | head -5
    mark_fail "nvidia-smi"
fi

echo
echo "===================================================================="
echo " 2) PyTorch / CUDA"
echo "===================================================================="
python3 - <<'PY'
import sys
try:
    import torch
    print(f"  torch.__version__       = {torch.__version__}")
    print(f"  torch.version.cuda      = {torch.version.cuda}")
    print(f"  torch.cuda.is_available = {torch.cuda.is_available()}")
    print(f"  torch.cuda.device_count = {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"    [{i}] {torch.cuda.get_device_name(i)} "
                  f"({torch.cuda.get_device_properties(i).total_memory/1024**3:.1f} GiB)")
        x = torch.ones(8, device="cuda") + 1
        assert x.sum().item() == 16
        print("  cuda tensor smoke-test  = OK")
    sys.exit(0 if torch.cuda.is_available() else 11)
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(12)
PY
RC=$?
[ $RC -eq 0 ] && mark_pass "torch + cuda" || mark_fail "torch + cuda (rc=$RC)"

echo
echo "===================================================================="
echo " 3) train_inpaint_deid_v2 import"
echo "===================================================================="
python3 - <<'PY'
import sys, importlib
mods = [
    "diffusers", "transformers", "pytorch_lightning",
    "piqa", "wandb", "lpips", "requests", "tqdm",
]
fail = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  import {m:<22s} ok")
    except Exception as e:
        print(f"  import {m:<22s} FAIL: {e}")
        fail.append(m)

# v2 학습 모듈을 실제로 부분 import (LoRA / Dataset / extractor)
try:
    sys.path.insert(0, "/workspace")
    from tools.train.train_inpaint_deid_v2 import (
        LoRALinear, inject_lora, InpaintDeIdDataset, FaceIdentityExtractor
    )
    print("  import train_inpaint_deid_v2 symbols ok")
except Exception as e:
    print(f"  train_inpaint_deid_v2 FAIL: {e}")
    fail.append("train_inpaint_deid_v2")

sys.exit(0 if not fail else 13)
PY
RC=$?
[ $RC -eq 0 ] && mark_pass "v2 module imports" || mark_fail "v2 module imports (rc=$RC)"

echo
if [ "$SKIP_OLLAMA" = "1" ]; then
    echo "(skipping ollama checks: --skip-ollama)"
else
    echo "===================================================================="
    echo " 4) ollama"
    echo "===================================================================="
    if ! command -v ollama >/dev/null 2>&1; then
        echo "  [ERROR] ollama 바이너리 없음. dockerfile 빌드가 누락됐을 수 있음."
        mark_fail "ollama binary"
    else
        echo "  binary: $(command -v ollama)"
        ollama --version 2>&1 | sed 's/^/  /'

        if ! curl -sf -m 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "  데몬 미실행 → 임시로 띄움 (60s)"
            nohup ollama serve >/tmp/ollama_sanity.log 2>&1 &
            OPID=$!
            for i in $(seq 1 30); do
                sleep 1
                curl -sf -m 2 http://localhost:11434/api/tags >/dev/null 2>&1 && break
            done
        fi

        if curl -sf -m 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
            mark_pass "ollama daemon"

            # GPU 인식 검증 (CPU 전용이면 fail)
            VRAM=$(grep -oE 'total_vram="[^"]+"' /tmp/ollama_sanity.log /tmp/ollama_serve.log 2>/dev/null | tail -n 1 || echo "")
            if echo "$VRAM" | grep -qE 'total_vram="0\s*B"'; then
                echo "  [WARN] GPU 미인식 (total_vram=0 B). libcuda symlink 확인 필요."
                mark_fail "ollama GPU"
            else
                [ -n "$VRAM" ] && echo "  $VRAM"
                mark_pass "ollama GPU"
            fi
        else
            tail -n 20 /tmp/ollama_sanity.log 2>/dev/null
            mark_fail "ollama daemon startup"
        fi

        [ -n "${OPID:-}" ] && kill "$OPID" 2>/dev/null || true
    fi
fi

echo
echo "===================================================================="
echo " Summary"
echo "===================================================================="
echo "  PASS (${#PASS[@]}): ${PASS[*]}"
echo "  FAIL (${#FAIL[@]}): ${FAIL[*]}"
[ "${#FAIL[@]}" -eq 0 ] && exit 0 || exit 1
