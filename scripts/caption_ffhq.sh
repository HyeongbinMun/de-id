#!/bin/bash
# FFHQ 데이터셋(또는 동일 구조의 images/<split>/...) 에 대해 LLaVA 캡션을
# 사전 생성한다. 결과는 <DATA_ROOT>/<CAPTIONS_SUBDIR>/<split>/<stem>.txt 로 저장.
#
# 사용 예:
#   bash scripts/caption_ffhq.sh
#   DATA_ROOT=/dataset/deid/ffhq SPLITS="train val test" bash scripts/caption_ffhq.sh
#   MODEL=llava:7b WORKERS=8 bash scripts/caption_ffhq.sh
#   AUTOSTART_OLLAMA=1 bash scripts/caption_ffhq.sh
set -euo pipefail

# -----------------------------------------------------------------------------
# 설정 (환경변수로 오버라이드)
# -----------------------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-/dataset/deid/ffhq}"
SPLITS="${SPLITS:-train val test}"
CAPTIONS_SUBDIR="${CAPTIONS_SUBDIR:-captions}"

MODEL="${MODEL:-llava:13b}"
SERVER_URL="${SERVER_URL:-${OLLAMA_HOST:-http://localhost:11434}}"
WORKERS="${WORKERS:-4}"
RETRIES="${RETRIES:-2}"
TIMEOUT="${TIMEOUT:-180}"
LIMIT="${LIMIT:-0}"
NO_SKIP="${NO_SKIP:-0}"
AUTOSTART_OLLAMA="${AUTOSTART_OLLAMA:-0}"
INSTALL_OLLAMA="${INSTALL_OLLAMA:-0}"   # 1이면 ollama 가 없을 때 자동 설치

INSTRUCTION="${INSTRUCTION:-}"
FIX_LIBCUDA="${FIX_LIBCUDA:-1}"   # 1이면 libcuda.so.1 심볼릭링크를 커널 드라이버 버전에 맞게 보정

# -----------------------------------------------------------------------------
# CUDA 드라이버 / userspace 라이브러리 버전 일치 보정
# -----------------------------------------------------------------------------
# 일부 컨테이너 이미지에는 /lib(/usr/lib)/x86_64-linux-gnu/ 안에 여러 버전의
# libcuda.so 가 들어있고, libcuda.so.1 이 호스트 커널 드라이버와 다른 버전을
# 가리키는 경우가 있다. 이 상태에서는 ollama 의 cuInit() 가 실패해 GPU 가
# 인식되지 않고 CPU 모드로 떨어진다 (총 VRAM 0 B 로 보고됨).
fix_libcuda_symlink() {
    [ "${FIX_LIBCUDA}" = "1" ] || return 0
    [ -r /proc/driver/nvidia/version ] || return 0

    local drv
    drv=$(head -n 1 /proc/driver/nvidia/version | grep -oP '\d+\.\d+\.\d+' | head -n 1)
    [ -z "${drv}" ] && return 0

    local fixed=0
    for d in /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu; do
        local target="${d}/libcuda.so.${drv}"
        local link="${d}/libcuda.so.1"
        [ -f "${target}" ] || continue
        local current=""
        [ -L "${link}" ] && current=$(readlink "${link}")
        if [ "${current}" != "libcuda.so.${drv}" ]; then
            ln -sf "libcuda.so.${drv}" "${link}"
            echo "[CUDA] ${link} → libcuda.so.${drv} (was: ${current:-missing})"
            fixed=1
        fi
    done
    [ "${fixed}" = "1" ] && ldconfig 2>/dev/null || true
}

fix_libcuda_symlink

# -----------------------------------------------------------------------------
# 검증
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY_TOOL="${REPO_ROOT}/tools/preprocess/common/caption_with_llava.py"

if [ ! -d "${DATA_ROOT}/images" ]; then
    echo "[ERROR] ${DATA_ROOT}/images 디렉토리가 없습니다."
    exit 1
fi
if [ ! -f "${PY_TOOL}" ]; then
    echo "[ERROR] ${PY_TOOL} 가 없습니다."
    exit 1
fi

# 실제로 존재하는 split 만 추리기 (없는 split 은 경고)
VALID_SPLITS=()
for s in ${SPLITS}; do
    if [ -d "${DATA_ROOT}/images/${s}" ]; then
        VALID_SPLITS+=("${s}")
    else
        echo "[WARN] ${DATA_ROOT}/images/${s} 없음 → 건너뜀"
    fi
done

if [ "${#VALID_SPLITS[@]}" -eq 0 ]; then
    echo "[ERROR] 처리할 split 이 없습니다."
    exit 1
fi

# -----------------------------------------------------------------------------
# (옵션) Ollama 자동 기동
# -----------------------------------------------------------------------------
OLLAMA_LOG="${OLLAMA_LOG:-/tmp/ollama_serve.log}"
OLLAMA_PID=""

ensure_pkg() {
    # 필요한 패키지 한 개를 설치한다 (이미 있으면 skip).
    # 인자: 1) 명령어 이름 (예: zstd, curl, tar)  2) apt 패키지 이름 (없으면 1과 동일)
    local cmd_name="$1"
    local pkg_name="${2:-$1}"

    if command -v "${cmd_name}" >/dev/null 2>&1; then
        return 0
    fi

    if command -v apt-get >/dev/null 2>&1; then
        echo "[Ollama] '${cmd_name}' 누락 → apt-get install -y ${pkg_name}"
        DEBIAN_FRONTEND=noninteractive apt-get update -qq || true
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${pkg_name}" \
            >/tmp/apt_${pkg_name}.log 2>&1 || {
                echo "[ERROR] apt-get install ${pkg_name} 실패. 로그: /tmp/apt_${pkg_name}.log"
                tail -n 20 "/tmp/apt_${pkg_name}.log" || true
                return 1
            }
    elif command -v dnf >/dev/null 2>&1; then
        echo "[Ollama] '${cmd_name}' 누락 → dnf install -y ${pkg_name}"
        dnf install -y "${pkg_name}" >/tmp/dnf_${pkg_name}.log 2>&1 || return 1
    elif command -v yum >/dev/null 2>&1; then
        echo "[Ollama] '${cmd_name}' 누락 → yum install -y ${pkg_name}"
        yum install -y "${pkg_name}" >/tmp/yum_${pkg_name}.log 2>&1 || return 1
    elif command -v apk >/dev/null 2>&1; then
        echo "[Ollama] '${cmd_name}' 누락 → apk add --no-cache ${pkg_name}"
        apk add --no-cache "${pkg_name}" >/tmp/apk_${pkg_name}.log 2>&1 || return 1
    else
        echo "[ERROR] 지원되는 패키지 매니저(apt-get/dnf/yum/apk) 가 없습니다."
        echo "        '${cmd_name}' 를 수동 설치 후 재시도하세요."
        return 1
    fi
}

install_ollama_via_tarball() {
    # ollama install.sh 가 zstd 를 요구하는 최신 버전을 받으므로,
    # zstd 가 못 깔리는 환경을 위한 fallback 으로 .tgz 직접 다운로드.
    local arch
    arch=$(uname -m)
    case "${arch}" in
        x86_64) arch="amd64" ;;
        aarch64|arm64) arch="arm64" ;;
        *) echo "[ERROR] 지원하지 않는 아키텍처: ${arch}"; return 1 ;;
    esac

    local url="https://github.com/ollama/ollama/releases/latest/download/ollama-linux-${arch}.tgz"
    local tmpdir
    tmpdir=$(mktemp -d)
    echo "[Ollama] tarball fallback: ${url}"
    if ! curl -fsSL "${url}" -o "${tmpdir}/ollama.tgz"; then
        echo "[ERROR] tarball 다운로드 실패."
        return 1
    fi
    tar -xzf "${tmpdir}/ollama.tgz" -C "${tmpdir}" || {
        echo "[ERROR] tarball 압축 해제 실패."
        return 1
    }
    if [ -f "${tmpdir}/bin/ollama" ]; then
        install -m 0755 "${tmpdir}/bin/ollama" /usr/local/bin/ollama
    elif [ -f "${tmpdir}/ollama" ]; then
        install -m 0755 "${tmpdir}/ollama" /usr/local/bin/ollama
    else
        echo "[ERROR] tarball 안에 ollama 바이너리가 없습니다."
        ls -la "${tmpdir}" || true
        return 1
    fi
    rm -rf "${tmpdir}"
}

install_ollama_if_missing() {
    if command -v ollama >/dev/null 2>&1; then
        return 0
    fi
    if [ "${INSTALL_OLLAMA}" != "1" ]; then
        echo "[ERROR] ollama 바이너리가 없습니다 (PATH 에 없음)."
        echo "        해결책 중 하나를 선택하세요:"
        echo "          (a) 컨테이너 안에 자동 설치:  INSTALL_OLLAMA=1 AUTOSTART_OLLAMA=1 bash $0"
        echo "          (b) 직접 설치:              curl -fsSL https://ollama.com/install.sh | sh"
        echo "          (c) 호스트에서 ollama serve 띄우고 컨테이너에서 접속:"
        echo "                # host:    OLLAMA_HOST=0.0.0.0:11434 ollama serve"
        echo "                # docker:  --network=host  또는  --add-host=host.docker.internal:host-gateway"
        echo "                SERVER_URL=http://host.docker.internal:11434 bash $0"
        exit 3
    fi

    # install.sh 가 요구하는 도구 자동 설치
    ensure_pkg curl  curl  || exit 3
    ensure_pkg tar   tar   || true   # tar 는 거의 항상 있음
    ensure_pkg zstd  zstd  || {
        echo "[Ollama] zstd 설치 실패 → tarball fallback 시도..."
    }

    echo "[Ollama] install.sh 로 설치 시도..."
    if curl -fsSL https://ollama.com/install.sh | sh; then
        :
    else
        echo "[Ollama] install.sh 실패 → tarball fallback..."
    fi

    if ! command -v ollama >/dev/null 2>&1; then
        install_ollama_via_tarball || {
            echo "[ERROR] ollama 설치 실패."
            exit 3
        }
    fi

    if ! command -v ollama >/dev/null 2>&1; then
        echo "[ERROR] 모든 설치 경로 실패."
        exit 3
    fi
    echo "[Ollama] 설치 완료: $(command -v ollama) ($(ollama --version 2>/dev/null || echo '?'))"
}

start_ollama_if_needed() {
    if curl -sf -m 3 "${SERVER_URL}/api/tags" >/dev/null 2>&1; then
        echo "[Ollama] 서버 응답 OK: ${SERVER_URL}"
        return 0
    fi

    if [ "${AUTOSTART_OLLAMA}" != "1" ]; then
        echo "[ERROR] Ollama 서버에 접근 불가: ${SERVER_URL}"
        echo "        해결책 중 하나를 선택하세요:"
        echo "          (a) 자동 기동:               AUTOSTART_OLLAMA=1 bash $0"
        echo "             (바이너리 없으면)         INSTALL_OLLAMA=1 AUTOSTART_OLLAMA=1 bash $0"
        echo "          (b) 직접 데몬 띄우기:         ollama serve & 그리고 다시 실행"
        echo "          (c) 호스트의 ollama 사용:     SERVER_URL=http://host.docker.internal:11434 bash $0"
        exit 2
    fi

    install_ollama_if_missing

    echo "[Ollama] 서버를 백그라운드로 기동합니다 (log: ${OLLAMA_LOG})..."
    # 컨테이너 안에서 systemd 가 없을 수 있으므로 직접 실행
    nohup ollama serve >"${OLLAMA_LOG}" 2>&1 &
    OLLAMA_PID=$!

    for i in $(seq 1 30); do
        sleep 1
        if curl -sf -m 3 "${SERVER_URL}/api/tags" >/dev/null 2>&1; then
            echo "[Ollama] 기동 완료 (pid=${OLLAMA_PID})"
            return 0
        fi
    done

    echo "[ERROR] Ollama 서버 기동 실패. ${OLLAMA_LOG} 마지막 30줄:"
    tail -n 30 "${OLLAMA_LOG}" || true
    exit 4
}

cleanup() {
    if [ -n "${OLLAMA_PID}" ] && kill -0 "${OLLAMA_PID}" 2>/dev/null; then
        echo "[Ollama] 백그라운드 서버 종료 (pid=${OLLAMA_PID})"
        kill "${OLLAMA_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

start_ollama_if_needed

# 우리가 띄운 데몬이면 GPU 인식 결과를 검증한다 (CPU 전용이면 경고)
verify_gpu_runtime() {
    [ -f "${OLLAMA_LOG}" ] || return 0
    local total_vram
    total_vram=$(grep -oE 'total_vram="[^"]+"' "${OLLAMA_LOG}" | tail -n 1 | grep -oE '"[^"]+"')
    if echo "${total_vram}" | grep -qE '"0\s*B"'; then
        echo "[WARN] ollama 가 GPU 를 인식하지 못했습니다 (total_vram=${total_vram})."
        echo "       libcuda.so.1 심볼릭링크가 커널 드라이버 버전과 일치하는지 확인하세요."
        echo "       /proc/driver/nvidia/version: $(head -n1 /proc/driver/nvidia/version 2>/dev/null || echo '?')"
        echo "       ls -la /lib/x86_64-linux-gnu/libcuda.so.1:"
        ls -la /lib/x86_64-linux-gnu/libcuda.so.1 2>&1 | sed 's/^/         /'
        echo "       (CPU 모드로 계속 진행할 수도 있지만, llava:13b 는 GPU 없이는 매우 느립니다.)"
        if [ "${REQUIRE_GPU:-0}" = "1" ]; then
            echo "[ERROR] REQUIRE_GPU=1 → GPU 미인식 시 종료."
            exit 5
        fi
    elif [ -n "${total_vram}" ]; then
        echo "[Ollama] inference compute OK (total_vram=${total_vram})"
    fi
}
# 우리가 띄운 데몬일 때만 의미 있음 (외부 ollama 면 OLLAMA_PID 가 비어있음)
[ -n "${OLLAMA_PID}" ] && verify_gpu_runtime

# 모델이 없으면 pull
if ! curl -sf -m 5 "${SERVER_URL}/api/tags" \
        | grep -q "\"name\":\"${MODEL}\""; then
    echo "[Ollama] '${MODEL}' 모델이 캐시에 없음 → pull 합니다 (네트워크 필요)..."
    if command -v ollama >/dev/null 2>&1; then
        OLLAMA_HOST="${SERVER_URL}" ollama pull "${MODEL}"
    else
        # ollama CLI 가 없는 환경에서는 HTTP API 로 pull 트리거
        curl -fsS -X POST "${SERVER_URL}/api/pull" \
            -H "Content-Type: application/json" \
            -d "{\"name\":\"${MODEL}\"}" \
            | tail -n 5 || true
    fi
fi

# 모델 워밍업: 1) GPU 메모리에 적재되는지 확인, 2) 첫 요청의 cold-start 지연 제거
echo "[Ollama] '${MODEL}' 워밍업 중 (GPU 적재 확인)..."
WARMUP_RESP=$(curl -sf -m 60 "${SERVER_URL}/api/generate" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"hi\",\"stream\":false,\"keep_alive\":\"30m\",\"options\":{\"num_predict\":4}}" \
    || echo "")
if [ -z "${WARMUP_RESP}" ]; then
    echo "[WARN] 워밍업 요청 실패 - 캡션 생성 단계에서 다시 시도됩니다."
fi

# /api/ps 로 현재 적재된 모델 정보 확인 (size_vram>0 이면 GPU 사용)
PS_INFO=$(curl -sf -m 5 "${SERVER_URL}/api/ps" 2>/dev/null || echo "")
echo "[Ollama] /api/ps:"
echo "${PS_INFO}" | python3 -c "
import json, sys
try:
    data = json.loads(sys.stdin.read() or '{}')
except Exception:
    data = {}
for m in data.get('models', []):
    name = m.get('name', '?')
    size = m.get('size', 0)
    size_vram = m.get('size_vram', 0)
    pct = (size_vram / size * 100) if size else 0
    print(f'  - {name}: total={size/1e9:.2f}GB, vram={size_vram/1e9:.2f}GB ({pct:.0f}% on GPU)')
    if size and size_vram == 0:
        print('    [WARN] 100% CPU - GPU 미사용. 위의 verify_gpu_runtime 경고 참조.')
" 2>/dev/null || echo "  (parsing failed)"

# -----------------------------------------------------------------------------
# 캡션 생성
# -----------------------------------------------------------------------------
echo
echo "===================================================================="
echo " FFHQ Caption Generation"
echo "===================================================================="
echo " DATA_ROOT       : ${DATA_ROOT}"
echo " SPLITS          : ${VALID_SPLITS[*]}"
echo " CAPTIONS_SUBDIR : ${CAPTIONS_SUBDIR}"
echo " MODEL           : ${MODEL}"
echo " SERVER_URL      : ${SERVER_URL}"
echo " WORKERS         : ${WORKERS}"
echo " NO_SKIP         : ${NO_SKIP}"
echo " LIMIT           : ${LIMIT}"
echo "===================================================================="

ARGS=(
    --dataset_dir "${DATA_ROOT}"
    --splits "${VALID_SPLITS[@]}"
    --captions_subdir "${CAPTIONS_SUBDIR}"
    --model "${MODEL}"
    --server_url "${SERVER_URL}"
    --workers "${WORKERS}"
    --retries "${RETRIES}"
    --timeout "${TIMEOUT}"
)
[ "${NO_SKIP}" = "1" ] && ARGS+=(--no_skip)
[ "${LIMIT}" -gt 0 ] 2>/dev/null && ARGS+=(--limit "${LIMIT}")
[ -n "${INSTRUCTION}" ] && ARGS+=(--instruction "${INSTRUCTION}")

python "${PY_TOOL}" "${ARGS[@]}"

echo
echo "[Done] captions saved under: ${DATA_ROOT}/${CAPTIONS_SUBDIR}/<split>/"
for s in "${VALID_SPLITS[@]}"; do
    cap_dir="${DATA_ROOT}/${CAPTIONS_SUBDIR}/${s}"
    if [ -d "${cap_dir}" ]; then
        n_img=$(find "${DATA_ROOT}/images/${s}" -maxdepth 1 -type f \
                  \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \
                     -o -iname "*.bmp" -o -iname "*.webp" \) 2>/dev/null | wc -l)
        n_cap=$(find "${cap_dir}" -maxdepth 1 -type f -name "*.txt" 2>/dev/null | wc -l)
        echo "  - ${s}: ${n_cap} / ${n_img} captioned"
    fi
done
