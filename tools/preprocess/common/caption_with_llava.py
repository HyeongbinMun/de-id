"""
Ollama LLaVA 기반 이미지 캡션 생성기
=====================================
얼굴 인페인팅 LoRA 학습용 per-image 텍스트 프롬프트를 사전 생성한다.
출력은 ``<dataset_dir>/captions/<split>/<stem>.txt`` 형식으로 저장된다.

사용 예:
    # FFHQ 기본 split 전체
    python tools/preprocess/common/caption_with_llava.py \
        --dataset_dir /dataset/deid/ffhq \
        --model llava:13b \
        --workers 4

    # 특정 split만
    python tools/preprocess/common/caption_with_llava.py \
        --dataset_dir /dataset/deid/ffhq \
        --splits train val \
        --workers 8

전제: 호스트에 ollama 데몬이 떠있어야 한다 (`ollama serve`).
모델 사전 다운로드: `ollama pull llava:13b`
"""

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_INSTRUCTION = (
    "You are generating concise visual captions for face image generation training. "
    "Describe ONLY visible attributes of the person and the photo in a single sentence "
    "(max 40 words). Cover, when visible: approximate age range, gender, ethnicity / "
    "skin tone, hair color and style, facial hair, glasses or accessories, expression, "
    "pose, lighting, and image style. "
    "Do NOT name any real or fictional person, do NOT mention celebrities, do NOT add "
    "introductions like 'This image shows'. Start directly with 'a photo of'."
)


def encode_image_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def request_caption(
    server_url: str,
    model: str,
    prompt: str,
    image_b64: str,
    timeout: int = 180,
    options: Optional[dict] = None,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": options or {"temperature": 0.2, "num_predict": 96},
    }
    resp = requests.post(
        f"{server_url.rstrip('/')}/api/generate",
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    text = (data.get("response") or "").strip()
    text = text.replace("\n", " ").replace("\r", " ").strip()
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def caption_one(
    img_path: Path,
    out_path: Path,
    server_url: str,
    model: str,
    instruction: str,
    retries: int = 2,
    timeout: int = 180,
) -> Tuple[Path, Optional[str], Optional[str]]:
    """Returns (img_path, caption_or_None, error_or_None)."""
    try:
        b64 = encode_image_b64(img_path)
    except Exception as e:
        return img_path, None, f"encode_failed: {e}"

    last_err = None
    for attempt in range(retries + 1):
        try:
            caption = request_caption(server_url, model, instruction, b64, timeout=timeout)
            if not caption:
                last_err = "empty_response"
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(caption + "\n")
            return img_path, caption, None
        except Exception as e:
            last_err = str(e)
            time.sleep(min(2.0 * (attempt + 1), 5.0))

    return img_path, None, last_err


def collect_split_jobs(
    dataset_dir: Path,
    splits: List[str],
    captions_subdir: str,
    skip_existing: bool,
) -> List[Tuple[Path, Path]]:
    jobs: List[Tuple[Path, Path]] = []
    for split in splits:
        img_dir = dataset_dir / "images" / split if split else dataset_dir / "images"
        if not img_dir.is_dir():
            print(f"  [SKIP] image dir not found: {img_dir}")
            continue

        cap_dir = (
            dataset_dir / captions_subdir / split if split
            else dataset_dir / captions_subdir
        )
        cap_dir.mkdir(parents=True, exist_ok=True)

        for p in sorted(img_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            out = cap_dir / f"{p.stem}.txt"
            if skip_existing and out.exists() and out.stat().st_size > 0:
                continue
            jobs.append((p, out))
    return jobs


def main():
    parser = argparse.ArgumentParser(
        description="Ollama LLaVA로 이미지 캡션을 생성한다."
    )
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="데이터셋 루트 (images/<split> 구조)")
    parser.add_argument("--splits", type=str, nargs="*", default=None,
                        help="처리할 split 목록 (미지정 시 images/ 하위 자동 탐색)")
    parser.add_argument("--captions_subdir", type=str, default="captions",
                        help="<dataset_dir>/<captions_subdir>/<split>/<stem>.txt 로 저장")
    parser.add_argument("--server_url", type=str,
                        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
                        help="ollama 서버 주소")
    parser.add_argument("--model", type=str, default="llava:13b",
                        help="ollama 모델 이름 (사전에 ollama pull 필요)")
    parser.add_argument("--workers", type=int, default=4,
                        help="동시 요청 수 (서버 처리량에 맞춰 조정)")
    parser.add_argument("--retries", type=int, default=2,
                        help="요청 실패 시 재시도 횟수")
    parser.add_argument("--timeout", type=int, default=180,
                        help="단일 요청 타임아웃(초)")
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION,
                        help="LLaVA에 전달할 지시문")
    parser.add_argument("--no_skip", action="store_true",
                        help="이미 존재하는 캡션도 다시 생성")
    parser.add_argument("--limit", type=int, default=0,
                        help=">0이면 처음 N개 이미지만 처리 (디버깅용)")
    parser.add_argument("--errors_log", type=str, default=None,
                        help="실패한 이미지 경로/사유를 기록할 파일")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.is_dir():
        print(f"[ERROR] dataset_dir not found: {dataset_dir}")
        sys.exit(1)

    if args.splits is None:
        images_root = dataset_dir / "images"
        if not images_root.is_dir():
            print(f"[ERROR] {images_root} not found. --splits 로 명시하세요.")
            sys.exit(1)
        splits = sorted(
            d.name for d in images_root.iterdir() if d.is_dir()
        )
        if not splits:
            splits = [""]
    else:
        splits = args.splits

    print(f"[Caption] dataset_dir = {dataset_dir}")
    print(f"[Caption] splits      = {splits}")
    print(f"[Caption] model       = {args.model}")
    print(f"[Caption] server      = {args.server_url}")

    try:
        ping = requests.get(f"{args.server_url.rstrip('/')}/api/tags", timeout=5)
        ping.raise_for_status()
    except Exception as e:
        print(f"[ERROR] ollama 서버에 접근할 수 없습니다: {e}")
        print("        `ollama serve` 가 실행 중인지, --server_url 이 올바른지 확인하세요.")
        sys.exit(2)

    jobs = collect_split_jobs(
        dataset_dir, splits, args.captions_subdir, skip_existing=not args.no_skip
    )

    if args.limit > 0:
        jobs = jobs[: args.limit]

    if not jobs:
        print("[Caption] 생성할 이미지가 없습니다 (전부 skip 또는 디렉토리 비어있음).")
        return

    print(f"[Caption] 총 {len(jobs):,}개 이미지에 대해 캡션 생성 시작.")

    errors: List[Tuple[Path, str]] = []
    success = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [
            ex.submit(
                caption_one,
                img, out,
                args.server_url, args.model, args.instruction,
                args.retries, args.timeout,
            )
            for img, out in jobs
        ]
        with tqdm(total=len(futures), desc="captioning", dynamic_ncols=True) as pbar:
            for fut in as_completed(futures):
                img_path, caption, err = fut.result()
                if err is None:
                    success += 1
                else:
                    errors.append((img_path, err))
                pbar.update(1)
                pbar.set_postfix(ok=success, fail=len(errors))

    print(f"\n[Caption] 성공: {success:,} / 실패: {len(errors):,}")

    if errors:
        log_path = (
            Path(args.errors_log) if args.errors_log
            else dataset_dir / args.captions_subdir / "errors.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            for p, e in errors:
                f.write(f"{p}\t{e}\n")
        print(f"[Caption] 실패 로그: {log_path}")
        print("        실패한 항목은 다시 동일 명령으로 재시도하면 자동으로 이어서 생성됩니다.")


if __name__ == "__main__":
    main()
