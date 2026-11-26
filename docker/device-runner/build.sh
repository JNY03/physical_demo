#!/usr/bin/env bash
# build.sh — ds-device-runner 이미지 빌드 (기기 단계 실행체, arm64 + amd64). 저장소 어디서 불러도 된다.
#
#   bash docker/device-runner/build.sh                 amd64 만 빌드해 로컬 도커에 싣는다 (ds-device-runner:<태그>) — 시험용
#   bash docker/device-runner/build.sh --push          arm64 + amd64 를 사설 레지스트리로 (127.0.0.1:5000/ds-device-runner:<태그>)
#   bash docker/device-runner/build.sh --arm64-load    arm64 만 로컬에 (qemu 로 띄워 보는 시험용)
#
# 환경변수
#   DS_REGISTRY      밀어 넣을 레지스트리 (기본 127.0.0.1:5000 — 이 기기의 ds-registry. 파이는 같은 것을 테일넷 주소로 당긴다)
#   TAG              태그 (기본: git 짧은 해시, 고친 채 커밋 안 한 파일이 있으면 -dirty)
#   MODELS           모델 목록 파일 (기본 docker/device-runner/models.txt)
#   DS_PYTHON        스테이징을 돌릴 Python (PyYAML 필요. 기본: 저장소 venv → python3)
#
# 태그는 늘 git 해시다 — 레지스트리(vision/model_registry.yaml functions.*.stages.device.image)가 그 태그를 가리킨다.
# 바꾼 실행기 · 가중치를 파이에 올리려면: 커밋 → 이 스크립트 --push → 레지스트리 image 태그를 새 해시로.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
REG="${DS_REGISTRY:-127.0.0.1:5000}"
MODE="${1:---load}"
PY="${DS_PYTHON:-}"
if [ -z "$PY" ]; then
    if [ -x "$REPO/venv/bin/python" ]; then PY="$REPO/venv/bin/python"; else PY=python3; fi
fi
if [ -z "${TAG:-}" ]; then
    TAG="$(git -C "$REPO" rev-parse --short HEAD)"
    [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] && TAG="$TAG-dirty"
fi
CTX="$(mktemp -d "${TMPDIR:-/tmp}/ds-device-runner.XXXXXX")"
trap 'rm -rf "$CTX"' EXIT
"$PY" "$HERE/stage.py" --out "$CTX" --models "${MODELS:-$HERE/models.txt}"

case "$MODE" in
--push)
    if ! curl -fsS "http://$REG/v2/" >/dev/null 2>&1; then
        echo "레지스트리 $REG 가 안 열림 — 이 기기라면: docker start ds-registry" >&2; exit 1
    fi
    docker buildx build --platform linux/arm64,linux/amd64 -f "$HERE/Dockerfile" \
        -t "$REG/ds-device-runner:$TAG" --push "$CTX"
    echo "밀어 넣음: $REG/ds-device-runner:$TAG (linux/arm64 · linux/amd64)"
    ;;
--arm64-load)
    docker buildx build --platform linux/arm64 -f "$HERE/Dockerfile" -t "ds-device-runner:$TAG-arm64" --load "$CTX"
    echo "로컬: ds-device-runner:$TAG-arm64"
    ;;
--load)
    docker buildx build --platform linux/amd64 -f "$HERE/Dockerfile" -t "ds-device-runner:$TAG" --load "$CTX"
    echo "로컬: ds-device-runner:$TAG"
    ;;
*)
    echo "모르는 선택 $MODE (--load | --push | --arm64-load)" >&2; exit 2
    ;;
esac
