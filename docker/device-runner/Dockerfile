# syntax=docker/dockerfile:1
#
# ds-device-runner — 기기 단계 실행체 (vision/device_stage.py). 파이 5(arm64, GPU 없음)에서 k3s 파드로 돈다.
#
# 이 Dockerfile 은 저장소 루트가 아니라 **스테이징 폴더**를 컨텍스트로 받는다 — 넣을 가중치를 고르기 위해서다.
# 손으로 만들지 말고 build.sh 를 쓴다(stage.py 가 폴더를 만들고 buildx 를 부른다):
#   bash docker/device-runner/build.sh                       # amd64 만 빌드해 로컬에 (--load) — 시험용
#   bash docker/device-runner/build.sh --push                # arm64 + amd64 를 사설 레지스트리 127.0.0.1:5000 으로
# 스테이징 폴더 모양:
#   requirements.txt   (docker/device-runner/requirements.txt 복사)
#   vision/device_stage.py · vision/runners/*.py · vision/model_registry.yaml · vision/vision_config.yaml
#   models/...         (docker/device-runner/models.txt 에 적은 모델의 파일만)
#
# 원칙
# - torch · ultralytics 가 없다. numpy · opencv-headless · onnxruntime(CPU) · ncnn · redis · PyYAML 뿐 — 실행기는
#   이것들만 import 한다(vision/runners/README.md). 그 밖이 필요한 실행기는 이 이미지에 넣지 않는다.
# - 아키텍처를 박은 것이 없다. 휠이 아키텍처별 판으로 깔린다 → buildx --platform linux/arm64,linux/amd64.
# - 레이어 순서: 의존성(잘 안 바뀜) → 가중치 → 코드(자주 바뀜). 코드를 고쳐도 가중치 층을 다시 받지 않는다.
# - 루트가 아닌 사용자(1000)로 돈다. 쓰는 파일이 없다(결과는 Redis 로만).

ARG PYTHON_IMAGE=python:3.12-slim-bookworm
FROM ${PYTHON_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DS_HOME=/ds \
    OMP_NUM_THREADS=2 \
    TZ=Asia/Seoul

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-deps -r /tmp/requirements.txt \
 && rm -f /tmp/requirements.txt \
 && find /usr/local/lib -name '__pycache__' -type d -prune -exec rm -rf {} + \
 && find /usr/local/lib -name 'tests' -path '*site-packages/numpy/*' -type d -prune -exec rm -rf {} +

WORKDIR /ds
COPY models/ /ds/models/
COPY vision/ /ds/vision/

USER 1000:1000
# k3s 는 레지스트리 command 로 이것을 덮어쓴다 (functions.<기능>.stages.device.command)
CMD ["python", "/ds/vision/device_stage.py", "--help"]
