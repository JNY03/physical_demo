# syntax=docker/dockerfile:1
#
# ds-vision-gpu-base — CUDA 12.6 + Python 3.14.4 + torch/torchvision cu126 + onnxruntime-gpu(CUDA 12) + ultralytics
#
# 빌드 (빌드 컨텍스트는 이 디렉터리만):
#   docker build -t 127.0.0.1:5000/ds-vision-gpu-base:dev --target ds-vision-gpu-base docker/vision-gpu
#
# 원칙 (Docker_배포_계획 §3 · §6 · §9):
# - 가중치는 넣지 않는다. models/ 는 실행 때 /ds/models 로 읽기 전용 마운트.
# - devel 이미지를 쓰지 않는다. CUDA 사용자 공간 라이브러리(cudart · cublas · cudnn · cufft · curand …)는
#   torch cu126 휠이 끌고 오는 nvidia-*-cu12 휠에 이미 들어 있으므로, 베이스는 가장 얇은 `base` 변형으로 둔다
#   (`runtime` 변형은 같은 라이브러리를 /usr/local/cuda 에 한 벌 더 넣어 ~2–3 GB 만 늘린다).
#   드라이버 라이브러리(libcuda · libdxcore, WSL 은 /usr/lib/wsl/lib)는 NVIDIA Container Toolkit 이 넣어 준다.
# - 레이어 순서는 잘 안 바뀌는 것 → 자주 바뀌는 것: Python → torch → onnxruntime → ultralytics/나머지.

ARG CUDA_BASE=nvidia/cuda:12.6.3-base-ubuntu24.04

FROM ghcr.io/astral-sh/uv:0.13.7 AS uv

FROM ${CUDA_BASE} AS ds-vision-gpu-base

ARG PYTHON_VERSION=3.14.4
ENV DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONUNBUFFERED=1 \
    DS_HOME=/ds \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    YOLO_OFFLINE=1

# 시스템 패키지: 최소 (opencv-headless 는 libGL 이 필요 없다. libglib 은 cv2 가 찾는다)
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libglib2.0-0t64 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

# 1) Python — 호스트 venv 와 같은 3.14.4 (python-build-standalone, uv 가 받음)
RUN uv python install ${PYTHON_VERSION} \
 && uv venv --python ${PYTHON_VERSION} /opt/venv

COPY constraints.txt /opt/ds/constraints.txt

# 2) torch · torchvision (cu126) — 가장 크고 가장 안 바뀌는 층
RUN uv pip install --python /opt/venv/bin/python \
      --index-url https://download.pytorch.org/whl/cu126 \
      --extra-index-url https://pypi.org/simple \
      --index-strategy unsafe-best-match \
      -c /opt/ds/constraints.txt \
      torch==2.14.1+cu126 torchvision==0.29.1+cu126

# 3) onnxruntime-gpu — CUDA 12 판의 마지막 줄은 1.26.x (1.27 부터 PyPI 판은 CUDA 13).
#    1.24.1 cp314 휠은 PyPI 공식이며 호스트 venv 것과 sha256 이 같다.
RUN uv pip install --python /opt/venv/bin/python -c /opt/ds/constraints.txt onnxruntime-gpu==1.24.1

# 4) ultralytics 와 나머지. ultralytics 가 요구하는 opencv-python(GUI 판)은 빼고 headless 를 넣는다.
COPY overrides.txt /opt/ds/overrides.txt
RUN uv pip install --python /opt/venv/bin/python \
      -c /opt/ds/constraints.txt --override /opt/ds/overrides.txt \
      ultralytics==8.4.170 opencv-python-headless==5.0.0.93 numpy==2.5.2 pyyaml \
 && python -c "import importlib.metadata as m; assert not any(d.metadata['Name']=='opencv-python' for d in m.distributions()), 'opencv-python 가 들어옴'"

WORKDIR /ds
CMD ["python", "-c", "import torch, onnxruntime as ort; print(torch.__version__, torch.cuda.is_available(), ort.get_available_providers())"]
