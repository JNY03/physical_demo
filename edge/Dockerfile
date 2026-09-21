# 엣지 노드 — **이쪽은 Docker가 정본이다.**
#
# 이유는 의존성이다. torch + CUDA + ultralytics + transformers + peft + MoGe
# 포크가 얽혀 있고, 이 기기의 호스트 Python은 3.14라 torch가 아직 안 올라간다.
# venv로 맞추면 기기를 옮길 때마다 그 조합을 다시 맞춰야 한다 — 컨테이너가 그
# 조합을 통째로 옮긴다(말단은 반대다: 카메라 때문에 venv가 정본이다).
#
#   docker compose up --build
#   docker compose run --rm edge ./fetch_models.sh     # 가중치 먼저
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# python3.10 — AerialMetric이 권장하는 버전이다. MoGe 포크와 peft가 그 위에서
# 검증됐고, 여기서 올리면 LoRA 로딩부터 다시 확인해야 한다.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip \
        git curl ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python

WORKDIR /app

# torch를 먼저, 따로 깐다. requirements.txt에 박지 않는 이유는 CUDA 버전에
# 묶여 있어서다 — 기기를 옮기면 이 한 줄만 바꾼다.
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py config.json fetch_models.sh ./
RUN chmod +x fetch_models.sh

# 가중치(1.5GB+)와 서드파티 코드, 산출물은 이미지에 굽지 않는다. 구우면 코드 한
# 줄 고칠 때마다 같이 옮겨지고, 이미지가 이전(移轉) 비용이 된다.
VOLUME ["/app/models", "/app/third_party", "/app/store"]

EXPOSE 8891
CMD ["python", "server.py"]
