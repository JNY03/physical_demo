# 엣지 — 린 프로필 (모델도 GPU 도 없이 돈다)
# ===========================================
# 이것이 프레임워크의 증거다. 받는 사람이 가중치 1.6GB 를 내려받고 CUDA 를 맞추기
# 전에, `docker build` 한 번으로 전체 파이프라인(말단에서 당기기 → 검출 → 판정
# 되밀기 → 스트리밍)이 도는 것을 먼저 본다.
#
# 무엇이 빠지나: torch·ultralytics·CUDA. 그래서 YOLOE·YOLO-World·MoGe2-Aerial 이
# 없다. 검출은 색 분할 provider 가 맡고(1280x720 에서 CPU 7.7ms 실측), 거리는
# 나오지 않는다.
#
# 무엇이 그대로인가: 말단 HTTP 2계층, 합의·추적, 스트리밍, 통합 명령 버스
# 어댑터, 저장소 회전. 규약이 전부 같으므로 나중에 모델을 붙여도 위층은
# 그대로다.
#
#   docker build -f Dockerfile.lean -t drone-edge:lean .
#   docker run --rm -p 8891:8891 \
#     -v "$PWD/config.json:/app/config.json:ro" drone-edge:lean
#
# 모델이 필요해지면 Dockerfile(CUDA) 쪽을 쓰고 config 의 provider 를 바꾼다.
FROM python:3.12-slim-bookworm

# opencv 가 런타임에 요구하는 것만. 빌드 도구는 넣지 않는다 — headless 휠에
# 컴파일이 필요 없다.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# **torch 도 ultralytics 도 넣지 않는다.** models.py 최상위 import 는
# cv2·numpy·onnxruntime 뿐이고 둘은 전부 지연 import 라, 색 provider 구성에서는
# 불리지 않는다. run.sh 가 설정을 읽어 요구 사항을 계산하므로 기동도 막지 않는다.
RUN pip install --no-cache-dir \
        opencv-python-headless numpy onnxruntime requests

WORKDIR /app
COPY . /app/

# 설정은 바깥에서 준다 — 말단 주소가 거기 들어간다. 이미지를 다시 굽지 않고
# 기기를 바꿀 수 있어야 한다.
VOLUME ["/app/store"]
EXPOSE 8891

RUN useradd -r -u 10001 edge && chown -R edge /app
USER edge

ENV PYTHONUNBUFFERED=1
CMD ["python3", "-u", "server.py"]
