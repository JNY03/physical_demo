# syntax=docker/dockerfile:1
#
# ds-core — 데이터 서버 · 스캔 창구 · 자원 기록/기억 · 감독자 · 웹 런처 (Docker 계획 §3 · D4)
#
# 빌드 (컨텍스트는 저장소 루트 — 코드를 그대로 넣는다. 보낼 것은 Dockerfile.dockerignore 가 고른다):
#   docker build -t ds-core:dev --target ds-core -f docker/core/Dockerfile .
# 곁딸린 판 (같은 파일의 다른 단계 — 바탕이 같아 레이어를 나눠 쓴다):
#   --target ds-vision-cpu  ds-core + onnxruntime(CPU) — GPU 없이 되는 비전 조각
#   --target ds-core-k3s    ds-core + orchestrator/ + kubectl — compose.k3s.yaml 이 쓴다
# ※ 기본 단계(--target 없이)는 마지막 단계다. 늘 --target 을 적으세요. compose 는 적어 준다.
#
# 원칙 (Docker_배포_계획 §2 · §3 · §6)
# - **torch 가 없다.** core 는 requirements/core.txt 만 깐다(약 0.4 GB). 데이터만 쓰려는 사람이 8 GB 를 받지 않게.
# - **가중치를 넣지 않는다.** models/ · external/ 은 실행 때 /ds/models · /ds/external 로 읽기 전용 마운트.
# - **기기 주소 · 계정도 넣지 않는다.** config/ 를 마운트하거나 DS_* · PD_HOST_* 환경변수로 준다.
# - 멀티 아키텍처(amd64 · arm64): 아키텍처를 박은 것이 없다. Python 은 uv 가 그 아키텍처 판을 받고,
#   휠도 아키텍처별 판을 받는다. 그래서 `docker buildx build --platform linux/amd64,linux/arm64` 로 둘을 낸다.
# - 레이어 순서는 잘 안 바뀌는 것 → 자주 바뀌는 것: 시스템 패키지 → Python → 의존성 → 코드.
#   코드를 고쳐도 다시 받는 것이 없다.

ARG BASE=ubuntu:24.04

FROM ghcr.io/astral-sh/uv:0.13.7 AS uv

# ---------------------------------------------------------------- ds-core
FROM ${BASE} AS ds-core

ARG PYTHON_VERSION=3.14.4
ENV DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/ds \
    DS_HOME=/ds \
    DS_PYTHON=/opt/venv/bin/python \
    TZ=Asia/Seoul

# 시스템 패키지: 최소.
#   libglib2.0-0t64  opencv-python-headless 의 cv2 가 찾는다 (headless 라 libGL 은 필요 없다)
#   tzdata           로그 시각(time.strftime)이 한국 시간으로 보이게. 없으면 UTC 로 찍혀 사람이 헷갈린다
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libglib2.0-0t64 tzdata \
 && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

# Python — 호스트 venv 와 같은 3.14.4 (python-build-standalone, uv 가 받는다). 아키텍처는 uv 가 고른다
RUN uv python install ${PYTHON_VERSION} \
 && uv venv --python ${PYTHON_VERSION} /opt/venv

# 의존성 — 코드보다 먼저. core 층만(torch · ultralytics · onnxruntime 없음)
COPY requirements/core.txt /opt/ds/requirements/core.txt
RUN uv pip install --python /opt/venv/bin/python -r /opt/ds/requirements/core.txt \
 && python -c "import cv2, av, redis, yaml, numpy, paho.mqtt.client; print('core 의존성 ok', cv2.__version__)"

# 저장소 코드의 .pyc 를 만들지 않는다 — /ds 는 root 소유이고 컨테이너는 호스트 사용자로 돈다(compose 의 user:).
# 쓰기가 막힌 폴더에 .pyc 를 쓰려다 마는 일을 아예 없앤다. 패키지 쪽은 이미 빌드할 때 컴파일해 뒀다(UV_COMPILE_BYTECODE)
ENV PYTHONDONTWRITEBYTECODE=1

# 코드. 가중치(models/ · external/)와 수집 데이터(stream_data/)는 들어가지 않는다 — 마운트로 받는다
WORKDIR /ds
COPY requirements/ /ds/requirements/
COPY config/hosts.example.json /ds/config.example/hosts.example.json
COPY server_stream_multi_source.py scan_desk.py conftest.py pytest.ini /ds/
COPY ds_resources/ /ds/ds_resources/
COPY vision/ /ds/vision/
COPY e2p/ /ds/e2p/
COPY tools/ /ds/tools/
COPY docker/core/entrypoint.sh /usr/local/bin/ds-entrypoint

# 마운트가 없어도 뜨도록 자리만 만든다 (없으면 상태 파일을 못 쓰고 바로 죽는다).
# 777 을 주는 셋은 **실행 중에 쓰는 곳**이다. 컨테이너는 호스트 사용자(uid 1000 등)로 도는데(compose 의 user:)
# 이미지 안의 폴더는 root 소유다 — 마운트가 없을 때도, 이름 볼륨(hf_cache)이 처음 만들어질 때도 쓸 수 있어야 한다.
#   e2p: slice360_live.py 가 지금 각도를 e2p/slice_angle.json 에 늘 고쳐 쓴다 (--angle-file 기본값)
RUN mkdir -p /ds/state /ds/stream_data /ds/config /ds/models /ds/external /ds/hf_cache \
 && chmod 0777 /ds/state /ds/stream_data /ds/hf_cache /ds/e2p \
 && chmod +x /usr/local/bin/ds-entrypoint

# 첫 실행에 preflight 한 장을 찍고(DS_PREFLIGHT=0 이면 건너뛴다) 받은 명령을 exec 한다 — 신호가 그대로 전달된다
ENTRYPOINT ["/usr/local/bin/ds-entrypoint"]
CMD ["python", "-m", "ds_resources.supervisor", "--role", "core"]

# ---------------------------------------------------------------- ds-vision-cpu
# core + onnxruntime(CPU). GPU · torch 없이 되는 비전 조각(CLIP ONNX 단서 검증 · ONNX 깊이 추적)만 돈다.
# ultralytics(YOLOE)를 쓰는 선택지는 torch 가 필요하다 → ds-vision-gpu (docker/vision-gpu/).
FROM ds-core AS ds-vision-cpu
COPY requirements/vision-cpu.txt /opt/ds/requirements/vision-cpu.txt
RUN uv pip install --python /opt/venv/bin/python -r /opt/ds/requirements/vision-cpu.txt \
 && python -c "import onnxruntime as ort; print('onnxruntime', ort.__version__, ort.get_available_providers())"
CMD ["python", "-m", "ds_resources.supervisor", "--role", "vision"]

# ---------------------------------------------------------------- ds-core-k3s
# core + orchestrator/ + kubectl. 계획 §3: ds-orchestrator 이미지를 두지 않는다 —
# k3s 플러그인(orchestrator.k3s_source)은 기록기와 오케스트레이터 cli **안에서** 경로 문자열로 불린다.
# 그래서 이미지를 더 만들지 않고 이 단계에서 파일과 kubectl 만 얹는다. core 가 orchestrator 를 import 하지는 않는다.
FROM ds-core AS ds-core-k3s
ARG TARGETARCH=amd64
# 이 현장의 k3s 서버와 같은 판(`k3s --version` = v1.36.4+k3s1). kubectl 은 ±1 마이너까지 맞지만 같은 판이 가장 안전하다
ARG KUBECTL_VERSION=v1.36.4
# kubectl 을 **파이썬으로** 받는다. curl 을 쓰려고 apt 를 다시 켜면 우분투 미러가 흔들릴 때
# (index 크기 불일치 — 실제로 한 번 겪었다) 이 단계가 통째로 깨진다. 파이썬은 이미 있다.
# 받은 것은 공식 .sha256 과 맞춰 본다 — 중간에 바뀐 파일을 kubectl 로 두지 않는다.
RUN python <<'PY'
import hashlib, os, urllib.request
ver, arch = os.environ["KUBECTL_VERSION"], os.environ["TARGETARCH"]
url = "https://dl.k8s.io/release/%s/bin/linux/%s/kubectl" % (ver, arch)
want = urllib.request.urlopen(url + ".sha256", timeout=60).read().decode().split()[0]
data = urllib.request.urlopen(url, timeout=600).read()
got = hashlib.sha256(data).hexdigest()
assert got == want, "kubectl sha256 가 다릅니다: %s != %s" % (got, want)
with open("/usr/local/bin/kubectl", "wb") as f:
    f.write(data)
os.chmod("/usr/local/bin/kubectl", 0o755)
print("kubectl", ver, arch, "ok", got[:12])
PY
COPY orchestrator/ /ds/orchestrator/
