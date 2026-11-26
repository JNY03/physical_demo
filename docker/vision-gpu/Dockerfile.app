# syntax=docker/dockerfile:1
#
# ds-vision-gpu — 베이스(ds-vision-gpu-base: CUDA 12.6 · torch cu126 · onnxruntime-gpu · ultralytics) 위에
# 이 저장소의 코드와 남은 파이썬 의존성을 얹는 얇은 층 (Docker 계획 §3 · D5).
#
# **파일을 둘로 나눈 까닭:** 베이스는 torch 를 받아 오래 걸리고 거의 바뀌지 않는다(컨텍스트도 docker/vision-gpu 하나면 된다).
# 코드는 자주 바뀐다. 한 파일에 두면 코드를 한 줄 고칠 때마다 큰 레이어의 캐시 열쇠가 흔들린다.
#
# 순서 (1번은 길다 — 한 번만 하면 된다):
#   1) 베이스:   docker build -t ds-vision-gpu-base:dev --target ds-vision-gpu-base docker/vision-gpu
#   2) 코드 층:  docker build -t ds-vision-gpu:dev --target ds-vision-gpu -f docker/vision-gpu/Dockerfile.app .
#      (--target ds-vision-gpu-k3s 로 orchestrator/ + kubectl 을 더 얹는다 — compose.k3s.yaml 이 쓴다)
#   compose 는 두 번째만 한다. 베이스가 없으면 거기서 멈추고 1번을 하라고 말한다.
#
# 넣지 않는 것: 가중치(models/) · 외부 코드(external/) · 기기 주소. 실행 때 마운트한다(계획 §6).

ARG BASE=ds-vision-gpu-base:dev

FROM ${BASE} AS ds-vision-gpu

ENV PYTHONPATH=/ds \
    PYTHONDONTWRITEBYTECODE=1 \
    DS_HOME=/ds \
    DS_PYTHON=/opt/venv/bin/python \
    TZ=Asia/Seoul \
    HF_HOME=/ds/hf_cache \
    HF_HUB_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# git: requirements/vision-gpu.txt 의 세 항목(clip · pipeline · utils3d)이 git+https 다.
# tzdata: 로그 시각을 한국 시간으로 (없으면 UTC 로 찍힌다)
RUN apt-get update \
 && apt-get install -y --no-install-recommends git tzdata \
 && rm -rf /var/lib/apt/lists/*

# 남은 의존성. 베이스에 이미 있는 torch · onnxruntime-gpu · ultralytics · numpy · opencv 는 판이 같아 다시 받지 않는다.
# (+cu126 꼬리표가 붙은 판을 찾으려면 베이스와 같은 색인 설정이 필요하다)
COPY requirements/ /opt/ds/requirements/
COPY docker/vision-gpu/overrides.txt /opt/ds/overrides.txt
RUN uv pip install --python /opt/venv/bin/python \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --override /opt/ds/overrides.txt \
      -r /opt/ds/requirements/vision-gpu.txt \
 && python -c "import importlib.metadata as m; assert not any(d.metadata['Name']=='opencv-python' for d in m.distributions()), 'opencv-python 가 들어옴'"

# 코드
WORKDIR /ds
COPY requirements/ /ds/requirements/
COPY config/hosts.example.json /ds/config.example/hosts.example.json
COPY server_stream_multi_source.py scan_desk.py conftest.py pytest.ini /ds/
COPY ds_resources/ /ds/ds_resources/
COPY vision/ /ds/vision/
COPY e2p/ /ds/e2p/
COPY tools/ /ds/tools/
COPY docker/core/entrypoint.sh /usr/local/bin/ds-entrypoint

# 777 셋은 실행 중에 쓰는 곳 — 컨테이너는 호스트 사용자로 돈다(compose 의 user:). ds-core 와 같은 까닭이다
RUN mkdir -p /ds/state /ds/stream_data /ds/config /ds/models /ds/external /ds/hf_cache \
 && chmod 0777 /ds/state /ds/stream_data /ds/hf_cache /ds/e2p \
 && chmod +x /usr/local/bin/ds-entrypoint

# ds-core 와 같은 첫 화면(preflight 한 장) → 그 뒤 받은 명령을 exec
ENTRYPOINT ["/usr/local/bin/ds-entrypoint"]
CMD ["python", "-m", "ds_resources.supervisor", "--role", "vision"]

# ---------------------------------------------------------------- ds-vision-gpu-k3s
# 위 + orchestrator/ + kubectl (계획 §3 — ds-orchestrator 이미지를 두지 않고 태그 변형으로 둔다)
FROM ds-vision-gpu AS ds-vision-gpu-k3s
ARG TARGETARCH=amd64
ARG KUBECTL_VERSION=v1.36.4
# ds-core 와 같은 방식: curl 을 쓰려고 apt 를 켜지 않는다 (미러가 흔들리면 이 단계가 깨진다). 공식 .sha256 으로 맞춰 본다
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
