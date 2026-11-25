# ds-vision-gpu-base 실측 (D5 첫 단계, 2026-11-25)

기기: WSL2 Ubuntu 26.04 · docker-ce 29.10.1 + NVIDIA Container Toolkit · RTX 4050 Laptop 6 GB (드라이버 615.78).
측정 전 GPU 사용량 0 MiB(다른 프로세스 없음) 확인. 스크립트: [`measure_gpu.py`](measure_gpu.py) — 호스트 venv 와 컨테이너에서 같은 파일.

```bash
# 빌드 · 올리기
docker build -t 127.0.0.1:5000/ds-vision-gpu-base:dev --target ds-vision-gpu-base docker/vision-gpu
docker push 127.0.0.1:5000/ds-vision-gpu-base:dev
# 컨테이너 실측 (가중치는 읽기 전용 마운트)
docker run --rm --gpus all -v $PWD/models:/ds/models:ro \
  -v $PWD/docker/vision-gpu/measure_gpu.py:/opt/ds/measure_gpu.py:ro \
  127.0.0.1:5000/ds-vision-gpu-base:dev python /opt/ds/measure_gpu.py
# 호스트 실측
venv/bin/python docker/vision-gpu/measure_gpu.py --models-dir models
```

## 1. 이미지

| 항목 | 값 |
|---|---|
| 베이스 | `nvidia/cuda:12.6.3-base-ubuntu24.04` (CUDA 라이브러리는 torch 의 nvidia-*-cu12 휠이 가져옴 → `runtime` 변형은 중복) |
| Python | 3.14.4 (python-build-standalone, uv 0.13.7) — 호스트 venv 와 같은 버전 |
| 패키지 | torch 2.14.1+cu126 · torchvision 0.29.1+cu126 · onnxruntime-gpu 1.24.1 · ultralytics 8.4.170 · opencv-python-headless 5.0.0.93 · numpy 2.5.2 (나머지는 `constraints.txt` = 호스트 pip freeze) |
| 크기 | 디스크 **12.5 GB**, 압축(레지스트리 전송) **4.32 GB** |
| 층별 | torch 층 6.87 GB (nvidia 휠 3.6 G · torch 1.7 G · triton 0.9 G) · onnxruntime 428 MB · ultralytics 외 455 MB · Python 134 MB |
| 빌드 시간 (캐시 없음) | **781 s** — torch 층 395 s(받기), 이미지 내보내기 293 s |
| 올리기 | `127.0.0.1:5000/ds-vision-gpu-base:dev` 21 s, digest `sha256:37e4337e61bc…` |

### onnxruntime-gpu 와 Python 3.14

- **PyPI 에 공식 cp314 CUDA 12 휠이 있다.** 호스트 venv 의 「로컬 휠」 1.24.1 은 PyPI 의
  `onnxruntime_gpu-1.24.1-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` 과 sha256 이 같다(`ced66900…`).
  그래서 3.12 로 내릴 필요가 없고 버전 고정만 하면 된다.
- PyPI 의 onnxruntime-gpu 는 **1.26.x 까지 CUDA 12**, **1.27.0 부터 CUDA 13**(nvidia-*-cu13 의존). 올릴 때 주의:
  1.27+ 로 가면 torch 도 cu13x 판과 맞추거나 CUDA 13 라이브러리를 따로 넣어야 한다(드라이버 615 는 CUDA 13 가능).
  CUDA 12 를 유지하면서 올릴 수 있는 상한은 1.26.0.
- torch 를 import 하지 않아도 `onnxruntime.preload_dlls()` 가 venv 의 nvidia 휠 라이브러리를 찾아 CUDA 제공자가 뜬다.
- 무해한 경고: `GPU device discovery failed … /sys/class/drm/card0/device/vendor` (WSL 에는 drm 이 없음). 호스트와 같다.

## 2. 동작 확인 (컨테이너, `--gpus all`)

- `torch.cuda.is_available()` True · `NVIDIA GeForce RTX 4050 Laptop GPU` · 1024² matmul 정상.
- `ort.get_available_providers()` = `[TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider]`.
- CLIP `clip-vit-base-patch32/onnx/vision_model.onnx`(읽기 전용 마운트) 가 CUDAExecutionProvider 로 실행, 중앙값 6.2–9.6 ms.

## 3. 호스트 venv 대 컨테이너

세 번씩 돌림(호스트 첫 회는 시작 때 기기 사용량이 143 MiB 라 메모리 증가분에서 뺐다). 메모리 값은 세 번 모두 MB 단위까지 같았다.

| 항목 | 호스트 venv | 컨테이너 | 차이 |
|---|---|---|---|
| init + 1 MiB 할당: 기기 증가 / torch 예약 / 컨텍스트 | 83 / 2 / **81** MB | 83 / 2 / **81** MB | 0 |
| + 512² matmul(cuBLAS): 기기 증가 / torch 예약 / 컨텍스트 | 127 / 24 / **103** MB | 127 / 24 / **103** MB | 0 |
| YOLOE-11l-seg-pf 640²: torch 예약 최대 | 478 MB | 478 MB | 0 |
| YOLOE: 기기 증가 | 607 MB | 607 MB | 0 |
| YOLOE: 컨텍스트 몫 (기기 증가 − 예약 최대) | **129 MB** | **129 MB** | 0 |
| YOLOE: 지연 중앙값 (예열 뒤 20회) | 40.2 · 39.8 (첫 회 46.6) ms | 39.6 · 41.2 · 41.4 ms | 중앙값끼리 +2.4% |
| YOLOE: p90 | 42.8 – 51.1 ms | 46.1 – 57.9 ms | 노트북 GPU 흔들림 수준 |
| YOLOE: 적재 + 첫 추론 | 3.6 – 4.1 s | 3.5 – 4.7 s | — |
| ONNX CLIP vision: 기기 증가 | 615 – 623 MB | 615 – 619 MB | 0 |

**결론 — `gpu_context_mb: 150` 은 그대로 둔다.**
- 컨테이너의 컨텍스트 몫은 호스트와 똑같다(129 MB). 컨테이너화가 컨텍스트 몫을 바꾸지 않는다.
- 레지스트리 근거값 135(2026-11-24, 469 − 334)와 비교하면 −4%(±10% 안). 150 은 21 MB 여유가 있는 위쪽 값으로 계속 맞다.
- 지연 · 메모리 모두 D5 통과 조건(±10%) 안.

**주의 — onnxruntime 세션은 torch 셈법에 안 잡힌다.** CLIP vision ONNX 한 세션이 기기를 ~615 MB 늘리지만
torch 예약량은 0 이다(ORT 자체 아레나 + cuDNN/cuBLAS 작업 공간). ONNX 모델의 GPU 몫은 `gpu_context_mb` 가 아니라
모델별 실측값으로 따로 잡아야 한다.

## 4. MPS (시험만)

**안 된다.** WSL 드라이버에는 `nvidia-cuda-mps-control` 이 없다. `nvidia-compute-utils-615` 패키지에서 바이너리만 꺼내 시험:
- 호스트: `nvidia-cuda-mps-control -d` → `Failed to initialize MPS control daemon`.
- 컨테이너: `-d` 는 0 을 돌려주지만 데몬이 곧 죽음 — control.log `Failed to initialize CUDA: 801`(CUDA_ERROR_NOT_SUPPORTED).
  클라이언트는 MPS 없이 그냥 돌고, `set_default_device_pinned_mem_limit` 도 적용되지 않음(1.5 GB 할당 성공).
- 따라서 WSL 에서 GPU 메모리 강제는 MPS 로 할 수 없다. 지금처럼 레지스트리 실측값 + 배치기로 막는다.

## 5. 남은 일

- 이 베이스 위에 `ds-vision-gpu`(vision/ · ds_resources 코드, lap · clip 등 추가 의존성) 단계를 얹는다 — requirements/ 정리 후.
- `YOLO_CONFIG_DIR=/tmp/Ultralytics` 가 없는 디렉터리라 첫 실행 때 경고 한 줄(동작은 정상). 다음 빌드에서 디렉터리를 만들어 둔다.
- targeted(UniDepth 등) 실측은 ds-vision-gpu 단계에서.
