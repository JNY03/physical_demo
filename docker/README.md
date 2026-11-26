# 도커로 띄우기 (처음 하는 사람용)

이 문서는 **순서대로 복사해 붙이면 되는 것**만 적는다. 왜 그런지는 [Docker 배포 계획](../Docker_배포_계획_261125.md) 에 있다.

띄우는 것은 셋이다.

| 프로필 | 들어가는 것 | 필요한 것 |
|---|---|---|
| `core` | 데이터 서버(파이 영상 · 상태 받기 · 저장 · Redis 스트림 · 스캔 창구) · 자원 기록 · 감독자 · **웹 런처** | 도커뿐 |
| `vision-gpu` | GPU 비전 (yoloe · river · targeted · depth) | NVIDIA GPU + **NVIDIA 컨테이너 툴킷** + 가중치 |
| `vision-cpu` | CPU 로 되는 비전 조각 (CLIP ONNX 단서 검증 · ONNX 깊이 추적) | 가중치 |

**GPU 가 없어도, 툴킷이 없어도 `core` 는 그대로 다 된다**: 영상 · 상태 받기, 저장, Redis 스트림, 스캔 창구,
자원 기록, 웹 런처. 안 되는 것은 GPU 비전 하나뿐이다 (→ [GPU 가 없을 때](#gpu-가-없거나-툴킷이-없을-때)).

---

## 0. 무엇이 필요한가

- **WSL(Ubuntu) 안의 docker-ce**. Docker Desktop(Windows)은 권하지 않는다 — host 네트워크가 제한적이라
  파이(테일넷)에서 영상을 못 당기고 대역폭 측정도 안 된다 (계획 §3).
  없으면 한 줄로 깐다(sudo 가 필요하다):

  ```bash
  sudo bash tools/docker/install_docker_wsl.sh
  ```

  깔고 나면 WSL 창을 닫고 다시 열어야 `sudo` 없이 `docker` 를 쓸 수 있다.
- **Redis** 가 `127.0.0.1:6380` 에 있어야 한다. 이 기기에는 이미 떠 있다(`mission-runtime-data-stream-redis`).
  확인: `docker ps | grep 6380`. 없으면 §1-3 에서 띄운다.
- **기기 주소**(파이 테일넷 IP)는 저장소에 없다. `config/hosts.json` 을 만들어 준다:

  ```bash
  cp config/hosts.example.json config/hosts.json
  # 편집기로 열어 주소를 채운다
  ```

- **가중치**(models/)는 이미지에 들어 있지 않다. 비전을 쓸 때만 필요하고, **호스트에서** 받는다:

  ```bash
  ./venv/bin/python tools/fetch_models.py --models yoloe
  ```

---

## 1. 띄우기

아래 명령은 모두 **저장소 뿌리**(`compose.yaml` 이 있는 곳)에서 돌린다.

### 1-1. 먼저 무엇이 되는지 본다 (아무것도 띄우지 않는다)

```bash
docker compose run --rm preflight
```

이 기기의 CPU · 메모리 · GPU · 디스크와 둘레(도커 · NVIDIA 툴킷 · k3s · Redis), 가중치를 읽어
**기능마다 되는지 · 안 되면 왜 · 무엇을 하면 풀리는지**를 한 장으로 찍는다. 읽기만 하고 아무것도 깔지 않는다.
(처음 빌드는 몇 분 걸린다. `core` 와 같은 이미지라 한 번만 기다리면 된다.)

### 1-2. core 띄우기

```bash
docker compose --profile core up -d
```

- 컨테이너가 뜨면 **첫 화면으로 위 preflight 한 장을 로그에 찍는다** — 그걸 보면 내 환경에서 무엇이 되는지 바로 안다.
- 웹 런처: 브라우저로 <http://127.0.0.1:10080/> (소스 · 모델 · 대상을 고르면 감독자가 맞춘다)
- 로그: `docker compose --profile core logs -f`
- 끄기: `docker compose --profile core down`

### 1-3. Redis 가 이 기기에 없을 때만

```bash
docker compose --profile redis up -d redis
```

이미 떠 있으면 **하지 않는다**. 같은 포트를 두 번 열 수 없고, 가시화와 파이가 보는 Redis 는 하나여야 한다.

### 1-4. GPU 비전까지 (GPU + 툴킷이 있을 때)

GPU 이미지는 두 번에 나눠 만든다. 첫 줄은 torch(약 3 GB)를 받아 **오래 걸린다(10~30분)**. 한 번만 하면 된다.

```bash
docker build -t ds-vision-gpu-base:dev --target ds-vision-gpu-base docker/vision-gpu
docker compose --profile core --profile vision-gpu up -d
```

### 1-5. CPU 비전 조각

```bash
docker compose --profile core --profile vision-cpu up -d
```

### 1-6. 파이(k3s)까지 쓸 때

`config/kubeconfig` 와 파이가 닿는 Redis 주소가 있어야 한다. 자세한 것은 [`compose.k3s.yaml`](../compose.k3s.yaml) 머리에 적어 뒀다.

```bash
export DS_REDIS_PUBLIC=redis://<이_기기_테일넷_주소>:6380/0
docker compose -f compose.yaml -f compose.k3s.yaml --profile core up -d
```

이것을 얹지 않으면 kubectl 이 없어 k3s 를 보지 않고, 지금처럼 이 기기 안에서만 돈다. **그래도 문제가 없다.**

---

## 2. 잘 떴는지 보기

```bash
docker compose --profile core ps        # STATUS 가 healthy 여야 한다
docker compose --profile core logs -f core
curl -s http://127.0.0.1:10080/health   # {"ok": true}
curl -s http://127.0.0.1:10011/health   # 소스(pi13)마다 포트가 다르다 — 10001~10012
```

상태 검사(healthcheck)가 보는 것:

| 서비스 | 보는 것 |
|---|---|
| `core` · `vision-gpu` · `vision-cpu` | `state/reported_<역할>.json` 의 시각이 15초 안에 새로 쓰였는지 (감독자가 살아 있다는 뜻) |
| `weblauncher` | `http://127.0.0.1:10080/health` |
| `redis` | `redis-cli -p 6380 ping` |

---

## 3. 안 될 때 무엇을 보는가

| 보이는 것 | 뜻 | 하는 일 |
|---|---|---|
| `could not select device driver ... gpu` | NVIDIA 컨테이너 툴킷이 없다 | `vision-gpu` 를 빼고 `core` 만 띄운다. 툴킷은 `sudo bash tools/docker/install_docker_wsl.sh` 가 깐다 |
| `pull access denied ... ds-vision-gpu-base` | GPU 베이스 이미지를 아직 안 만들었다 | §1-4 의 첫 줄을 먼저 돌린다 |
| `pi13 주소가 없습니다` | `config/hosts.json` 이 없거나 비어 있다 | §0 의 `cp config/hosts.example.json config/hosts.json` |
| 첫 화면에 `GPU: 없음` · `docker 없음` | ds-core 컨테이너에는 `nvidia-smi` 도 `docker` 명령도 **일부러** 없다(docker 소켓을 주지 않는다 — 계획 §5). 호스트에 GPU 가 있어도 이렇게 보인다 | GPU 가 보이는지는 `vision-gpu` 컨테이너의 첫 화면에서 본다. 호스트 쪽 판단은 `./venv/bin/python -m ds_resources.cli preflight` |
| 첫 화면의 「더 깔면 되는 것」에 가중치가 안 보인다 | 그 줄은 호스트 venv 가 있을 때만 나온다(받는 명령이 호스트 것이라서) | 바로 아래 「컨테이너에서 더 읽을 것」이 받는 길을 적어 준다 |
| 로그에 `쓸 수 있는 모델이 없음` | 가중치가 없다 | **호스트에서** `./venv/bin/python tools/fetch_models.py --models <선택지>`. `models/` 는 컨테이너에 읽기 전용으로 들어가므로 컨테이너 안에서는 받을 수 없다 |
| `address already in use` | 그 포트를 이미 쓰는 것이 있다 (옛 WSL 창 방식 · 다른 Redis) | `docker compose --profile core down` 하고 옛 창(`tools/ds_stop.sh`)을 먼저 끈다 |
| `다른 오케스트레이터가 이미 돌고 있습니다` | 호스트 venv 쪽 비전이 돌고 있다 | 둘 중 하나만 둔다. `state/` 를 바인드 마운트한 덕에 **같은 GPU 를 두 번 띄우지는 않는다**(계획 §6) |
| STATUS 가 `unhealthy` | 감독자가 보고를 못 쓰고 있다 | `docker compose logs core` 와 `cat state/reported_core.json` |
| 로그 시각이 이상하다 | 시간대 | `.env` 에 `TZ=Asia/Seoul` (기본값이다) |

전부 지우고 처음부터: `docker compose --profile core --profile vision-gpu down` → `docker image rm ds-core:dev` → §1 부터.

---

## 4. GPU 가 없거나 툴킷이 없을 때

| 되는 것 | 안 되는 것 |
|---|---|
| 파이 · 로봇 · 드론 영상 · 상태 받기, 저장, Redis 스트림, 스캔 창구 | GPU 비전 (yoloe · river · targeted · depth) |
| 360 슬라이스, 자원 기록 · 기억, 노드 스트림 | |
| 웹 런처, 원하는 상태 · 감독자 | |
| 가시화가 보는 HTTP · Redis 전부 (포트 · 형식이 그대로다) | |
| CPU 로 되는 비전 조각 (`vision-cpu` 프로필) | |

> **이 기기에서 확인하지 못한 것:** 이 기기에는 NVIDIA 컨테이너 툴킷이 없다. 그래서 `vision-gpu` 는
> **compose 설정 검사까지만** 확인했다. **GPU 이미지 빌드와 컨테이너 안에서 GPU 가 보이는지는 확인하지 못했다.**
> 툴킷이 있는 기기에서 `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi` 가 먼저 되는지 보고,
> 그 뒤 계획 D5(컨테이너에서 `gpu_context_mb` 다시 재기)를 해야 한다.

---

## 5. 무엇이 어디에 놓이는가

| 호스트 | 컨테이너 | 쓰기 | 왜 |
|---|---|---|---|
| `state/` | `/ds/state` | 읽기+쓰기 | 기억 · 실행 상태 · 감독자 보고. **이름 볼륨이 아니라 바인드 마운트** — 파일 잠금(flock)은 같은 파일에서만 통해서, 옛 창 방식과 컨테이너가 서로의 잠금을 봐야 같은 GPU 를 두 번 띄우지 않는다 |
| `stream_data/` | `/ds/stream_data` | 읽기+쓰기 | 받은 영상 · 상태 (호스트에서 바로 열어 본다) |
| `models/` | `/ds/models` | **읽기 전용** | 가중치. 이미지에 넣지 않는다 |
| `external/` | `/ds/external` | **읽기 전용** | 외부 코드 (UniDepth · AerialMetric) |
| `config/` | `/ds/config` | **읽기 전용** | `hosts.json` · `vision_config.local.yaml` · `kubeconfig`(선택) |

환경변수는 `compose.yaml` 머리의 `x-ds-env` 한 곳에 모여 있다. 바꾸고 싶은 것은 저장소 뿌리의 `.env` 에 적는다
(파일이 없어도 된다):

```
DS_NODE_NAME=laptop-abc            # 노드 이름. host 네트워크면 비워 둬도 호스트 이름이 된다
DS_REDIS=redis://127.0.0.1:6380/0
DS_WEB_HOST=127.0.0.1              # 0.0.0.0 은 인증 없이 망 전체에 웹 런처를 연다 — 믿는 망에서만
DS_WEB_ALLOWED_HOSTS=ds-pc         # 127.0.0.1 밖의 이름으로 들어올 때 받아 줄 이름
PD_HOST_PI13=100.x.y.z             # config/hosts.json 대신 환경변수로 줄 때
TZ=Asia/Seoul
```

---

## 6. 이미지와 Dockerfile

| 이미지 | 아키텍처 | 만드는 명령 | 안에 있는 것 |
|---|---|---|---|
| `ds-core:dev` | amd64 · arm64 | `docker build -t ds-core:dev --target ds-core -f docker/core/Dockerfile .` | 데이터 서버 · 스캔 창구 · 자원 기록 · 감독자 · 웹 런처. **torch 없음**(0.27 GB — 실측) |
| `ds-vision-cpu:dev` | amd64 · arm64 | `docker build -t ds-vision-cpu:dev --target ds-vision-cpu -f docker/core/Dockerfile .` | 위 + onnxruntime(CPU) (0.33 GB — 실측) |
| `ds-core:dev-k3s` | amd64 · arm64 | `docker build -t ds-core:dev-k3s --target ds-core-k3s -f docker/core/Dockerfile .` | 위 + `orchestrator/` + kubectl v1.36.4 (0.29 GB — 실측) |
| `ds-vision-gpu-base:dev` | amd64 | `docker build -t ds-vision-gpu-base:dev --target ds-vision-gpu-base docker/vision-gpu` | CUDA 12.6 · torch cu126 · onnxruntime-gpu · ultralytics (크고 오래 걸린다, 거의 안 바뀐다) |
| `ds-vision-gpu:dev` | amd64 | `docker build -t ds-vision-gpu:dev --target ds-vision-gpu -f docker/vision-gpu/Dockerfile.app .` | 베이스 + 이 저장소 코드 · 남은 의존성 |

- 비전 GPU 판의 Dockerfile 이 둘인 까닭: 베이스는 torch 를 받아 오래 걸리고 거의 안 바뀐다. 코드는 자주 바뀐다.
  한 파일에 두면 코드를 한 줄 고칠 때마다 큰 레이어의 캐시가 흔들린다.
- `Dockerfile.dockerignore` 는 **보낼 것**을 고른다. 없으면 빌드가 시작하기도 전에 `.git`(253 MB) ·
  `models/`(991 MB) · `stream_data/`(1.7 GB) 를 복사한다.
- arm64(파이)는 같은 명령에 `docker buildx build --platform linux/arm64` 를 쓴다.
  **이 기기에서는 amd64 만 빌드해 확인했다** — arm64 는 확인하지 못했다(휠이 있는지부터 봐야 한다).

### 6-1. 기기 단계 실행체 `ds-device-runner` (파이에서 도는 작은 모델)

기능의 **기기 단계**(object_detect · zeroshot_detect · segmentation · depth_estimate 의 `device`)를 파이 CPU 로
돌리는 이미지다. 실행체는 [`vision/device_stage.py`](../vision/device_stage.py) 하나이고, 어떤 모델을 돌릴지는
k3s 가 파드를 띄울 때 **역할 슬롯에 고른 모델**로 정한다(`--model {slot.detector}` — `orchestrator/k3s_executor.py`).
torch · ultralytics 가 없다(numpy · opencv-headless · onnxruntime CPU · ncnn · redis · PyYAML).

```bash
docker start ds-registry                              # 이 기기의 사설 레지스트리 (:5000)
bash docker/device-runner/build.sh                    # amd64 만, 로컬 도커에 — 시험
bash docker/device-runner/build.sh --push             # arm64 + amd64 → 127.0.0.1:5000/ds-device-runner:<git 해시>
```

- **모델 더하기**: 실행기 파일 `vision/runners/<이름>.py` 는 언제나 다 들어간다. 가중치만
  [`docker/device-runner/models.txt`](device-runner/models.txt) 에 한 줄(레지스트리 키 또는 `models/…` 경로) 더한다.
  레지스트리 `models.<키>` 에 `runs_on: [cpu_arm64]` · `size: small` 이 있으면 기기 슬롯 선택지에 뜬다.
  커밋 → `build.sh --push` → 레지스트리 `functions.*.stages.device.image` 의 태그를 새 해시로.
- 결과는 Redis `device_stage:<소스>:<기능>`(검출 상위 k 개 · 처리 ms · CPU% — 프레임은 내지 않는다),
  하트비트는 `cpu_telemetry:<UNIT_ID>`. 입력은 그 카메라의 파이에 놓이면 `http://<노드>:8090/stream/1`(MJPEG),
  아니면 서버 Redis `camera_stream:<소스>`.

---

## 7. 가시화는 바뀌지 않는다

가시화는 소스 포트의 HTTP(`/frame` · `/stream` · `/state` · `/vision` · `/health` · 스캔 창구)와
Redis `:6380` 으로만 붙는다. 컨테이너는 **host 네트워크**라 주소도 포트도 지금과 같고, 스트림 이름 · 칸 · 형식도
그대로다(칸 추가만 있다). 그래서 가시화 쪽은 아무것도 고치지 않는다 (계획 §11).
바뀌지 않았는지 확인하는 검사가 따로 있다: `./venv/bin/python tools/contract_check.py`.
