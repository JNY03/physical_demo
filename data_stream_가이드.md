# data_stream 사용 가이드

로봇 2대·드론·360 카메라의 영상과 상태를 받아 **Redis 로 흘리고 디스크에 저장**하는 프로그램과, 그 프레임에 비전 모델을 돌리는 프로그램의 사용법이다.

---

## 1. 경로와 접속

### 1-1. 폴더 위치 (WSL Ubuntu 안)

```
~/data_stream/                              ← 저장소 루트 (명령은 여기서 실행) = /home/khw67/data_stream
├── venv/                                   ← 파이썬 가상환경 (redis · paho-mqtt · torch · ultralytics 설치됨)
├── server_stream_multi_source.py           ← 데이터 스트림 (받기 · Redis · 저장)
├── hosts.local.json                        ← 로봇·드론 주소 (tailscale IP)
├── vision/
│   ├── vision_infer.py                     ← 비전 추론 (YOLOE · UniDepth · MoGe2)
│   └── vision_config.yaml                  ← 비전 설정
├── 360camera_work/
│   └── stream_to_server.py                 ← 360 카메라 → 서버 업로드
├── models/  external/                      ← 모델 가중치 · UniDepth · AerialMetric 코드
└── stream_data/                            ← 저장된 데이터 (아래 3장)
```

Windows 탐색기에서는 주소창에 `\\wsl$\Ubuntu\home\khw67\data_stream` 을 입력하면 열린다.

### 1-2. VS Code 로 WSL 접속

1. Windows 에 VS Code 를 설치하고 (설치·초기 설정 전체는 `배포_가이드.md`), 확장(Extensions, `Ctrl+Shift+X`)에서 **WSL** (Microsoft) 을 설치한다.
2. `F1` → **`WSL: Connect to WSL`** (또는 왼쪽 아래 `><` 초록 버튼 → *Connect to WSL*).
   배포판이 여러 개면 **`WSL: Connect to WSL using Distro...`** → `Ubuntu` 를 고른다.
3. 새 창의 왼쪽 아래에 `WSL: Ubuntu` 가 보이면 접속된 것이다.
4. **File → Open Folder** → `~/data_stream` (`/home/khw67/data_stream`) 를 연다.
5. 터미널은 `` Ctrl+` `` 로 연다 (WSL 의 bash 로 열린다).

> 다른 방법: Ubuntu 터미널에서 `cd ~/data_stream && code .` 를 실행해도 같은 창이 열린다.

---

## 2. 실행 순서

모든 명령은 저장소 루트(`~/data_stream`)에서 실행한다. 터미널을 새로 열 때마다 **0) venv 활성화**부터 한다.

### 0) venv 활성화

```bash
cd ~/data_stream
source venv/bin/activate        # 프롬프트 앞에 (venv) 가 붙으면 성공
```

### 1) Redis 확인 (자동으로 켜진다)

Redis(6380)는 사용자 systemd 서비스 `redis-datastream` 으로 **WSL 이 켜질 때 자동으로 시작**한다 (등록 방법은 `배포_가이드.md` 9장).

```bash
redis-cli -p 6380 ping                      # PONG 이면 정상
redis-cli -h 100.125.104.23 -p 6380 ping    # tailscale 쪽도 PONG 이어야 다른 기기가 구독할 수 있다
systemctl --user restart redis-datastream   # 둘 중 하나라도 안 되면 다시 시작
```

Redis 가 꺼져 있어도 스트림은 멈추지 않지만, **파일로만 저장되고** 비전 live 가 동작하지 않는다.

### 2) 데이터 스트림 — `server_stream_multi_source.py` (터미널 1)

```bash
python server_stream_multi_source.py                    # 전체 소스 (cam360 · drone · robot1 · ep-001 · realsense · cam360_slice)
python server_stream_multi_source.py --on drone         # 드론만
python server_stream_multi_source.py --on drone robot1  # 골라 켜기
python server_stream_multi_source.py --drone 100.85.243.54   # 주소를 직접 지정 (기본은 hosts.local.json)
```

pi11(포트 **10008**) · pi12(포트 **10009**) · pi13(포트 **10011**)은 **모듈 카메라**(CSI imx708)다. pi6 의 RealSense 와 달리 `/dev/video*` 가 베이어 RAW 라
`cv2.VideoCapture` 로는 못 받는다 — 파이 쪽에서 `rpicam-vid` 가 MJPEG 로 내고(`picam_work/`),
받는 모양만 로봇과 같아 데이터 스트림 쪽에는 **새 풀러가 없다**.

```bash
python server_stream_multi_source.py --on pi11                 # pi11 만
python server_stream_multi_source.py --on pi11 pi12 pi13       # 모듈 카메라 셋 다
```

셋은 **같은 기기·같은 카메라**(Pi 3 Model B · imx708)이고 같은 `picam_mjpeg.py` 를 쓴다 —
기기를 더할 때 서버에서 늘리는 것은 `SOURCES` · `PICAM_SOURCES` · `DEFAULT_HOST` · `--piNN` 뿐이고
가져오는 코드는 손대지 않는다.

리본 접촉이 나쁘면 **고장이 소프트웨어처럼 보인다**. pi13 이 그랬고, 증상은 두 단계로 나타난다 —
센서가 아예 안 보이거나(`No cameras available`), 보이는데 스트리밍만 실패한다
(`imx708 … Failed to write reg 0x0100. error = -5`). 둘 다 리본을 다시 꽂는 것이 먼저다.
`picam_work/README.md` 에 증상 표가 있다.

| 포트 | 소스 | 받는 곳 |
|---|---|---|
| 10001 | cam360 | 360 노트북이 보내는 업로드 (`/upload_pano`) + 상태 (`/state`: GPS · 배터리 · 온도) |
| 10002 | drone | pi3 `:8890/api/frame?cam=0` + MQTT `:1883` |
| 10003 | robot1 | pi7 `:8090/stream/1` + MQTT `:1883` |
| 10004 | ep-001 | pi1 RoboMaster EP — `ws://…:8091/ws/ep-001`(H.264 → JPEG, 원본은 실행 폴더 `video.h264`) + MQTT `:1883` ep-001 토픽만 (2026-11-24, 옛 robot2 자리) |
| 10005 | realsense | pi6 `:8090/stream/1` (컬러만, 상태 없음 · 11-13 ds5 로 옮겼다가 11-16 pi6 로 되돌림) |
| 10006 | cam360_slice | 같은 노트북의 `e2p/slice360_live.py` 가 cam360 파노라마를 잘라 올림 (`/upload`, 상태 없음) — 5) |
| 10007 | a2 | pi10 `:8090/stream/1` + MQTT `:1883` (a2-bridge) |
| 10008 | pi11 | pi11 `:8090/stream/1` (상태 없음 — pi11 에 브로커가 없다) |
| 10009 | pi12 | pi12 `:8090/stream/1` (상태 없음 — pi12 에 브로커가 없다) |
| 10011 | pi13 | pi13 `:8090/stream/1` (상태 없음 — pi13 에 브로커가 없다). **10010 은 비워 둔다** — 개발 기계에서 다른 프로그램이 쓰고 있었다 |
| 10012 | drone2 | ds5 `:8890/api/frame?cam=0` + MQTT `:1883` (둘째 드론 x500-002 · drone 과 같은 drone-agent) |

- 10초마다 소스별 수신 속도(장/초, 건/초)가 찍힌다. 종료는 `Ctrl+C`.
- 브라우저 확인: `http://localhost:10002/` (영상 + 비전), `/stream` (원본), `/health` (상태 JSON).
- 가시화에서 가져갈 주소 (포트마다 같다):
  - **영상 + 상태 한 번에**: `http://<이 노트북>:<포트>/frame?since=<지난 번호>` — 본문은 최신 JPEG, 헤더 `X-State` 에 그 순간의 최신 상태 JSON,
    `X-Frame-N` 에 번호, `X-Frame-Meta` 에 프레임 헤더. 받은 `X-Frame-N` 을 `since` 로 다시 부르면 새 장이 올 때까지 기다렸다 준다 (없으면 204).
    `&format=json` 이면 `{n, meta, state, image_b64}` JSON 하나로 준다.
  - 영상: `http://<이 노트북>:<포트>/stream` (MJPEG — `<img src>` 에 그대로 넣는다)
  - 상태: `http://<이 노트북>:<포트>/state` (JSON, 1초마다 읽는다). `latest.payload` 가 최신 상태 한 건이다.
    360 은 `gps{lat, lon, alt_m, fix_type, satellites}` · `battery{level_pct, charging}` · `temperature_c{battery, board, module}`.
    `gps.lat`/`lon` 은 fix 가 없으면(`fix_type < 2`) `null` 이다. 드론 · 로봇은 MQTT 로 받은 토픽별 최신 값이 `by_topic` 에 있다.
  - Redis 로 받으려면 `camera_stream:<소스>` (영상) · `state_stream:<소스>` (상태).

### 3) 비전 추론 — `vision/vision_infer.py` (터미널 2, 선택)

```bash
python vision/vision_infer.py                                      # live · 전체 기기 · yoloe
python vision/vision_infer.py --devices drone --models yoloe unidepth   # 드론에 YOLOE + 거리
python vision/vision_infer.py --devices drone --models yoloe --runs latest   # 마지막 저장본에 나중에 적용
python vision/vision_infer.py --list                               # 기기·실행 폴더·프레임 수만 보기
```

- live 모드는 2)와 Redis 가 켜져 있어야 한다. 결과는 브라우저 `http://localhost:<포트>/vision?model=yoloe` 에서 본다.
- 모델: `yoloe`(검출) · `river`(하천 위험물) · `targeted`(지정 클래스 탐지) · `unidepth` · `moge2_aerial`(깊이 m).
  드론은 검출별 GPS 좌표도 붙는다. 검출 모델을 여럿 고르면 모두 돌고, 깊이 결합 · GPS 는 첫 검출 모델에만 붙는다.

#### 지정 클래스 탐지 (`targeted`, 2026-11-23)

대상과 근거를 골라 찾는다. 계획과 설계는 `탐지대상_런처_계획_261123.md`.

```bash
python vision/vision_infer.py --models targeted                         # 기본 대상 6개 · 기본 근거 · 거리 켬
python vision/vision_infer.py --models targeted --target person --target chair --no-distance
python vision/vision_infer.py --models targeted --target "traffic cone:size=300x300x700:colour=orange"
python vision/vision_infer.py --models targeted --target door --cues default,aspect=off --cards always --count
python vision/vision_infer.py --catalog                                 # 고를 수 있는 것 (런처가 쓴다)
python vision/cue_catalog.py --check 프로필.json                        # 프로필 검사
```

- **대상**: `person` · `door` · `chair` · `desk` · `demo_door`(시연 장소 하늘색 문, main 문 탐지 근거 그대로) ·
  `demo_box`(Box_Detection 학습 검출기 v9). 대상 파일 `vision/assets/targets/<이름>.json` 을 더하면 목록에 뜬다.
  목록에 없는 이름은 제로샷으로 찾는다(크기 · 색을 주면 형상 · 크기 거부 · 색상 근거가 붙는다).
- **근거**: 탐지기(텍스트 · prompt-free · OVD 확신의 noisy-OR, 중심 명사가 같은 비슷한 이름은 0.7 배) · 글 설명 ·
  특징 사전 · 형상 · 색상 · CLIP 색 · CLIP 모양 · 채도 · 색상 H · 종횡비 · OVD 짝 IoU · 바닥 위치 · 크기 거부.
  역할은 점수 · 필수(통과 못 하면 판정 안 함, 못 잰 것도 통과 아님) · 거부. 쓸 수 없는 근거는 런처에 안 나온다.
- **거리 측정**을 끄면 UniDepth 를 적재하지 않는다(크기 거부 · 거리 라벨도 꺼진다). `demo_box` 는 Go1 카메라
  (464x400) 프레임에서 박스 높이 식으로 재고, 다른 카메라에서는 관문 판정을 보류하고 UniDepth 거리를 쓴다.
- **GroundingDINO** 는 `demo_door` 를 고를 때만 적재된다(프레임당 약 0.5 초). 스냅샷은 `models/grounding-dino-tiny`.
- 판정 하한은 대부분 측정 전 값이다(결과에 `provisional`). 다른 카메라 · 장소에서는 다시 재야 한다.

#### 오버레이와 웹 화면

- 런처 오버레이 탭 = 모델별 라벨 항목 · 탐지 개수 · 카드(끔 · 마우스 올릴 때 · 항상) · 판정 아닌 검출 등.
  **마우스 올릴 때** 카드는 웹 화면 `http://<주소>:<포트>/vision/view?model=targeted` 에서만 보인다(영상에는 안 들어감).
  주소에 `&cards=off|hover|always` 를 붙이면 그 화면만 바뀐다.
- **구역**: 웹 화면에서 빈 곳을 끌면 그 기기의 탐지를 그 안으로 제한한다(분할 마스크의 50% 이상이 안에 있어야 남는다).
  구역마다 ✕, 위쪽 '구역 모두 지우기'. 창구는 `GET/POST/DELETE /control/roi` (Redis `vision_roi:<소스>`).

#### 바탕화면에 런처 깔기 (2026-11-26)

- `powershell -NoProfile -ExecutionPolicy Bypass -File tools\launcher\install_desktop.ps1`
  → `%USERPROFILE%\DataStreamLauncher\` 에 **이 저장소의 런처 둘**(`launcher.ps1` · `web_launcher.cmd`)을 복사하고
  바탕화면에 바로가기 둘(**데이터 스트림 실행** · **데이터 스트림 웹 런처**)을 만든다. 새 런처를 만들지 않는다.
- 그 기기 값(배포판 · 저장소 자리 · profile.json 자리)은 `launcher.local.json` 에만 적는다 — 저장소 파일은 고치지 않는다.
  다시 깔아도 `settings.json` · `profile.json` 은 그대로다(`-Force` 를 주면 `launcher.local.json` 도 다시 쓴다).
- **실행 바탕은 리눅스다.** 이 저장소는 fcntl 파일 잠금(`state` · 원하는 상태) · `/proc` 읽기 · `setsid`/`killpg` ·
  셸 스크립트(`tools/ds_run.sh` · `ds_stop.sh`) · `nvidia-smi` · `kubectl` 에 기댄다 — Windows 파이썬으로는 돌지 않는다.
  바로가기는 그 바탕을 부르는 얇은 조각이다: WinForms 런처는 `wsl.exe` 로 창을 띄우고, 웹 런처 바로가기는 주소만 연다.
  GPU 는 WSL2 의 CUDA 를 그대로 쓴다(이 기기 실측: RTX 3060 12 GB).
- 네이티브 리눅스 · 컨테이너에서는 바로가기가 필요 없다 — `python -m ds_resources.weblauncher`(+ `ds_resources.supervisor`)
  를 그대로 띄우고 브라우저로 연다. Docker 바탕도 같다(compose 로 띄우고 `web_launcher.cmd` 로 주소만 연다).

#### 바탕화면 「데이터 스트림 실행」 런처 (2026-11-23 · 단계 화면 2026-11-26)

- `%USERPROFILE%\DataStreamLauncher\launcher.ps1` (위 설치 스크립트가 복사한다). 저장소 사본 `tools/launcher/launcher.ps1`
  (UTF-8 BOM — PowerShell 5.1 이 한글을 읽으려면 필요). 고치면 설치 스크립트를 다시 돌려 복사한다.
- **고르는 순서 그대로 다섯 단계다.** 탭을 눌러 바로 가도 되고 아래 **← 이전 · 다음 →** 로 한 단계씩 가도 된다.

  | 단계 | 고르는 것 |
  |---|---|
  | **소스** | 소스 — **지금 닿는 것만 고를 수 있다**(안 닿는 것은 흐려지고 풍선말에 까닭). cam360 옆 「세부 설정」(슬라이스 켜기 · u), robot1 옆 「cam 번호」, 「연결 재확인」, 「소스 추가」 |
  | **기능** | 인지 켜기 · **기능 소스별 배치**(세로 기능, 가로 켠 소스. 기능마다 쓸 모델을 고르고, 소스 칸으로 어느 소스에 올릴지 고른다. 모델이 없는 기능은 아래쪽에 흐려지고 「받기」가 붙는다) · 「추가」(확장 기능 — 객체 재식별 · 추적) · 「모델 추가」 · 「탐지 시험」 · **근거 표**(기능 · 근거 · 역할 · 모델 · 예상 GPU MB) |
  | **지정 클래스** | 대상 · 대상별 근거 · 거리 측정 — **지정 클래스 탐지를 골랐을 때만** 열리고, 「다음」이 아니면 건너뛴다 |
  | **오버레이** | 고른 기능마다 라벨 · 탐지 개수 · 카드 등 (쓸 수 있는 것만). 「오버레이 미리보기」로 저장된 장에 그려 본다 |
  | **배치** | 자원 배분 · 배치 보기 · **자동/수동**. 노드마다 CPU · 메모리 · GPU 막대와 범례(어느 소스의 어느 기능이 얼마나 쓰는지), 노드별 「자원 한도」. 수동이면 작업마다 `자동 · 노드 이름 · 올리지 않음` 중에서 고른다 — **고를 수 있는 기기는 소스에서 켠 것뿐이다** |

- **한국어 / English** — 아래 가운데 상자에서 바꾼다. **화면 글만** 바뀐다. 모델 이름(yoloe) · 근거 이름(탐지기 · 글 설명) ·
  대상 이름(person)은 카탈로그(`vision/cue_catalog.py`)가 주는 그대로다 — 거기까지 옮기려면 비전 쪽에 영문 칸을 더해야 한다.
  고른 언어는 `settings.json` 의 `lang` 에 남아 다시 열면 그대로다.
- 근거 표의 예상 GPU 는 카탈로그의 `gpu_mb_base` · `gpu_mb_option` · `gpu_mb_targeted` 를 그대로 읽은 **어림값**이다
  (실제 필요량은 오케스트레이터가 실측 · 추정으로 다시 잡는다 — 「배치 보기」가 보여 준다).
- **실행 뒤에도 열려 있다.** 바꾸면 1초 뒤 반영: 대상 · 근거 · 거리 · 오버레이는 `profile.json` 만(비전이 실행 중에 바꾼다,
  새 모델은 미리 적재하는 동안 옛 설정으로), 기능 · 켜기 · **기기별 기능** · **수동 자리** → 비전 창만, 소스 · 로봇 캠 → 서버와 비전,
  슬라이스 → 슬라이스 창만. 틀린 지정이면 옛 설정을 유지하고 아래 상태 줄에 이유가 나온다.
- 창마다 이름표(`tools/ds_run.sh`, `/tmp/ds_launcher/`)가 붙는다. ✕ 로 닫으면 확인한 뒤 전부 끈다.
  런처가 비정상으로 꺼져도 창은 남고, 다시 열면 '실행 중' 으로 이어받는다. 손으로 끄기: `tools/ds_stop.sh <이름>`, 목록: `tools/ds_status.sh`.
- **소스 추가** — 소스 탭의 「소스 추가」를 누르면 이름 · 주소 · 카메라 포트 · 경로를 적는 창이 뜬다. 「연결 시험」이 그 주소를
  열어 **JPEG 한 장이 오는지** 보고 크기와 해상도를 적어 준다(실측: pi12 `1280x720, 33599 바이트, 90 ms`). 되는 것만 「넣기」로
  `config/sources.json` 에 적히고, 소스 줄에 바로 나타난다. 기기를 아직 안 켰으면 「기기를 아직 켜지 않았습니다」를 켜고 넣는다.
  같은 일을 글로: `venv/bin/python tools/source_add.py --test --host 100.1.2.3` · `--save --name pi14 --host 100.1.2.3` · `--list` · `--remove pi14`.
  **더한 소스는 데이터 스트림을 다시 시작해야 들어온다** — 서버 포트가 하나 늘어나니(10013부터) 가시화 쪽에도 그 포트를 알려 준다.
- 로봇 캠 번호는 robot1 에만 쓰인다(ep-001 · a2 · pi11~13 은 카메라가 하나). 슬라이스 u 를 비우면 마지막 u, v 는 −30 고정.
- 시험: `$env:DS_LAUNCHER_SELFTEST=1` 로 실행하면 창 없이 선택을 코드로 바꿔 보고 결과를 JSON 으로 찍는다
  (프로필 · 단계 이동 · 한영 전환 · 기기별 기능 · 수동 자리 · 컨트롤이 화면 밖으로 나가지 않는지까지).

#### 웹 런처 (2026-11-25, Docker 계획 D7)

- `python -m ds_resources.weblauncher [--host 127.0.0.1] [--port 10080] [--redis redis://127.0.0.1:6380/0] [--state-dir state]`
  → 브라우저로 `http://localhost:10080/`. 화면과 규칙은 위 런처와 같다(탭 셋 · 1초 뒤 반영 · 배치 보기 · 상태 줄).
- **원하는 상태(`ds_desired` · `state/desired.json`)만 쓴다.** 프로세스를 띄우고 끄는 것은 감독자(`ds_resources.supervisor`)다.
  상태 줄은 감독자 보고(`ds_reported:<역할>`)의 `converged` · 구성 요소 상태 · 이유(「반영 안 됨」)를 보여 준다.
- 화면 선택은 원하는 상태의 `ui` 칸(런처 settings.json 과 같은 모양)에 남아 다시 열면 그대로다.
- **인증이 없다.** 기본은 127.0.0.1 에만 연다. `--host 0.0.0.0` 은 그 망의 누구나 실행 · 끄기를 할 수 있다 — 테일넷처럼 믿는 망에서만.
  인증이 없으므로 `Host` 헤더를 검사한다: `localhost` · `127.0.0.1` · `::1` · 이 기기 이름 · `DS_WEB_ALLOWED_HOSTS`(쉼표 구분)
  만 받고 그 밖은 403(포트는 무시). 아무 웹페이지가 `POST /api/plan` · `/api/register_target` 을 부르는 것과 DNS rebinding 을 막는다.
  테일넷 이름 · 컨테이너 이름으로 열려면 `DS_WEB_ALLOWED_HOSTS=wsl.tail1234.ts.net` 처럼 넣는다.
- Windows 바로가기: `tools/launcher/web_launcher.cmd [URL]` — **브라우저만 연다**(주소: 인자 → `DS_WEB_LAUNCHER_URL` → `http://localhost:10080/`).
  서버(`python -m ds_resources.weblauncher`)와 감독자(`python -m ds_resources.supervisor --role all`)는 따로 떠 있어야 한다 —
  웹 런처는 원하는 상태만 쓰고, 실제로 띄우는 것은 감독자다.
- 탭 다섯 — 스트림 · 지정 클래스 · 오버레이 · 모델 매칭 · 수동 배치. WinForms 런처와 같은 규칙이다.
  수동 배치 탭은 `GET /api/places` (노드 · 놓을 수 있는 작업)를 읽어 고르게 한다.

#### 스캔 탐지 창구 `/scan/*` (2026-11-24)

- 가시화가 장치를 45° 씩 돌리며 쓰는 창구. 그 장치 소스의 서버 포트(`go1-001` → robot1 `:10003`, `ep-001` → ep-001 `:10004`,
  `a2-001` → a2 `:10007`)에 `prepare → mark × 8 → finish → result`. 돌고 있는 `targeted` 결과에서 각도마다 표시 0.5초 뒤 한 장을 잡아
  **그 장의 값**으로 판정 · 방위(스캔 시작 기준 −180~180) · 거리를 낸다. 규약은 `스캔탐지창구_요청_VZ_261123.md`, 설계는 `스캔탐지_창구_계획_261124.md`.
- 그 기기의 `targeted` 비전이 돌고 있어야 한다(아니면 `no_vision`). 판 동안 그 비전 프로세스의 대상은 판 대상 하나뿐이고 거리를 매 장 잰다.
  판이 끝나면(또는 서버가 죽으면 20초 안에) 원래 대상으로 돌아간다. 판 동안 런처에서 바꾼 것은 끝난 뒤 반영된다(상태 줄 「스캔 중 — 끝나면 반영」).
- 판 기록: `<--save>/scans/<scan_id>/` (`result.json` · 각도별 `original` · `overlay` 그림).
- 시험 한 판(로봇 없이, 돌지 않고): `venv/bin/python tools/scan_smoke.py --target door` (`--target object --query chair` 등).
- 시험: `test_scan_desk.py`(저장소 루트) · `vision/test_scan_assess.py` · `vision/test_scan_vision.py`.

### 4) 360 카메라 — `360camera_work/stream_to_server.py` (360 카메라 노트북)

360 카메라 Wi-Fi 에 연결된 노트북에서 실행한다 (필요 패키지: `requests`, `opencv-python`).

```bash
python stream_to_server.py --dry-run --count 20       # 업로드 없이 카메라 스트림만 점검
python stream_to_server.py --upload http://100.125.104.23:10001/upload_pano   # 데이터 스트림 노트북으로 보내기
python stream_to_server.py --upload http://127.0.0.1:10001/upload_pano       # 데이터 스트림이 같은 노트북에서 돌 때
python stream_to_server.py --fps 3                    # 기본 5fps → 3fps (인자 없이 실행하면 7866 · 10001 서버 두 곳으로 보낸다)
```

- `100.125.104.23` 은 이 노트북의 tailscale IP 다 (바뀌면 `tailscale ip -4` 로 확인). 360 노트북도 같은 tailnet 에 로그인되어 있어야 한다.
- 카메라 IP 기본값은 `192.168.100.124` 이고, 다르면 `--ip` 로 바꾼다. 종료는 `Ctrl+C`.
- GPS · 배터리 · 온도를 1초마다 `/upload_pano` 와 같은 서버의 `/state` 로 보낸다 (`--telemetry 2` 로 주기 변경, `0` 이면 끔).
  7866 pano_receiver 처럼 `/state` 가 없는 곳은 한 번 알리고 건너뛴다. 실내에서는 위치가 안 잡혀 `gps.lat` 이 `null` 이다.

### 5) 360 슬라이스 — `e2p/sweep_latest.py` · `e2p/slice360_live.py` (데이터 스트림 노트북, 선택)

360 파노라마(10001)를 정한 방향의 90도 원근 사진으로 잘라 **포트 10006(`cam360_slice`)** 으로 내놓는다.
파노라마 원본은 그대로 남는다. 자르기는 데이터 스트림 밖의 따로 도는 파일이 하므로, 이 파일이 죽어도 받기·저장은 계속된다.
처음 한 번 `pip install py360convert` (requirements.txt 에 있다).

```bash
# ⓪ 데이터 스트림을 띄우기 전에 u 고르기 (바탕화면 「e2p sweep」 바로가기와 같다)
python e2p/sweep_camera.py                   # 360 카메라 지금 화면 한 장 → 원본 + u 24장(v -30) 한눈 그림을 사진 앱으로 연다
                                             # 「360 카메라 실행」이 카메라를 쥐고 있으면 데이터 스트림의 10초 안 최신 장을 쓴다

# ① 각도 고르기: 지금 들어오는 파노라마를 15도 간격 24장으로 잘라 본다 (2)·4) 가 돌고 있을 때)
python e2p/sweep_latest.py --v 0 -15 -30     # → stream_data/cam360/<실행>/sweep/…/_sheet_v*.jpg 부터 본다
python e2p/sweep_latest.py --from disk       # 데이터 스트림이 꺼져 있으면 마지막 저장본으로

# ② 실시간 슬라이스 (터미널 하나를 계속 쓴다)
python e2p/slice360_live.py --u 150 --v -35  # u: 0 = 파노라마 가운데, + 오른쪽 · v: 0 = 수평, - 아래
python e2p/slice360_live.py --set 165 -35    # 다른 터미널에서: 돌고 있는 것의 각도 바꾸기 (다음 장부터)
python e2p/slice360_live.py --show           # 지금 각도 (e2p/slice_angle.json)

# ③ 슬라이스에 박스 검출 (선택)
python vision/vision_infer.py --devices cam360_slice drone
```

- `--on` 으로 소스를 고를 때는 `cam360_slice`(또는 `slice`)를 같이 넣는다. 빠지면 slice360_live 에 「올리기 실패 … 10006」이 뜬다.
- 확인: 데이터 스트림 10초 보고의 `cam360_slice 영상 N장/초` 가 cam360 과 비슷하면 된다. 브라우저 `http://localhost:10006/`.
- 기본 크기 960×960 (장당 약 130KB — 파노라마 약 1MB 의 1/8 이라 가시화 망 부담이 적다). `--width` 로 바꾼다.
- 장마다 meta 의 `slice` 칸에 어느 파노라마(`source_n`)를 어떤 각도(`u_deg` · `v_deg` · `fov_deg`)로 잘랐는지 남는다.
  `timestamp` 는 원본 파노라마를 찍은 시각 그대로다.
- 가시화 연결 관리: 「카메라」에 `http://<이 노트북>:10006/stream`, 「객체 탐지 추론 스트림」 포트에 `10006` (직접 입력).
  다른 기기에서 공유기 IP 로 들어오면 portproxy · 방화벽에 10006 도 연다.

---

## 3. 저장된 데이터 확인

### 저장 구조

```
stream_data/<소스>/<실행 시작 YYYYMMDD_HHMMSS>/
├── frames/000000001.jpg …        받은 영상 (JPEG 그대로)
├── frames.jsonl                  영상 메타 — 한 장 = 한 줄 (frame_file · received_at · sender)
├── states.jsonl                  상태 — 한 건 = 한 줄 (topic · payload: GPS · 고도 · 자세 · 배터리 등)
└── vision/<모델>/                 비전 결과 (results.jsonl · overlay/*.jpg · depth/*.npy)
```

실행할 때마다 새 폴더가 생기고, 지우거나 덮어쓰지 않는다.

### 빠르게 보는 명령

```bash
cd ~/data_stream/stream_data
ls -t drone | head                                         # 최근 실행 폴더
R=drone/$(ls -t drone | head -1)                           # 가장 최근 드론 실행
ls $R/frames | wc -l; wc -l $R/states.jsonl                # 영상 장수 · 상태 건수
head -n 1 $R/frames.jsonl | python -m json.tool            # 영상 메타 한 줄
grep -m1 '/state"' $R/states.jsonl | python -m json.tool   # 드론 상태 한 건 (gps · altitude · attitude)
explorer.exe "$(wslpath -w $R/frames)"                     # Windows 탐색기로 이미지 열기
```

- 비전 결과 목록: `python vision/vision_infer.py --list` (루트에서, venv 상태)
- Redis 최근 항목 수: `redis-cli -p 6380 XLEN camera_stream:drone`
