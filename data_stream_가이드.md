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
python server_stream_multi_source.py                    # 전체 소스 (cam360 · drone · robot1 · robot2 · realsense · cam360_slice)
python server_stream_multi_source.py --on drone         # 드론만
python server_stream_multi_source.py --on drone robot1  # 골라 켜기
python server_stream_multi_source.py --drone 100.85.243.54   # 주소를 직접 지정 (기본은 hosts.local.json)
```

pi11(포트 **10008**) · pi12(포트 **10009**)는 **모듈 카메라**(CSI imx708)다. pi6 의 RealSense 와 달리 `/dev/video*` 가 베이어 RAW 라
`cv2.VideoCapture` 로는 못 받는다 — 파이 쪽에서 `rpicam-vid` 가 MJPEG 로 내고(`picam_work/`),
받는 모양만 로봇과 같아 데이터 스트림 쪽에는 **새 풀러가 없다**.

```bash
python server_stream_multi_source.py --on pi11            # pi11 만
python server_stream_multi_source.py --on pi11 pi12       # 모듈 카메라 둘 다
```

pi12 는 pi11 과 **같은 기기·같은 카메라**(Pi 3 Model B · imx708)이고 같은 `picam_mjpeg.py` 를 쓴다.

| 포트 | 소스 | 받는 곳 |
|---|---|---|
| 10001 | cam360 | 360 노트북이 보내는 업로드 (`/upload_pano`) + 상태 (`/state`: GPS · 배터리 · 온도) |
| 10002 | drone | pi3 `:8890/api/frame?cam=0` + MQTT `:1883` |
| 10003 | robot1 | pi7 `:8090/stream/1` + MQTT `:1883` |
| 10004 | robot2 | pi1 `:8090/stream/1` + MQTT `:1883` |
| 10005 | realsense | pi6 `:8090/stream/1` (컬러만, 상태 없음) |
| 10006 | cam360_slice | 같은 노트북의 `e2p/slice360_live.py` 가 cam360 파노라마를 잘라 올림 (`/upload`, 상태 없음) — 5) |
| 10007 | a2 | pi10 `:8090/stream/1` + MQTT `:1883` (a2-bridge) |
| 10008 | pi11 | pi11 `:8090/stream/1` (상태 없음 — pi11 에 브로커가 없다) |
| 10009 | pi12 | pi12 `:8090/stream/1` (상태 없음 — pi12 에 브로커가 없다) |

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
- 모델: `yoloe`(검출) · `unidepth` · `moge2_aerial`(깊이 m). 드론은 검출별 GPS 좌표도 붙는다.

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
