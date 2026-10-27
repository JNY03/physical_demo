# data_stream 사용 가이드

로봇 2대·드론·360 카메라의 영상과 상태를 받아 **Redis 로 흘리고 디스크에 저장**하는 프로그램과, 그 프레임에 비전 모델을 돌리는 프로그램의 사용법이다.

---

## 1. 경로와 접속

### 1-1. 폴더 위치 (WSL Ubuntu 안)

```
~/physical_demo/                            ← 저장소 루트 (명령은 여기서 실행)
├── venv/                                   ← 파이썬 가상환경 (redis · paho-mqtt · torch · ultralytics 설치됨)
├── server_stream_multi_source.py           ← 데이터 스트림 (받기 · Redis · 저장)
├── hosts.local.json                        ← 로봇·드론 주소 (tailscale IP)
├── vision/
│   ├── vision_infer.py                     ← 비전 추론 (YOLOE · UniDepth · MoGe2)
│   └── vision_config.yaml                  ← 비전 설정
├── 360camera_work/
│   └── stream_to_server.py                 ← 360 카메라 → 서버 업로드
├── models/  external/                      ← 모델 가중치 · UniDepth · AerialMetric 코드 (깃에 없음)
└── stream_data/                            ← 저장된 데이터 (아래 3장)
```

Windows 탐색기에서는 주소창에 `\\wsl$\Ubuntu\home\<WSL 사용자>\physical_demo` 을 입력하면 열린다.

### 1-2. VS Code 로 WSL 접속

1. Windows 에 VS Code 를 설치하고 (설치·초기 설정 전체는 `배포_가이드.md`), 확장(Extensions, `Ctrl+Shift+X`)에서 **WSL** (Microsoft) 을 설치한다.
2. `F1` → **`WSL: Connect to WSL`** (또는 왼쪽 아래 `><` 초록 버튼 → *Connect to WSL*).
   배포판이 여러 개면 **`WSL: Connect to WSL using Distro...`** → `Ubuntu` 를 고른다.
3. 새 창의 왼쪽 아래에 `WSL: Ubuntu` 가 보이면 접속된 것이다.
4. **File → Open Folder** → `~/physical_demo` (`/home/<WSL 사용자>/physical_demo`) 를 연다.
5. 터미널은 `` Ctrl+` `` 로 연다 (WSL 의 bash 로 열린다).

> 다른 방법: Ubuntu 터미널에서 `cd ~/physical_demo && code .` 를 실행해도 같은 창이 열린다.

---

## 2. 실행 순서

모든 명령은 저장소 루트(`~/physical_demo`)에서 실행한다. 터미널을 새로 열 때마다 **0) venv 활성화**부터 한다.

### 0) venv 활성화

```bash
cd ~/physical_demo
source venv/bin/activate        # 프롬프트 앞에 (venv) 가 붙으면 성공
```

### 1) Redis 확인

`배포_가이드.md` 9장대로 `redis-datastream` 서비스를 등록했으면 **WSL 이 켜질 때 자동으로 시작**한다. 확인만 한다:

```bash
redis-cli -p 6380 ping                                        # PONG 이면 정상
redis-cli -h <데이터 스트림 노트북 tailscale IP> -p 6380 ping    # 다른 기기가 구독하려면 이쪽도 PONG
systemctl --user restart redis-datastream                     # 안 되면 다시 시작
```

서비스를 등록하지 않았으면 WSL 을 켤 때마다 직접 띄운다:

```bash
redis-server --port 6380 --bind "127.0.0.1 <데이터 스트림 노트북 tailscale IP>" --protected-mode no \
  --save "" --appendonly no --daemonize yes
```

Redis 가 꺼져 있어도 스트림은 멈추지 않지만, **파일로만 저장되고** 비전 live 가 동작하지 않는다.

### 2) 데이터 스트림 — `server_stream_multi_source.py` (터미널 1)

```bash
python server_stream_multi_source.py                    # 전체 소스 (cam360 · drone · robot1 · robot2)
python server_stream_multi_source.py --on drone         # 드론만
python server_stream_multi_source.py --on drone robot1  # 골라 켜기
python server_stream_multi_source.py --drone <드론 pi tailscale IP>   # 주소를 직접 지정 (기본은 hosts.local.json)
```

| 포트 | 소스 | 받는 곳 |
|---|---|---|
| 10001 | cam360 | 360 노트북이 보내는 업로드 (`/upload_pano`) |
| 10002 | drone | pi3 `:8890/api/frame?cam=0` + MQTT `:1883` |
| 10003 | robot1 | pi7 `:8090/stream/1` + MQTT `:1883` |
| 10004 | robot2 | pi1 `:8090/stream/1` + MQTT `:1883` |

- 10초마다 소스별 수신 속도(장/초, 건/초)가 찍힌다. 종료는 `Ctrl+C`.
- 브라우저 확인: `http://localhost:10002/` (영상 + 비전), `/stream` (원본), `/health` (상태 JSON).

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
python stream_to_server.py --upload http://<데이터 스트림 노트북 tailscale IP>:10001/upload_pano   # 데이터 스트림 노트북으로 보내기
python stream_to_server.py --upload http://127.0.0.1:10001/upload_pano       # 데이터 스트림이 같은 노트북에서 돌 때
python stream_to_server.py --fps 3                    # 기본 5fps → 3fps (인자 없이 실행하면 7866 · 10001 서버 두 곳으로 보낸다)
```

- `<데이터 스트림 노트북 tailscale IP>` 는 그 노트북에서 `tailscale ip -4` 로 확인한다. 360 노트북도 같은 tailnet 에 로그인되어 있어야 한다.
- 카메라 IP 기본값은 `192.168.100.124` 이고, 다르면 `--ip` 로 바꾼다. 종료는 `Ctrl+C`.

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
cd ~/physical_demo/stream_data
ls -t drone | head                                         # 최근 실행 폴더
R=drone/$(ls -t drone | head -1)                           # 가장 최근 드론 실행
ls $R/frames | wc -l; wc -l $R/states.jsonl                # 영상 장수 · 상태 건수
head -n 1 $R/frames.jsonl | python -m json.tool            # 영상 메타 한 줄
grep -m1 '/state"' $R/states.jsonl | python -m json.tool   # 드론 상태 한 건 (gps · altitude · attitude)
explorer.exe "$(wslpath -w $R/frames)"                     # Windows 탐색기로 이미지 열기
```

- 비전 결과 목록: `python vision/vision_infer.py --list` (루트에서, venv 상태)
- Redis 최근 항목 수: `redis-cli -p 6380 XLEN camera_stream:drone`
